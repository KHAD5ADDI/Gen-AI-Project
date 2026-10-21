"""Outils simples appelés par la boucle ReAct."""

import ast
import operator
from datetime import date
from functools import lru_cache

import pandas as pd

from rag_assistant.config import CORPUS_PATH

MAX_CHARS = 6000
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression):
    """Calcule une expression arithmétique sans exécuter de code."""
    try:
        tree = ast.parse(expression.replace("%", ""), mode="eval")
    except SyntaxError as err:
        raise ValueError("expression invalide") from err

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("expression interdite")

    value = evaluate(tree.body)
    return f"{round(value, 6):g}"


def days_between(arguments):
    """Renvoie le nombre de jours entre deux dates ISO séparées par une virgule."""
    first, second = arguments.split(",", 1)
    return str((date.fromisoformat(second.strip()) - date.fromisoformat(first.strip())).days)


@lru_cache(maxsize=1)
def load_texts():
    """Lit une seule fois les textes complets du corpus."""
    rows = pd.read_parquet(CORPUS_PATH, columns=["doc_id", "text"])
    return dict(zip(rows["doc_id"], rows["text"]))


def make_tools(retriever, seen, theme=None, since=None):
    """Crée les outils et garde l'ordre des fiches réellement consultées."""

    def remember(doc_id):
        if doc_id not in seen:
            seen.append(doc_id)

    def search(query):
        if not query.strip():
            raise ValueError("requête vide")
        blocks = []
        chunks = (retriever.search(query) if theme is None and since is None
                  else retriever.search(query, theme=theme, since=since))
        for chunk in chunks:
            doc_id = chunk["doc_id"]
            remember(doc_id)
            section = f" > {chunk['section']}" if chunk.get("section") else ""
            blocks.append(
                f"[{doc_id}] ({chunk['date']}) {chunk['title']}{section} : {chunk.get('body', '')}"
            )
        return "\n\n".join(blocks) if blocks else "Aucun passage trouvé."

    def get_document(doc_id):
        doc_id = doc_id.strip()
        texts = load_texts()
        if doc_id not in texts:
            raise ValueError(f"fiche inconnue : {doc_id}")
        remember(doc_id)
        text = texts[doc_id]
        if len(text) > MAX_CHARS:
            note = f"[Fiche tronquée : 6 000 caractères affichés sur {len(text)}. Utilise search pour trouver un passage précis.]"
            return f"{text[:MAX_CHARS]}\n{note}"
        return text

    return {
        "search": search,
        "get_document": get_document,
        "calculator": calculator,
        "days_between": days_between,
    }
