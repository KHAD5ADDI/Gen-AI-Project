"""Prépare la vérification manuelle du golden set.

    uv run python scripts/verify_golden.py

Pour chaque question, le script écrit dans eval/golden_review.md :
- la question, la réponse attendue et l'URL de la fiche, pour la comparer avec le site ;
- les passages de la fiche qui contiennent les éléments clés de la réponse (chiffres, sigles,
  numéros cerfa...), avec le titre de leur section, pour vérifier la réponse ET la situation ;
- pour les questions sans gold_docs, les fiches qui ressemblent le plus à la question, pour
  vérifier qu'aucune ne contient la réponse.
Il signale aussi les identifiants absents du corpus. Il ne juge pas : c'est à vous de cocher.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from rag_assistant.config import CORPUS_PATH, GOLDEN_PATH, ROOT

OUT = ROOT / "eval" / "golden_review.md"
MAX_HITS = 6  # passages affichés par fiche

# Nombre suivi éventuellement de son unité : « 3 mois », « 30 000 € », « 16703 ».
NUMBER = re.compile(r"\d+(?:[ \u202f.,]\d{3})*(?:,\d+)?(?:\s?(?:€|%|mois|jours?|ans?|m²|heures?))?")
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9+]{1,}\b")
STOP = {"quelle", "quelles", "quel", "quels", "faut", "peut", "pour", "dans", "avec", "sans",
        "est", "une", "des", "les", "doit", "combien", "comment", "qui", "que", "sur", "son",
        "être", "après", "avant", "temps", "elle", "votre", "vous", "faire", "demande"}


def keywords(answer: str) -> list[str]:
    """Éléments à retrouver dans la fiche : chiffres et sigles, sinon les mots les plus longs."""
    found = []
    for n in NUMBER.findall(answer):
        n = " ".join(n.split())
        # Un nombre seul et court (« 3 », « 03 ») apparaît partout : on ne garde que les
        # nombres avec unité, ou d'au moins 3 chiffres (numéros cerfa, années, montants).
        if " " in n or not n.isdigit() or len(n) >= 3:
            found.append(n)
    found += [a for a in ACRONYM.findall(answer) if a not in {"L", "A"}]
    if not found:
        words = sorted({w for w in re.findall(r"[a-zà-ÿ]{4,}", answer.lower()) if w not in STOP},
                       key=len, reverse=True)
        found = words[:3]
    return list(dict.fromkeys(found))


def passages(text: str, keys: list[str]) -> list[tuple[str, str]]:
    """(titre de section, ligne) pour chaque ligne contenant au moins un mot-clé."""
    section, hits = "(introduction)", []
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:]
            continue
        if any(k.lower() in line.lower() for k in keys):
            hits.append((section, line.strip()))
    return hits


def closest(df: pd.DataFrame, question: str, n: int = 3) -> list[tuple[str, str, int]]:
    """Fiches contenant le plus de mots de la question (indice pour les questions hors corpus)."""
    words = [w for w in re.findall(r"[a-zà-ÿ]{5,}", question.lower()) if w not in STOP]
    if not words:
        return []
    scores = df["text"].str.lower().apply(lambda t: sum(w in t for w in words))
    best = scores.sort_values(ascending=False).head(n)
    return [(df.at[i, "doc_id"], df.at[i, "title"], int(s)) for i, s in best.items() if s > 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    df = pd.read_parquet(CORPUS_PATH)
    by_id = df.set_index("doc_id")
    questions = [json.loads(l) for l in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    out = [f"# Vérification manuelle du golden set ({len(questions)} questions)\n",
           ("Pour chaque question : vérifier que la réponse attendue est exacte, qu'elle correspond "
            "à la situation décrite par la question (titre de section), et cocher la case.\n")]
    missing = []
    for q in questions:
        out.append(f"\n---\n\n## {q['id']}: {q['question']}\n")
        out.append(f"**Réponse attendue :** {q['answer']}\n")
        if not q["gold_docs"]:
            out.append("**gold_docs vide** : vérifier qu'aucune fiche ne contient la réponse.\n")
            near = closest(df, q["question"])
            if near:
                out.append("Fiches les plus proches (à survoler) :")
                out += [f"- {d} · {t} ({s} mots communs)" for d, t, s in near]
            out.append("\n- [ ] Vérifié : la réponse n'est pas dans le corpus\n")
            continue
        keys = keywords(q["answer"])
        for doc_id in q["gold_docs"]:
            if doc_id not in by_id.index:
                missing.append((q["id"], doc_id))
                out.append(f"**⚠ {doc_id} absent du corpus.**\n")
                continue
            row = by_id.loc[doc_id]
            out.append(f"**{doc_id} · {row['title']}**  \n{row['url']}\n")
            hits = passages(row["text"], keys)
            out.append(f"Éléments cherchés : {', '.join(keys)}\n")
            if not hits:
                out.append("**⚠ Aucun passage trouvé avec ces éléments : vérifier sur le site.**\n")
            for section, line in hits[:MAX_HITS]:
                out.append(f"- *{section}*  \n  {line}")
            if len(hits) > MAX_HITS:
                out.append(f"- … {len(hits) - MAX_HITS} autres passages (voir `check_corpus.py --show {doc_id}`)")
        out.append("\nAutres fiches contenant tous les éléments clés de la réponse :")
        for _, row in df.iterrows():
            if row["doc_id"] in q["gold_docs"] or not keys:
                continue
            if all(k.lower() in row["text"].lower() for k in keys):
                hits = passages(row["text"], keys)
                if hits:
                    section, line = hits[0]
                else:
                    section = "(titre de section)"
                    line = next(line[3:] for line in row["text"].splitlines()
                                if line.startswith("## ") and any(k.lower() in line.lower() for k in keys))
                out.append(f"- {row['doc_id']} · {row['title']}  \n  *{section}* : {line}")
        out.append("\n- [ ] Vérifié : réponse exacte et adaptée à la situation de la question\n")

    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"{len(questions)} questions → {args.out}")
    if missing:
        print("⚠ Identifiants absents du corpus :", missing)
    else:
        print("Tous les identifiants gold_docs existent dans le corpus.")


if __name__ == "__main__":
    main()
