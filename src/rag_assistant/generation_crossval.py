"""Pipeline avec validation croisée : un second appel LLM vérifie la cohérence."""
from llama_index.core.llms import ChatMessage
from .tracing import Trace, chat
from .generation import ROLE, RULES, SYSTEM_PROMPT, Reponse, REFUS_TECHNIQUE, format_passages, parse_and_validate, extract_json

VERIFIER_PROMPT = """Tu es un vérificateur de réponses RAG. On t'a donné des passages source et une réponse générée.
Réponds uniquement avec un JSON : {"coherent": true/false, "reason": "explication courte"}
- coherent: true dans TOUS les cas sauf si la réponse invente un chiffre précis (montant, délai, pourcentage) absent des passages, ou cite une organisation/loi/formulaire inexistant dans les passages
- coherent: false UNIQUEMENT pour des hallucinations factuelles claires (chiffre inventé, source fictive)
- Les différences de formulation, les résumés partiels et les omissions ne rendent PAS une réponse incohérente"""


def verify_response(llm, question: str, chunks, reponse: Reponse, trace: Trace) -> bool:
    passages_text = format_passages(chunks)
    content = (
        f"<passages>\n{passages_text}\n</passages>\n\n"
        f"<question>{question}</question>\n\n"
        f"<réponse_générée>{reponse.answer}</réponse_générée>\n"
        f"<citations_déclarées>{', '.join(reponse.citations)}</citations_déclarées>"
    )
    messages = [
        ChatMessage(role="system", content=VERIFIER_PROMPT),
        ChatMessage(role="user", content=content),
    ]
    try:
        response = chat(llm, messages, trace, purpose="crossval_verify")
        import json
        result = json.loads(extract_json(response.message.content))
        coherent = result.get("coherent", False)
        if not coherent:
            trace.errors.append(f"crossval: {result.get('reason', 'incohérence détectée')}")
        return coherent
    except Exception as e:
        trace.errors.append(f"crossval_error: {e}")
        return True  # en cas d'erreur du vérificateur, on garde la réponse


def generate_with_crossval(llm, question: str, chunks, trace: Trace, max_attempts: int = 2) -> Reponse:
    allowed = {c["doc_id"] for c in chunks}
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"<passages>\n{format_passages(chunks)}\n</passages>\n\n<question>{question}</question>"),
    ]
    reponse = None
    for attempt in range(max_attempts):
        response = chat(llm, messages, trace, purpose=f"crossval_gen_{attempt}")
        try:
            reponse = parse_and_validate(response.message.content, allowed)
            break
        except ValueError as err:
            trace.errors.append(f"crossval_gen_{attempt}: {err}")
            messages += [
                response.message,
                ChatMessage(role="user", content=f"Sortie invalide ({err}). Renvoie uniquement le JSON demandé."),
            ]

    if reponse is None:
        return REFUS_TECHNIQUE.model_copy()

    if not reponse.answerable:
        return reponse

    coherent = verify_response(llm, question, chunks, reponse, trace)
    if not coherent:
        return Reponse(
            answer="La réponse générée n'est pas cohérente avec les sources. Je ne peux pas garantir sa fiabilité.",
            citations=[],
            answerable=False,
        )

    return reponse
