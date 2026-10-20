"""Résultat d'évaluation après un échec du modèle."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from run_eval import error_report, report

from rag_assistant.schemas import Reponse
from rag_assistant.tracing import Trace


def test_erreur_garde_le_format_du_resultat_normal():
    question = {"id": "q001", "type": "simple", "split": "dev", "question": "Question", "answer": "Réponse"}
    llm = SimpleNamespace(provider="ollama", name="faux")
    normal = {
        "reponse": Reponse(answer="Réponse", citations=["F1"]),
        "retrieved": ["F1"], "status": "done", "iterations": 1, "invalid_calls": 0,
        "steps": [], "tokens_in": 10, "tokens_out": 5, "seconds": 0.2,
    }
    trace = Trace()
    trace.record(normal)

    item = error_report(question, RuntimeError("indisponible"), 1.234, llm)

    assert list(item) == list(report(question, normal, trace, 0.2, llm))
    assert item["status"] == "erreur"
    assert item["answerable"] is False
    assert item["generated_answer"] == "Erreur d'appel au modèle : RuntimeError"
    assert item["latency_sec"] == 1.234
    assert item["steps"] == item["tool_calls"] == item["cited_sources"] == item["retrieved_document_IDs"] == []
    assert item["tokens_in"] == item["tokens_out"] == item["cost_usd"] == 0
