"""Routes de l'interface web, sans index ni modèle."""

from types import SimpleNamespace

import app_html
from rag_assistant.schemas import Reponse


def test_question_refusee_tant_que_le_systeme_charge():
    app_html.state.update(ready=False, llm=None, retriever=None, error=None)
    client = app_html.app.test_client()

    assert client.get("/status").get_json()["ready"] is False
    assert client.post("/ask", json={"question": "Bonjour"}).status_code == 503


def test_fiches_du_corpus():
    client = app_html.app.test_client()

    assert client.get("/api/doc/F1142").get_json()["title"].startswith("Diagnostic immobilier")
    assert client.get("/api/doc/abc").status_code == 400
    assert client.get("/api/doc/F0").status_code == 404
    assert client.get("/api/docs?q=plomb").get_json()["total"] > 0


def test_reponse_avec_trace(monkeypatch):
    def faux_agent(llm, retriever, question, trace):
        trace.record({
            "steps": [{"text": "Thought: je cherche", "action": ["search", "plomb"], "observation": "[F1142] ...",
                       "seconds": 0.5, "tool_seconds": 0.1, "output_chars": 12}],
            "status": "done", "tokens_in": 100, "tokens_out": 20, "seconds": 0.5,
        })
        return Reponse(answer="Oui.", citations=["F1142"]), ["F1142"]

    monkeypatch.setattr("rag_assistant.agent.run_agent", faux_agent)
    app_html.state.update(ready=True, llm=SimpleNamespace(provider="ollama", name="llama3.1:8b"), retriever=None)
    data = app_html.app.test_client().post("/ask", json={"question": "Plomb ?"}).get_json()

    assert data["answer"] == "Oui." and data["citations"] == ["F1142"]
    assert data["status"] == "done" and data["tokens_in"] == 100
    assert data["steps"][0]["tool"] == "search" and data["steps"][0]["observation"] == "[F1142] ..."
    assert data["cost_usd"] > 0
