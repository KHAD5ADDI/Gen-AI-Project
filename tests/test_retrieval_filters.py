"""Filtres de recherche sans index, modèle ni réseau."""

from types import SimpleNamespace

import pytest

import app_html
from rag_assistant.retrieval import HybridRetriever, filtre_chroma, garde
from rag_assistant.schemas import Reponse
from rag_assistant.tools import make_tools


@pytest.mark.parametrize(("theme", "since", "attendu"), [
    (None, None, None),
    ("Logement", None, {"theme": "Logement"}),
    (None, 20260101, {"date_num": {"$gte": 20260101}}),
    ("Logement", 20260101, {"$and": [
        {"theme": "Logement"}, {"date_num": {"$gte": 20260101}},
    ]}),
])
def test_filtre_chroma(theme, since, attendu):
    assert filtre_chroma(theme, since) == attendu


@pytest.mark.parametrize(("chunk", "theme", "since", "attendu"), [
    ({"theme": "Logement", "date_num": 20260101}, None, None, True),
    ({"theme": "Logement", "date_num": 20260101}, "Logement", 20260101, True),
    ({"theme": "Logement", "date_num": 20251231}, "Logement", 20260101, False),
    ({"theme": "Logement", "date_num": 20260101}, "Papiers - Citoyenneté - Élections", None, False),
])
def test_garde(chunk, theme, since, attendu):
    assert garde(chunk, theme, since) is attendu


def test_bm25_classe_uniquement_les_passages_filtres():
    ret = object.__new__(HybridRetriever)
    ret.chunks_list = [
        {"chunk_id": "F1#0", "theme": "Papiers - Citoyenneté - Élections", "date_num": 20260201},
        {"chunk_id": "F2#0", "theme": "Logement", "date_num": 20250101},
        {"chunk_id": "F3#0", "theme": "Logement", "date_num": 20260101},
    ]
    ret.docstore = {chunk["chunk_id"]: chunk for chunk in ret.chunks_list}
    ret.tokens = [["passeport"], ["loyer"], ["loyer"]]
    ret.bm25 = SimpleNamespace(get_scores=lambda tokens: [3, 2, 1])

    assert ret.bm25_search("loyer", top_n=2) == ["F1#0", "F2#0"]
    assert ret.bm25_search("loyer", top_n=2, theme="Logement", since=20260101) == ["F3#0"]


def test_outil_search_transmet_les_filtres():
    class FauxRetriever:
        def search(self, query, theme=None, since=None):
            assert (query, theme, since) == ("loyer", "Logement", 20260101)
            return [{"doc_id": "F3", "date": "2026-01-01", "title": "Loyer", "body": "Texte"}]

    seen = []
    texte = make_tools(FauxRetriever(), seen, theme="Logement", since=20260101)["search"]("loyer")
    assert "[F3]" in texte
    assert seen == ["F3"]


@pytest.mark.parametrize("body", [
    {"question": "Bonjour", "theme": "Inconnu"},
    {"question": "Bonjour", "since": "2025x"},
    {"question": "Bonjour", "since": 1999},
    {"question": "Bonjour", "since": True},
])
def test_ask_refuse_les_filtres_invalides(body):
    app_html.state.update(ready=True, llm=object(), retriever=object())
    response = app_html.app.test_client().post("/ask", json=body)
    assert response.status_code == 400
    assert response.get_json()["error"]


def test_ask_transmet_les_filtres(monkeypatch):
    def faux_agent(llm, retriever, question, trace, theme=None, since=None):
        assert (question, theme, since) == ("Bonjour", "Logement", 20250101)
        trace.record({"steps": [], "status": "done", "tokens_in": 0, "tokens_out": 0, "seconds": 0})
        return Reponse(answer="Oui.", citations=[]), []

    monkeypatch.setattr("rag_assistant.agent.run_agent", faux_agent)
    app_html.state.update(ready=True, llm=SimpleNamespace(provider="ollama", name="faux"), retriever=object())
    response = app_html.app.test_client().post("/ask", json={
        "question": "Bonjour", "theme": "Logement", "since": 2025,
    })
    assert response.status_code == 200
    assert response.get_json()["theme"] == "Logement"
    assert response.get_json()["since"] == 2025


def test_api_filtres_liste_les_themes_et_annees():
    data = app_html.app.test_client().get("/api/filtres").get_json()
    assert data["themes"] == ["Logement", "Papiers - Citoyenneté - Élections"]
    assert data["annees"] == sorted(data["annees"], reverse=True)
    assert 2026 in data["annees"]
