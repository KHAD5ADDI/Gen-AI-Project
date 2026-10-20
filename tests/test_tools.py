"""Outils de l'agent, sans modèle ni réseau."""

import pytest

from rag_assistant.tools import calculator, days_between, make_tools


class FakeRetriever:
    def search(self, query):
        assert query == "procuration"
        return [
            {
                "doc_id": "F1",
                "date": "2026-02-02",
                "title": "Vote par procuration",
                "section": "En France",
                "body": "Une procuration faite en France.",
                "text": "Une procuration faite en France.",
            },
            {
                "doc_id": "F1",
                "date": "2026-02-02",
                "title": "Vote par procuration",
                "body": "Autre passage.",
                "text": "Autre passage.",
            },
        ]


def test_calculator_comme_dans_le_lab():
    assert calculator("4.00 - 0.00") == "4"
    assert calculator("2 ** 10") == "1024"
    assert calculator("(3.75 - 2.00)") == "1.75"
    with pytest.raises(ValueError):
        calculator("__import__('os').system('rm -rf ~')")
    with pytest.raises(ValueError):
        calculator("open('secrets.txt').read()")


def test_days_between_comme_dans_le_lab():
    assert days_between("2024-01-01, 2024-03-01") == "60"


def test_get_document_tronque_une_fiche_reelle():
    tools = make_tools(FakeRetriever(), [])
    text = tools["get_document"]("F1314")
    assert len(text.split("\n[Fiche tronquée :", 1)[0]) == 6000
    assert "[Fiche tronquée : 6 000 caractères affichés sur " in text
    with pytest.raises(ValueError, match="fiche inconnue"):
        tools["get_document"]("F999999")


def test_search_formate_et_enregistre_une_seule_fois():
    seen = []
    tools = make_tools(FakeRetriever(), seen)
    assert tools["search"]("procuration") == (
        "[F1] (2026-02-02) Vote par procuration > En France : Une procuration faite en France.\n\n"
        "[F1] (2026-02-02) Vote par procuration : Autre passage."
    )
    assert seen == ["F1"]
    with pytest.raises(ValueError, match="requête vide"):
        tools["search"]("  ")
