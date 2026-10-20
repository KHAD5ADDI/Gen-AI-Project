"""Contrat du corpus (data/raw/corpus.parquet) : ce que les étapes suivantes supposent."""
import re

import pandas as pd

from rag_assistant.config import CORPUS_PATH

CORPUS = pd.read_parquet(CORPUS_PATH)
COLUMNS = ["doc_id", "title", "text", "theme", "fil_ariane", "date", "url", "type"]
THEMES = {"Logement", "Papiers - Citoyenneté - Élections"}


def test_colonnes():
    assert list(CORPUS.columns) == COLUMNS


def test_identifiants_uniques_au_format_F1234():
    assert CORPUS["doc_id"].is_unique
    assert CORPUS["doc_id"].str.fullmatch(r"F\d+").all()


def test_aucun_champ_vide():
    for col in ["title", "text", "theme", "date", "url"]:
        assert (CORPUS[col].str.strip() != "").all(), col


def test_dates_au_format_iso():
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in CORPUS["date"])


def test_selection_documentee_dans_la_data_card():
    assert set(CORPUS["theme"]) == THEMES
    assert (CORPUS["text"].str.len() >= 200).all()
    assert len(CORPUS) == 526
