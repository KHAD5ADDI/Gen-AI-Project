"""Contrat du golden set (eval/golden.jsonl), au format du sujet."""
import json

import pandas as pd

from rag_assistant.config import CORPUS_PATH, GOLDEN_PATH

GOLDEN = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
DOC_IDS = set(pd.read_parquet(CORPUS_PATH, columns=["doc_id"])["doc_id"])


def test_champs_du_sujet():
    for q in GOLDEN:
        assert {"id", "question", "answer", "gold_docs"} <= set(q), q.get("id")
        assert isinstance(q["gold_docs"], list)


def test_identifiants_uniques():
    ids = [q["id"] for q in GOLDEN]
    assert len(ids) == len(set(ids))


def test_gold_docs_existent_dans_le_corpus():
    absents = [(q["id"], d) for q in GOLDEN for d in q["gold_docs"] if d not in DOC_IDS]
    assert absents == []


def test_types_et_splits():
    types = {"simple", "conditionnelle", "hors_corpus", "injection"}
    for q in GOLDEN:
        assert q["type"] in types, q["id"]
        assert q["split"] in {"dev", "test"}, q["id"]
    for kind in types:
        assert any(q["type"] == kind and q["split"] == "test" for q in GOLDEN)


def test_gold_docs_selon_le_type():
    for q in GOLDEN:
        if q["type"] == "hors_corpus":
            assert not q["gold_docs"], q["id"]
        if q["type"] in {"simple", "conditionnelle"}:
            assert q["gold_docs"], q["id"]
