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
