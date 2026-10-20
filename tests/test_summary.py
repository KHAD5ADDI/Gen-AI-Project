"""Tests du résumé sur deux petites configurations."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from summary import bootstrap_diff, summarize


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_resume_deux_configurations(tmp_path):
    golden = [
        {"id": "q1", "type": "simple", "split": "dev", "gold_docs": ["F1"]},
        {"id": "q2", "type": "conditionnelle", "split": "test", "gold_docs": ["F2"]},
        {"id": "q3", "type": "hors_corpus", "split": "dev", "gold_docs": []},
        {"id": "q4", "type": "injection", "split": "test", "gold_docs": []},
    ]
    golden_path = tmp_path / "golden.jsonl"
    write_jsonl(golden_path, golden)
    directories = []
    for model, scores in [("modele-a", [1.0, 0.5, 1.0, 0.0]), ("modele-b", [0.0, 1.0, 1.0, 1.0])]:
        directory = tmp_path / model
        directory.mkdir()
        directories.append(directory)
        results = []
        judgments = []
        for index, question in enumerate(golden):
            results.append({
                "id": question["id"], "type": question["type"], "split": question["split"],
                "provider": "openrouter", "model": model, "status": "done", "iterations": 2,
                "invalid_calls": index % 2, "retrieved_document_IDs": ["F1"] if index == 0 else [],
                "cited_sources": ["F1"] if index == 0 else [], "tokens_in": 10,
                "tokens_out": 5, "cost_usd": 0.001, "latency_sec": 1.0,
                "expected_answer": "Attendu", "generated_answer": "Obtenu",
            })
            judgments.append({
                "id": question["id"], "type": question["type"], "split": question["split"],
                "judged_by": "juge" if index < 2 else "règle",
                "correct": {1.0: "yes", 0.5: "partially", 0.0: "no"}[scores[index]],
                "score": scores[index], "supported_by_sources": index == 0,
                "reason": "Motif", "judge_cost_usd": 0.0001,
            })
        write_jsonl(directory / "results.jsonl", results)
        write_jsonl(directory / "judgments.jsonl", judgments)

    table, types, detail = summarize(directories, golden_path)
    a, b = table.to_dict("records")
    assert (a["config"], b["config"]) == ("openrouter/modele-a", "openrouter/modele-b")
    assert (a["score_juge"], a["score_juge_test"], a["refus_corrects"]) == (0.625, 0.25, "1/2")
    assert (b["score_juge"], b["score_juge_test"], b["refus_corrects"]) == (0.75, 1.0, "2/2")
    assert a["appels_invalides"] == 0.25
    assert a["fiche_citee"] == 0.5
    assert len(types) == 2 and len(detail) == 8


def test_bootstrap_identique_et_reproductible():
    mean, interval = bootstrap_diff([0, 1, 0.5], [0, 1, 0.5])
    assert mean == 0 and np.array_equal(interval, [0, 0])
    first = bootstrap_diff([0, 1, 0], [1, 0, 1], n=100, seed=4)
    second = bootstrap_diff([0, 1, 0], [1, 0, 1], n=100, seed=4)
    assert first[0] == second[0] and np.array_equal(first[1], second[1])
