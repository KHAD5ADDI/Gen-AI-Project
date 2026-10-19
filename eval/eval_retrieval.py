"""Évalue Recall@5, MRR et nDCG@5 des fiches attendues pour quatre configurations.

Configurations : dense, BM25, hybride RRF et hybride + rerank.
Les questions sans gold_docs sont ignorées. Les métriques sont calculées par fiche
sur les 30 passages utilisés par le pipeline.

Usage : uv run python eval/eval_retrieval.py [--index DIR] [--golden eval/golden.jsonl] [--out eval/]
Sorties : retrieval_results.jsonl, retrieval_summary.csv, retrieval_summary.md.
"""

import torch  # noqa: F401, I001  (chargé en premier : évite un conflit de DLL sous Windows)
import argparse
import json
from pathlib import Path

import pandas as pd

from metrics import fiches, mrr, ndcg_at_k, recall_at_k  # lancé par son chemin : eval/ est dans sys.path
from rag_assistant.config import CHROMA_DIR, GOLDEN_PATH
from rag_assistant.ingest import IngestionPipeline
from rag_assistant.retrieval import HybridRetriever, rrf

PRE_K = 30  # même valeur que dans HybridRetriever.search
CONFIGS = ("dense", "bm25", "hybride (RRF)", "hybride + rerank")


def charger_questions(path):
    """Lit les questions qui possèdent au moins une fiche attendue."""
    questions = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                question = json.loads(line)
                if question.get("gold_docs"):
                    questions.append(question)
    return questions


def evaluer(questions, ret):
    """Évalue les quatre classements de chaque question."""
    rows = []
    misses = []
    for q in questions:
        question = q["question"]
        dense = ret.dense_search(question, top_n=PRE_K)
        bm25 = ret.bm25_search(question, top_n=PRE_K)
        rankings = (
            dense,
            bm25,
            rrf([dense, bm25], k=60, depth=PRE_K),
            [c["chunk_id"] for c in ret.search(question, top_k=PRE_K)],
        )
        for config, ranking in zip(CONFIGS, rankings):
            documents = fiches(ranking)
            row = {
                "id": q["id"],
                "type": q.get("type", "-"),
                "config": config,
                "gold_docs": q["gold_docs"],
                "top5_fiches": documents[:5],
                "recall@5": recall_at_k(documents, q["gold_docs"], 5),
                "mrr": mrr(documents, q["gold_docs"]),
                "ndcg@5": ndcg_at_k(documents, q["gold_docs"], 5),
            }
            rows.append(row)
            if config == "hybride + rerank" and row["recall@5"] == 0:
                misses.append((row, question[:70]))
    return rows, misses


def resume(rows):
    """Calcule les moyennes des métriques par configuration."""
    summary = pd.DataFrame(rows).groupby("config", sort=False).agg({
        "id": "count",
        "recall@5": "mean",
        "mrr": "mean",
        "ndcg@5": "mean",
    }).reset_index()
    return summary.rename(columns={"id": "questions", "recall@5": "Recall@5",
                                   "mrr": "MRR", "ndcg@5": "nDCG@5"}).round(3)


def tableau(summary):
    """Présente le résumé sous forme de tableau Markdown."""
    lines = ["| Configuration | Questions | Recall@5 | MRR | nDCG@5 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.3f} | {row[3]:.3f} | {row[4]:.3f} |")
    return "\n".join(lines)


def main():
    """Évalue la recherche et écrit les résultats dans le dossier choisi."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=CHROMA_DIR)
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    questions = charger_questions(args.golden)
    pipeline = IngestionPipeline(persist_directory=args.index)
    ret = HybridRetriever(ingestion_pipeline=pipeline)
    rows, misses = evaluer(questions, ret)
    summary = resume(rows)

    print(f"Résumé ({len(questions)} questions) :")
    print(tableau(summary))
    types = list(dict.fromkeys(q.get("type", "-") for q in questions))
    if len(types) > 1:
        for question_type in types:
            print(f"\nType : {question_type}")
            print(tableau(resume([row for row in rows if row["type"] == question_type])))

    print("\nQuestions sans fiche attendue dans les 5 premières après reranking :")
    for row, question in misses:
        print(f"{row['id']} : gold={row['gold_docs']}, top5={row['top5_fiches']}, question={question}")
    if not misses:
        print("Aucune")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "retrieval_results.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary.to_csv(args.out / "retrieval_summary.csv", index=False, float_format="%.3f")
    (args.out / "retrieval_summary.md").write_text(tableau(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
