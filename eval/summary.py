"""Compare les résultats et verdicts de plusieurs configurations."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from metrics import mrr, ndcg_at_k, recall_at_k

TYPES = ("simple", "conditionnelle", "hors_corpus", "injection")
GOLDEN_PATH = Path(__file__).with_name("golden.jsonl")
COLUMNS = (
    "config", "questions", "score_juge", "score_juge_test", "refus_corrects", "reponses_appuyees",
    "recall@5", "mrr", "ndcg@5", "fiche_citee", "iterations_moy", "iterations_max",
    "appels_invalides", "statuts", "tokens_in_moy", "tokens_out_moy", "usd_par_question",
    "usd_juge_total", "latence_mediane_s",
)
DETAIL_COLUMNS = (
    "config", "id", "type", "split", "status", "judged_by", "correct", "score",
    "supported_by_sources", "reason", "expected_answer", "generated_answer", "cited_sources",
    "retrieved_document_IDs",
)


def read_jsonl(path):
    """Lit un fichier JSONL."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def average(values):
    """Calcule une moyenne, ou zéro si la liste est vide."""
    return statistics.mean(values) if values else 0.0


def summarize(directories, golden_path=GOLDEN_PATH):
    """Construit le résumé et le détail de chaque configuration."""
    golden = {item["id"]: item for item in read_jsonl(Path(golden_path))}
    rows, type_rows, details = [], [], []
    for directory in directories:
        directory = Path(directory)
        results = read_jsonl(directory / "results.jsonl")
        judgments = {item["id"]: item for item in read_jsonl(directory / "judgments.jsonl")}
        if not results:
            raise ValueError(f"Aucun résultat dans {directory}.")
        config = f"{results[0]['provider']}/{results[0]['model']}"
        pairs = [(result, judgments[result["id"]], golden[result["id"]]) for result in results]
        with_gold = [(result, gold) for result, _, gold in pairs if gold["gold_docs"]]
        refusals = [(result, judgment) for result, judgment, gold in pairs if not gold["gold_docs"]]
        judged = [judgment for _, judgment, _ in pairs if judgment["judged_by"] == "juge"]
        iterations = sum(result["iterations"] for result in results)
        statuses = Counter(result["status"] for result in results)
        rows.append({
            "config": config,
            "questions": len(results),
            "score_juge": round(average([judgment["score"] for _, judgment, _ in pairs]), 3),
            "score_juge_test": round(average([
                judgment["score"] for _, judgment, _ in pairs if judgment["split"] == "test"
            ]), 3),
            "refus_corrects": f"{sum(judgment['correct'] == 'yes' for _, judgment in refusals)}/{len(refusals)}",
            "reponses_appuyees": round(average([
                float(judgment["supported_by_sources"]) for judgment in judged
            ]), 3),
            "recall@5": round(average([
                recall_at_k(result["retrieved_document_IDs"], gold["gold_docs"], 5)
                for result, gold in with_gold
            ]), 3),
            "mrr": round(average([
                mrr(result["retrieved_document_IDs"], gold["gold_docs"])
                for result, gold in with_gold
            ]), 3),
            "ndcg@5": round(average([
                ndcg_at_k(result["retrieved_document_IDs"], gold["gold_docs"], 5)
                for result, gold in with_gold
            ]), 3),
            "fiche_citee": round(average([
                float(bool(set(result["cited_sources"]) & set(gold["gold_docs"])))
                for result, gold in with_gold
            ]), 3),
            "iterations_moy": round(average([result["iterations"] for result in results]), 3),
            "iterations_max": max(result["iterations"] for result in results),
            "appels_invalides": round(sum(result["invalid_calls"] for result in results) / iterations
                                     if iterations else 0, 3),
            "statuts": ", ".join(f"{status} {count}" for status, count in statuses.items()),
            "tokens_in_moy": round(average([result["tokens_in"] for result in results]), 3),
            "tokens_out_moy": round(average([result["tokens_out"] for result in results]), 3),
            "usd_par_question": round(average([result["cost_usd"] for result in results]), 4),
            "usd_juge_total": round(sum(judgment["judge_cost_usd"] for _, judgment, _ in pairs), 4),
            "latence_mediane_s": round(statistics.median(result["latency_sec"] for result in results), 3),
        })
        type_rows.append({"config": config, **{
            kind: round(average([
                judgment["score"] for _, judgment, _ in pairs if judgment["type"] == kind
            ]), 3) for kind in TYPES
        }})
        for result, judgment, _ in pairs:
            details.append({
                "config": config, "id": result["id"], "type": result["type"], "split": result["split"],
                "status": result["status"], "judged_by": judgment["judged_by"],
                "correct": judgment["correct"], "score": judgment["score"],
                "supported_by_sources": judgment["supported_by_sources"], "reason": judgment["reason"],
                "expected_answer": result["expected_answer"],
                "generated_answer": result["generated_answer"],
                "cited_sources": ", ".join(result["cited_sources"]),
                "retrieved_document_IDs": ", ".join(result["retrieved_document_IDs"]),
            })
    return (pd.DataFrame(rows, columns=COLUMNS),
            pd.DataFrame(type_rows, columns=("config", *TYPES)),
            pd.DataFrame(details, columns=DETAIL_COLUMNS))


def bootstrap_diff(correct_a, correct_b, n=10_000, seed=0):
    """Estime l'écart de score par bootstrap apparié (Session 8)."""
    rng = np.random.default_rng(seed)
    a = np.array(correct_a, float)
    b = np.array(correct_b, float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])


def markdown_table(frame):
    """Formate une table simple sans dépendance supplémentaire."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.to_dict("records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.4f}" if column.startswith("usd_") else f"{value:.3f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    """Écrit les tables CSV et Markdown, puis les affiche."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", metavar="DIR", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--compare", nargs=2, metavar=("NAME_A", "NAME_B"))
    args = parser.parse_args()
    summary, types, detail = summarize(args.directories)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    types.to_csv(args.out / "summary_types.csv", index=False)
    detail.to_csv(args.out / "per_question.csv", index=False)
    first, second = markdown_table(summary), markdown_table(types)
    (args.out / "summary.md").write_text(first + "\n\n" + second + "\n", encoding="utf-8")
    print(first + "\n\n" + second)

    if args.compare:
        name_a, name_b = args.compare
        configs = summary["config"].tolist()
        if configs.count(name_a) != 1 or configs.count(name_b) != 1:
            raise ValueError("Chaque configuration comparée doit apparaître une fois.")
        by_config = {name: detail[detail["config"] == name].set_index("id") for name in (name_a, name_b)}
        a, b = by_config[name_a], by_config[name_b]
        if set(a.index) != set(b.index):
            raise ValueError("Les configurations comparées doivent avoir les mêmes questions.")
        mean, (low, high) = bootstrap_diff(a["score"].tolist(), b.loc[a.index, "score"].tolist())
        print(f"{name_b} - {name_a} = {mean:+.1%} (IC 95 % {low:+.1%} à {high:+.1%})")
        print("L'intervalle contient 0." if low <= 0 <= high else "L'intervalle ne contient pas 0.")


if __name__ == "__main__":
    main()
