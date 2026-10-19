"""Évaluation de l'agent sur le golden set.

Usage : uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
        [--split all|dev|test] [--limit N]

Le champ `type` du golden set donne le type de chaque question.

Questions     type             Nombre  Ce qui est testé
------------  ---------------  ------  --------------------------------------------
q001 à q020   simple           20      Un fait précis dans une seule fiche
q021 à q031   conditionnelle   11      Une précision donnée dans la question
q032 à q035   hors_corpus      4       Un sujet absent du corpus
q036 à q040   injection        5       Une consigne trompeuse dans la question
"""

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from rag_assistant.agent import answer
from rag_assistant.ingest import IngestionPipeline
from rag_assistant.llm import get_llm
from rag_assistant.retrieval import HybridRetriever
from rag_assistant.tracing import Trace, cout_usd


def read_questions(path, split, limit):
    """Lit et filtre les questions dans leur ordre d'origine."""
    with open(path, encoding="utf-8") as source:
        questions = [json.loads(line) for line in source if line.strip()]
    if split != "all":
        questions = [question for question in questions if question["split"] == split]
    return questions[:limit] if limit is not None else questions


def report(question, result, trace, latency, llm):
    """Prépare une ligne JSON avec les champs demandés par le sujet."""
    steps = [{**step, "observation": (step["observation"] or "")[:600]} for step in result["steps"]]
    reponse = result["reponse"]
    return {
        "id": question["id"],
        "type": question["type"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["answer"],
        "generated_answer": reponse.answer,
        "answerable": reponse.answerable,
        "cited_sources": reponse.citations,
        "retrieved_document_IDs": result["retrieved"],
        "status": result["status"],
        "stop_reason": result["status"],
        "iterations": result["iterations"],
        "invalid_calls": result["invalid_calls"],
        "n_llm_calls": len(result["steps"]),
        "n_tool_calls": len(trace.tool_calls),
        "tool_calls": trace.tool_calls,
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
        "latency_sec": round(latency, 3),
        "cost_usd": cout_usd(result["tokens_in"], result["tokens_out"], llm.provider, llm.name),
        "provider": llm.provider,
        "model": llm.name,
        "steps": steps,
    }


def print_summary(results):
    """Affiche les agrégats de la boucle ReAct."""
    iterations = [item["iterations"] for item in results]
    total_iterations = sum(iterations)
    invalid = sum(item["invalid_calls"] for item in results)
    statuses = dict(Counter(item["status"] for item in results))
    costs = [item["cost_usd"] for item in results]
    latencies = [item["latency_sec"] for item in results]
    print(f"Questions : {len(results)}")
    print(f"Statuts : {statuses}")
    print(f"Itérations : moyenne {statistics.mean(iterations) if iterations else 0:.2f}, maximum {max(iterations, default=0)}")
    print(f"Taux d'appels invalides : {invalid / total_iterations if total_iterations else 0:.3f}")
    print(f"Coût moyen par question : {statistics.mean(costs) if costs else 0:.6f} USD")
    print(f"Durée médiane par question : {statistics.median(latencies) if latencies else 0:.2f} s")


def main():
    """Exécute et écrit l'évaluation, une ligne après chaque question."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    retriever = HybridRetriever(pipeline)
    llm = get_llm()
    questions = read_questions(args.questions, args.split, args.limit)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with (output / "results.jsonl").open("w", encoding="utf-8") as destination:
        for question in questions:
            start = time.perf_counter()
            result = answer(llm, retriever, question["question"])
            latency = time.perf_counter() - start
            trace = Trace()
            trace.record(result)
            item = report(question, result, trace, latency, llm)
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
            destination.flush()
            results.append(item)
            print(
                f"{item['id']} {item['status']} {item['iterations']} itérations "
                f"{item['tokens_in']}+{item['tokens_out']} tokens "
                f"{item['latency_sec']:.2f} s {item['cost_usd']:.6f} USD"
            )
    print_summary(results)


if __name__ == "__main__":
    main()
