"""Juge les réponses de l'agent avec le protocole du lab 08."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from rag_assistant.config import JUDGE_MODEL
from rag_assistant.tracing import cout_usd


class Verdict(BaseModel):
    correct: Literal["yes", "partially", "no"]
    supported_by_sources: bool
    reason: str = Field(description="One sentence, quoting the reference")


SCORE = {"yes": 1.0, "partially": 0.5, "no": 0.0}

JUDGE_PROMPT = """Grade this answer about French administrative procedures (Service-Public fiches).
Question: {question}
Reference answer: {reference}
Sources the model received:
{sources}
Answer to grade: {answer}
Grade correctness against the reference, and whether every claim is supported by the sources.
Do not reward length or style."""


def make_client():
    """Prépare le client OpenRouter du juge."""
    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Clé OPENROUTER_API_KEY absente : renseignez-la dans .env.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=60)


def sources_text(result, max_chars=12000):
    """Rassemble les observations des outils qui donnent accès aux fiches."""
    observations = [
        step["observation"] or ""
        for step in result["steps"]
        if step["action"] is not None and step["action"][0] in ("search", "get_document")
    ]
    # Limite le coût du juge.
    return "\n\n".join(observations)[:max_chars]


def judge_one(client, question, reference, sources, answer):
    """Demande un verdict structuré ; un appel échoué vaut non."""
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, sources=sources, answer=answer)
    try:
        response = client.chat.completions.parse(
            model=JUDGE_MODEL,
            temperature=0,
            response_format=Verdict,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.parsed, response.usage
    except Exception:  # noqa: BLE001  (tout échec du juge compte comme « no », selon le lab 08)
        return Verdict(correct="no", supported_by_sources=False, reason="judge call failed"), None


def grade(result, gold, client):
    """Note une question, avec la règle de refus avant le juge."""
    usage = None
    if not gold["gold_docs"]:
        refused = not result["answerable"]
        verdict = Verdict(
            correct="yes" if refused else "no",
            supported_by_sources=refused,
            reason="Refus attendu : l'assistant a refusé." if refused else "Refus attendu : l'assistant a répondu.",
        )
        judged_by = "règle"
    elif not result["answerable"]:
        verdict = Verdict(
            correct="no", supported_by_sources=False,
            reason="Pas de réponse alors qu'une réponse était attendue.",
        )
        judged_by = "règle"
    else:
        verdict, usage = judge_one(
            client, result["question"], gold["answer"], sources_text(result), result["generated_answer"]
        )
        judged_by = "juge"
    tokens_in = usage.prompt_tokens if usage is not None else 0
    tokens_out = usage.completion_tokens if usage is not None else 0
    return {
        "id": result["id"],
        "type": result["type"],
        "split": result["split"],
        "judged_by": judged_by,
        "correct": verdict.correct,
        "score": SCORE[verdict.correct],
        "supported_by_sources": verdict.supported_by_sources,
        "reason": verdict.reason,
        "judge_model": JUDGE_MODEL if judged_by == "juge" else None,
        "judge_tokens_in": tokens_in,
        "judge_tokens_out": tokens_out,
        "judge_cost_usd": cout_usd(tokens_in, tokens_out, "openrouter", JUDGE_MODEL) if usage else 0,
    }


def read_jsonl(path):
    """Lit les lignes JSON non vides."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    """Écrit un verdict par ligne et affiche les moyennes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=Path(__file__).with_name("golden.jsonl"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    golden = {item["id"]: item for item in read_jsonl(args.golden)}
    results = read_jsonl(args.results)
    if args.limit is not None:
        results = results[:args.limit]
    client = make_client() if any(item["answerable"] and golden[item["id"]]["gold_docs"] for item in results) else None
    scores_by_type = defaultdict(list)
    costs = []
    destination = args.results.with_name("judgments.jsonl")
    with destination.open("w", encoding="utf-8") as file:
        for result in results:
            row = grade(result, golden[result["id"]], client)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            scores_by_type[row["type"]].append(row["score"])
            costs.append(row["judge_cost_usd"])
            print(f"{row['id']} {row['judged_by']} {row['correct']} {row['reason'][:90]}")
    all_scores = [score for scores in scores_by_type.values() for score in scores]
    print(f"Score moyen : {mean(all_scores) if all_scores else 0:.3f}")
    for kind, scores in scores_by_type.items():
        print(f"{kind} : {mean(scores):.3f}")
    print(f"Coût total du juge : {sum(costs):.4f} USD")


if __name__ == "__main__":
    main()
