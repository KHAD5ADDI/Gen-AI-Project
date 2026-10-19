"""Évaluation de l'assistant sur le golden set.

    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

Golden set : eval/golden.jsonl, 40 questions au format du sujet (id, question, answer, gold_docs)

Questions     Type                   Nombre  Ce qui est testé
------------  ---------------------  ------  ----------------------------------------------------
q001 à q020   Simples                20      Un fait précis dans une seule fiche : 2 reprennent mot
                                             pour mot le titre de la fiche, 9 sont proches du
                                             texte, 9 sont formulées comme le ferait un usager
q021 à q031   Conditionnelles        11      La réponse dépend d'une précision donnée dans la
                                             question
q032 à q035   Hors corpus            4       Sujets hors du domaine, que le modèle connaît par
                                             son entraînement
q036 à q040   Injections de prompt   5       Tentatives de détourner les règles de l'assistant

Détail des questions et des pièges testés : eval/golden_notes.md
"""

import json
import time
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llama_index.llms.ollama import Ollama
from src.rag_assistant.ingest import IngestionPipeline
from src.rag_assistant.retrieval import HybridRetriever
from src.rag_assistant.agent import run_agent
from src.rag_assistant.tracing import Trace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pipeline = IngestionPipeline(persist_directory="./chroma_db_final")
    retriever = HybridRetriever(ingestion_pipeline=pipeline)
    llm = Ollama(model="llama3.1:8b", request_timeout=120.0)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.out) / "results.jsonl"

    with open(out_file, "w", encoding="utf-8") as f_out:
        for q in questions:
            trace = Trace()
            start_time = time.perf_counter()
            
            reponse, retrieved_ids = run_agent(llm, retriever, q["question"], trace)
            latency = time.perf_counter() - start_time
            
            result = {
                "id": q.get("id", "unknown"),
                "question": q["question"],
                "expected_answer": q.get("answer", ""),
                "generated_answer": reponse.answer,
                "cited_sources": reponse.citations,
                "retrieved_document_IDs": retrieved_ids,
                "latency_sec": round(latency, 2),
                "tool_calls": len(trace.tool_calls),
                "answerable": reponse.answerable
            }
            
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Évalué: {result['id']} en {latency:.1f}s")

if __name__ == "__main__":
    main()
