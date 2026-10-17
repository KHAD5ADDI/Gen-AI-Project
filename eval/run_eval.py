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
