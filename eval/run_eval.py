import json
import time
import argparse argparse
from pathlib import Path
import ollama # Utilisation du modèle local recommandé
from src.rag_assistant.retrieval import Retriever
from src.rag_assistant.ingest import IngestionPipeline

def build_prompt(question, docs):
    """Construit le prompt avec la consigne stricte de citer les sources[cite: 3]."""
    sources_text = "\n\n".join([f"[{i+1}] {doc['text']}" for i, doc in enumerate(docs)])
    system_prompt = (
        "Tu es un assistant juridique sur les droits des citoyens français. "
        "Réponds UNIQUEMENT en utilisant les sources numérotées fournies. "
        "Termine chaque phrase par le numéro de la source (ex: [1]). "
        "Si la réponse n'est pas dans les sources, réponds 'Je ne trouve pas l'information'."
    )
    user_prompt = f"Sources:\n{sources_text}\n\nQuestion: {question}"
    return system_prompt, user_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Initialisation du pipeline
    pipeline = IngestionPipeline(persist_directory="/content/chroma_db_final")
    retriever = Retriever(ingestion_pipeline=pipeline)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.out) / "results.jsonl"

    with open(out_file, "w", encoding="utf-8") as f_out:
        for q in questions:
            start_time = time.time()

            # 1. Retrieval
            retrieved_chunks = retriever.retrieve_and_rerank(q["question"], top_k=4)
            retrieved_ids = [chunk["doc_id"] for chunk in retrieved_chunks]

            # 2. Prompting & LLM Generation (Local via Ollama)
            system_p, user_p = build_prompt(q["question"], retrieved_chunks)

            response = ollama.chat(
                model="llama3.1:8b", # ou mistral
                messages=[
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p}
                ]
            )

            latency = time.time() - start_time

            # 3. Extraction des métriques
            result = {
                "id": q["id"],
                "question": q["question"],
                "expected_answer": q["answer"],
                "generated_answer": response['message']['content'],
                "cited_sources": retrieved_ids,
                "retrieved_document_IDs": retrieved_ids,
                "tokens_in": response.get('prompt_eval_count', 0),
                "tokens_out": response.get('eval_count', 0),
                "latency_sec": round(latency, 2),
                "tool_calls": 0, # À incrémenter si vous utilisez un Agent paramétré
                "stop_reason": response.get('done_reason', 'stop')
            }

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Évalué: {q['id']} en {latency:.1f}s")

if __name__ == "__main__":
    main()
