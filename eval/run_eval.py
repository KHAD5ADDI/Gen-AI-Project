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

"""

import json
import time
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_assistant.ingest import IngestionPipeline
from src.rag_assistant.retrieval import HybridRetriever
from src.rag_assistant.agent import run_agent
from src.rag_assistant.llm import get_llm, provider_of
from src.rag_assistant.tracing import Trace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pipeline = IngestionPipeline(persist_directory="./chroma_db_final")
    retriever = HybridRetriever(ingestion_pipeline=pipeline)
    llm = get_llm()  # modèle choisi dans le .env (LLM_PROVIDER), Ollama par défaut
    provider, model = provider_of(llm), llm.model
    print(f"Modèle : {provider} / {model}")

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_file = Path(args.out) / "results.jsonl"

    with open(out_file, "w", encoding="utf-8") as f_out:
        for q in questions:
            trace = Trace(provider=provider, model=model)
            start_time = time.perf_counter()
            reponse, retrieved_ids = run_agent(llm, retriever, q["question"], trace)
            latency = time.perf_counter() - start_time
            s = trace.summary()

            result = {
                "id": q.get("id", "unknown"),
                "question": q["question"],
                "expected_answer": q.get("answer", ""),
                "generated_answer": reponse.answer,
                "cited_sources": reponse.citations,
                "retrieved_document_IDs": retrieved_ids,
                "answerable": reponse.answerable,
                "provider": provider,
                "model": model,
                "tokens_in": s["tokens_in"],
                "tokens_out": s["tokens_out"],
                "latency_sec": round(latency, 2),
                "cost_usd": s["cost_usd"],
                "n_llm_calls": s["n_llm_calls"],
                "n_tool_calls": len(s["tool_calls"]),
                "stop_reason": s["stop_reason"],
                "tool_calls": s["tool_calls"],  # détail : outil, arguments, durée, taille du résultat
                "llm_calls": s["llm_calls"],  # détail par appel : rôle, tokens, durée, raison d'arrêt
                "errors": s["errors"],
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()  # une ligne par question, même si le run est interrompu
            print(f"Évalué: {result['id']} en {latency:.1f}s | {s['tokens_in']} tokens in, "
                  f"{s['tokens_out']} out | {s['n_llm_calls']} appels LLM | {s['cost_usd']:.4f} USD")

if __name__ == "__main__":
    main()
