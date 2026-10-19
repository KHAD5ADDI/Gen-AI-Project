"""Compare les 3 variantes de pipeline : original, confidence, crossval."""
import sys, os, time
os.chdir(r"C:\Users\khadi\OneDrive\Bureau\AlbertSchool\Gen-AI-Project")

print("=== Chargement ===", flush=True)
from src.rag_assistant.ingest import IngestionPipeline
from src.rag_assistant.retrieval import HybridRetriever
from src.rag_assistant.tracing import Trace
from llama_index.llms.ollama import Ollama

pipeline = IngestionPipeline(persist_directory=r"C:\Users\khadi\OneDrive\Bureau\AlbertSchool\Gen-AI-Project\chroma_db_final")
retriever = HybridRetriever(pipeline)
llm = Ollama(model="llama3.2:3b", request_timeout=300.0, context_window=4096)
print(f"Pipeline chargée : {len(pipeline.docstore)} chunks\n", flush=True)

QUESTIONS = [
    "Quel est le délai de restitution du dépôt de garantie ?",
    "Quels sont les recours en cas de loyer impayé ?",
    "Comment résilier un bail d'habitation ?",
]


def run_pipeline(name, generate_fn, question, chunks):
    trace = Trace()
    t0 = time.time()
    try:
        reponse = generate_fn(llm, question, chunks, trace)
        elapsed = time.time() - t0
        return {
            "pipeline": name,
            "answerable": reponse.answerable,
            "answer_len": len(reponse.answer),
            "citations": reponse.citations,
            "extra": getattr(reponse, "confidence", None),
            "elapsed": round(elapsed, 1),
            "errors": trace.errors,
        }
    except Exception as e:
        return {"pipeline": name, "error": str(e), "elapsed": round(time.time() - t0, 1)}


for question in QUESTIONS:
    print(f"\n{'='*60}", flush=True)
    print(f"QUESTION : {question}", flush=True)
    chunks = retriever.search(question, top_k=4)
    print(f"Passages récupérés : {len(chunks)}", flush=True)
    if not chunks:
        print("Aucun passage - question ignorée", flush=True)
        continue

    from src.rag_assistant.generation import generate as gen_original
    from src.rag_assistant.generation_confidence import generate_with_confidence
    from src.rag_assistant.generation_crossval import generate_with_crossval

    for name, fn in [
        ("ORIGINAL", gen_original),
        ("CONFIDENCE", generate_with_confidence),
        ("CROSSVAL", generate_with_crossval),
    ]:
        print(f"\n--- {name} ---", flush=True)
        result = run_pipeline(name, fn, question, chunks)
        if "error" in result:
            print(f"ERREUR : {result['error']} ({result['elapsed']}s)", flush=True)
        else:
            print(f"Answerable : {result['answerable']}", flush=True)
            print(f"Longueur réponse : {result['answer_len']} chars", flush=True)
            print(f"Citations : {result['citations']}", flush=True)
            if result["extra"] is not None:
                print(f"Confidence : {result['extra']}/100", flush=True)
            print(f"Temps : {result['elapsed']}s", flush=True)
            if result["errors"]:
                print(f"Erreurs trace : {result['errors']}", flush=True)

print("\n=== FIN ===", flush=True)
