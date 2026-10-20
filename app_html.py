"""Interface Flask + HTML de l'agent ReAct : réponse, sources, trace et fiches du corpus.

Lancement : uv run python app_html.py   puis http://localhost:5050
Le modèle se choisit dans .env (LLM_PROVIDER, OLLAMA_MODEL, OPENROUTER_MODEL), comme pour l'évaluation.
Variable optionnelle : APP_CHROMA_DIR (dossier de l'index, chroma_db_final par défaut).
"""
import html
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

# Évite le conflit de DLL OpenMP entre torch et ChromaDB sous Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: F401  (chargé avant ChromaDB, nécessaire sous Windows)
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static_html")

state = {"retriever": None, "llm": None, "ready": False, "error": None}
ask_lock = threading.Lock()  # un seul appel d'agent à la fois (Ollama local)


def load_system():
    try:
        from rag_assistant.ingest import IngestionPipeline
        from rag_assistant.llm import get_llm
        from rag_assistant.retrieval import HybridRetriever

        chroma_dir = os.getenv("APP_CHROMA_DIR", str(ROOT / "chroma_db_final"))
        print(f"Index : {chroma_dir}", flush=True)
        # Un ancien index lève une erreur (affichée dans le badge) : le reconstruire avec
        # scripts/run_ingest.py, pour que l'interface utilise les mêmes passages que l'évaluation.
        pipeline = IngestionPipeline(persist_directory=chroma_dir)
        print(f"Index OK ({len(pipeline.docstore)} passages). Chargement du retriever...", flush=True)
        state["retriever"] = HybridRetriever(ingestion_pipeline=pipeline)
        state["llm"] = get_llm()  # même choix de modèle que l'évaluation (.env)
        state["ready"] = True
        print(f"Système prêt ({state['llm'].provider} / {state['llm'].name}).", flush=True)
    except Exception as err:  # noqa: BLE001
        state["error"] = str(err)
        print(f"ERREUR CHARGEMENT : {err}", flush=True)
        traceback.print_exc()


CORPUS_PATH = ROOT / "data" / "raw" / "corpus.parquet"
_corpus = {"df": None}
DOC_ID = re.compile(r"^F\d+$")


def get_corpus():
    if _corpus["df"] is None:
        import pandas as pd

        df = pd.read_parquet(CORPUS_PATH, columns=["doc_id", "title", "text", "theme", "fil_ariane", "date", "url"])
        df["_search"] = (df["title"] + " " + df["text"]).str.lower()
        _corpus["df"] = df.sort_values("title").reset_index(drop=True)
    return _corpus["df"]


@app.route("/api/docs")
def list_docs():
    df = get_corpus()
    query = (request.args.get("q") or "").strip().lower()
    theme = (request.args.get("theme") or "").strip()
    if theme:
        df = df[df["theme"] == theme]
    if query:
        df = df[df["_search"].str.contains(query, regex=False)]
    rows = df.head(100)[["doc_id", "title", "theme", "date"]].to_dict(orient="records")
    return jsonify({
        "total": len(df),
        "themes": sorted(get_corpus()["theme"].unique().tolist()),
        "docs": rows,
    })


@app.route("/api/doc/<doc_id>")
def get_doc(doc_id):
    if not DOC_ID.match(doc_id):
        return jsonify({"error": "Identifiant de fiche invalide."}), 400
    df = get_corpus()
    row = df[df["doc_id"] == doc_id]
    if row.empty:
        return jsonify({"error": f"Fiche inconnue : {doc_id}"}), 404
    doc = row.iloc[0]
    data = {key: doc[key] for key in ("doc_id", "title", "theme", "fil_ariane", "date", "url", "text")}
    data["fil_ariane"] = html.unescape(data["fil_ariane"])
    return jsonify(data)


@app.route("/")
def index():
    return send_from_directory("static_html", "index.html")


@app.route("/status")
def status():
    llm = state["llm"]
    return jsonify({
        "ready": state["ready"],
        "error": state["error"],
        "model": f"{llm.provider} / {llm.name}" if llm else None,
    })


@app.route("/ask", methods=["POST"])
def ask():
    if not state["ready"]:
        return jsonify({"error": "Système en cours de chargement, réessaie dans quelques secondes."}), 503
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question vide."}), 400

    from rag_assistant.agent import run_agent
    from rag_assistant.tracing import Trace, cout_usd

    llm = state["llm"]
    trace = Trace()
    with ask_lock:
        start = time.perf_counter()
        try:
            reponse, seen = run_agent(llm, state["retriever"], question, trace)
        except Exception as err:  # noqa: BLE001
            traceback.print_exc()
            return jsonify({"error": f"Erreur de l'agent : {err}"}), 500
        latency = round(time.perf_counter() - start, 2)

    steps = []
    for step in trace.steps:
        action = step["action"]
        steps.append({
            "thought": step["text"],
            "tool": action[0] if action else None,
            "input": action[1] if action else None,
            "observation": step["observation"] or "",
            "seconds": round(step["seconds"], 2),
        })
    try:
        cost = cout_usd(trace.tokens_in, trace.tokens_out, llm.provider, llm.name)
    except ValueError:
        cost = None

    return jsonify({
        "answer": reponse.answer,
        "citations": reponse.citations,
        "answerable": reponse.answerable,
        "docs": seen,
        "latency": latency,
        "status": trace.status,
        "tokens_in": trace.tokens_in,
        "tokens_out": trace.tokens_out,
        "cost_usd": cost,
        "steps": steps,
    })


if __name__ == "__main__":
    threading.Thread(target=load_system, daemon=True).start()
    app.run(port=5050, debug=False, threaded=True)
