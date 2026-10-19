"""Construit l'index vectoriel (Chroma) à partir du corpus.

À relancer après tout changement du découpage ou du modèle d'embedding :
    uv run python scripts/run_ingest.py [--index DIR] [--size N] [--overlap N]
"""
import argparse
from pathlib import Path

from rag_assistant.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from rag_assistant.ingest import IngestionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=CHROMA_DIR)
    parser.add_argument("--size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    # L'index existant peut avoir l'ancien découpage : on va le reconstruire.
    pipeline = IngestionPipeline(persist_directory=args.index, load_docstore=False)
    pipeline.ingest_parquet(size=args.size, overlap=args.overlap)
