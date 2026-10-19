"""Construit l'index vectoriel (Chroma) à partir du corpus.

À relancer après tout changement du découpage ou du modèle d'embedding :
    uv run python scripts/run_ingest.py
"""
from rag_assistant.ingest import IngestionPipeline

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.ingest_parquet()
