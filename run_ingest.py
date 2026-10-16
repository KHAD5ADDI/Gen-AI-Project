from src.rag_assistant.ingest import IngestionPipeline

if __name__ == "__main__":
    pipeline = IngestionPipeline(persist_directory="./chroma_db_final")
    pipeline.ingest_parquet("data/raw/corpus.parquet")
