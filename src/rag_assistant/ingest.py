import pandas as pd
import chromadb
from src.rag_assistant.chunking import chunk_document
from src.rag_assistant.embedding import Embedder

class IngestionPipeline:
    def __init__(self, persist_directory="/content/chroma_db"):
        self.embedder = Embedder()

        # Si persist_directory est None, on utilise une base en mémoire pour les tests
        if persist_directory is None:
            print("Mode test : utilisation de ChromaDB en mémoire (EphemeralClient).")
            self.chroma_client = chromadb.EphemeralClient()
        else:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Nettoyage robuste
        try:
            self.chroma_client.delete_collection(name="rag_sp_documents")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(name="rag_sp_documents")
        self.docstore = {}

    def ingest_parquet(self, parquet_path="/content/corpus.parquet"):
        # ... (Le reste de la méthode ingest_parquet reste strictement identique)
        print(f"Lecture du corpus depuis {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        all_chunks = []
        for row in df.to_dict(orient="records"):
            all_chunks.extend(chunk_document(row))

        print(f"Encodage de {len(all_chunks)} chunks (avec préfixe 'passage:')...")
        texts = [c["text"] for c in all_chunks]

        embeddings = self.embedder.encode_chunks(texts)

        ids = [c["chunk_id"] for c in all_chunks]
        metadatas = [{
            "doc_id": c["doc_id"],
            "date": c["date"],
            "title": c["title"],
            "theme": c["theme"]
        } for c in all_chunks]

        print("Insertion dans ChromaDB et le docstore (BM25)...")
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        for chunk in all_chunks:
            self.docstore[chunk["chunk_id"]] = chunk

        print(f"Ingestion terminée : {len(all_chunks)} chunks indexés.")
