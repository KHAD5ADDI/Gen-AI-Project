import pandas as pd
import chromadb
from .chunking import chunk_document
from .embedding import Embedder

class IngestionPipeline:
    def __init__(self, persist_directory="./chroma_db_final", parquet_path="data/raw/corpus.parquet"):
        self.embedder = Embedder()
        self.chroma_client = chromadb.PersistentClient(path=persist_directory) if persist_directory else chromadb.EphemeralClient()
        # On ouvre l'index existant sans l'effacer, et on reconstruit le docstore
        # (en mémoire) en redécoupant le parquet : rapide, aucun embedding recalculé.
        self.collection = self.chroma_client.get_or_create_collection(name="rag_sp_documents")
        self.docstore = {}
        if self.collection.count() > 0:
            for chunk in self._chunks(parquet_path):
                self.docstore[chunk["chunk_id"]] = chunk

    def _chunks(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        all_chunks = []
        for row in df.to_dict(orient="records"):
            all_chunks.extend(chunk_document(row))
        return all_chunks

    def ingest_parquet(self, parquet_path="data/raw/corpus.parquet"):
        # La remise à zéro de l'index se fait ici, plus à l'ouverture.
        self.chroma_client.delete_collection(name="rag_sp_documents")
        self.collection = self.chroma_client.create_collection(name="rag_sp_documents")
        all_chunks = self._chunks(parquet_path)
            
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.encode_chunks(texts)
        ids = [c["chunk_id"] for c in all_chunks]
        metadatas = [{"doc_id": c["doc_id"], "date": c["date"], "title": c["title"], "theme": c["theme"]} for c in all_chunks]
        
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
