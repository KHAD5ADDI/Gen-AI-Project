import pandas as pd
import chromadb
from .chunking import chunk_document
from .embedding import Embedder

class IngestionPipeline:
    def __init__(self, persist_directory="./chroma_db_final"):
        self.embedder = Embedder()
        self.chroma_client = chromadb.PersistentClient(path=persist_directory) if persist_directory else chromadb.EphemeralClient()
        
        try:
            self.chroma_client.delete_collection(name="rag_sp_documents")
        except Exception:
            pass
            
        self.collection = self.chroma_client.create_collection(name="rag_sp_documents")
        self.docstore = {} 
        
    def ingest_parquet(self, parquet_path="data/raw/corpus.parquet"):
        df = pd.read_parquet(parquet_path)
        all_chunks = []
        for row in df.to_dict(orient="records"):
            all_chunks.extend(chunk_document(row))
            
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
