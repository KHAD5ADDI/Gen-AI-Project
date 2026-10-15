import pandas as pd
import chromadb
from src.rag_assistant.chunking import chunk_document
from src.rag_assistant.embedding import Embedder

class IngestionPipeline:
    def __init__(self, persist_directory="./chroma_db"):
        self.embedder = Embedder()
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="rag_sp_documents")
        self.docstore = {} # Stockage local requis pour la recherche lexicale (BM25)
        
    def ingest_parquet(self, parquet_path="data/raw/corpus.parquet"):
        """
        Charge le corpus Parquet, le découpe, l'encode et l'indexe.
        """
        print(f"Lecture du corpus depuis {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        all_chunks = []
        for row in df.to_dict(orient="records"):
            all_chunks.extend(chunk_document(row))
            
        if not all_chunks:
            print("Aucun chunk généré. Vérifiez le fichier Parquet.")
            return

        print(f"Encodage de {len(all_chunks)} chunks...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.encode_chunks(texts)
        
        ids = [c["chunk_id"] for c in all_chunks]
        
        # ChromaDB n'accepte pas les valeurs nulles (None) dans les métadonnées
        metadatas = [{
            "doc_id": c["doc_id"], 
            "date": c["date"] if c["date"] else "Inconnue", 
            "title": c["title"] if c["title"] else "Sans titre",
            "body": c["body"]
        } for c in all_chunks]
        
        print("Insertion dans ChromaDB et le docstore...")
        # 1. Indexation Dense (ChromaDB)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        # 2. Indexation Lexicale (Docstore)
        for chunk in all_chunks:
            self.docstore[chunk["chunk_id"]] = chunk
            
        print(f"Ingestion terminée : {len(all_chunks)} chunks indexés à partir de {len(df)} fiches.")
