import os
import glob
import chromadb
from src.rag_assistant.chunking import parse_xml_document, chunk_document
from src.rag_assistant.embedding import Embedder

class IngestionPipeline:
    def __init__(self, persist_directory="./chroma_db"):
        self.embedder = Embedder()
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="rag_xml_documents")
        self.docstore = {} # Stockage local requis pour la recherche lexicale (BM25)
        
    def ingest_xml_directory(self, xml_dir):
        """
        Lit tous les fichiers XML du dossier, les parse, les découpe et les indexe.
        """
        xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
        all_chunks = []
        
        for file_path in xml_files:
            raw_doc = parse_xml_document(file_path)
            all_chunks.extend(chunk_document(raw_doc))
            
        if not all_chunks:
            print("Aucun chunk généré. Vérifiez le dossier XML.")
            return

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.encode_chunks(texts)
        
        ids = [c["chunk_id"] for c in all_chunks]
        metadatas = [{"doc_id": c["doc_id"], "date": c["date"], "title": c["title"], "body": c["body"]} for c in all_chunks]
        
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
            
        print(f"Ingestion terminée : {len(all_chunks)} chunks indexés à partir de {len(xml_files)} fichiers XML.")
