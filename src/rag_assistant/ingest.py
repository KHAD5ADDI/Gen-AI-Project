"""Construit et lit l'index Chroma."""

import chromadb
import pandas as pd

from .chunking import chunk_document
from .config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, CORPUS_PATH, EMBED_MODEL
from .embedding import Embedder


class IngestionPipeline:
    """Gère les passages et leurs vecteurs dans Chroma."""

    def __init__(self, persist_directory=CHROMA_DIR, load_docstore=True):
        """Ouvre l'index et charge ses passages, sauf avant une reconstruction."""
        self.chroma_client = (
            chromadb.PersistentClient(path=str(persist_directory))
            if persist_directory else chromadb.EphemeralClient()
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION, configuration={"hnsw": {"space": "cosine"}}
        )
        self.docstore = self._load_docstore() if load_docstore else {}
        self.embedder = Embedder()

    def _load_docstore(self):
        """Relit les passages stockés dans l'index."""
        data = self.collection.get(include=["documents", "metadatas"])
        if data["metadatas"]:
            first_metadata = data["metadatas"][0] or {}
            if "body" not in first_metadata or "embed_model" not in first_metadata:
                raise RuntimeError(
                    "Cet index utilise l'ancien découpage. Reconstruisez-le avec uv run python scripts/run_ingest.py."
                )
        return {
            chunk_id: {**metadata, "chunk_id": chunk_id, "text": document}
            for chunk_id, metadata, document in zip(data["ids"], data["metadatas"], data["documents"])
        }

    def _chunks(self, parquet_path, size, overlap):
        """Découpe les fiches du parquet."""
        df = pd.read_parquet(parquet_path)
        all_chunks = []
        for row in df.to_dict(orient="records"):
            all_chunks.extend(chunk_document(row, size=size, overlap=overlap))
        return all_chunks

    def ingest_parquet(self, parquet_path=CORPUS_PATH, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        """Reconstruit l'index et ses métadonnées à partir du corpus."""
        self.chroma_client.delete_collection(name=COLLECTION)
        self.collection = self.chroma_client.create_collection(
            name=COLLECTION, configuration={"hnsw": {"space": "cosine"}}
        )
        all_chunks = self._chunks(parquet_path, size, overlap)

        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedder.encode_chunks(texts)
        ids = [chunk["chunk_id"] for chunk in all_chunks]
        metadatas = []
        for chunk in all_chunks:
            metadata = {key: value for key, value in chunk.items() if key != "text"}
            metadata["embed_model"] = EMBED_MODEL
            metadata["chunk_size"] = size
            metadata["chunk_overlap"] = overlap
            metadatas.append(metadata)

        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        self.docstore = self._load_docstore()
