from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL

class Embedder:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, texts):
        # Préfixe obligatoire pour les passages avec E5
        formatted_texts = [f"passage: {text}" for text in texts]
        return self.model.encode(formatted_texts, normalize_embeddings=True).tolist()
        
    def encode_query(self, query):
        # Préfixe obligatoire pour les questions avec E5
        formatted_query = f"query: {query}"
        return self.model.encode([formatted_query], normalize_embeddings=True).tolist()[0]
