from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialise le modèle d'embedding.
        Par défaut, all-MiniLM-L6-v2 génère des vecteurs de 384 valeurs.
        """
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, texts):
        """
        Encode une liste de textes en vecteurs.
        La normalisation à 1 est requise pour que le produit scalaire équivaille à la similarité cosinus.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
