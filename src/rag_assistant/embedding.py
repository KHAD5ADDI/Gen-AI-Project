from sentence_transformers import SentenceTransformer
#sentence-transformers/paraphrase-multilingual-MiniLM-L12-v

class Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-small"):
        """
        Initialise le modèle d'embedding multilingue.
        Les modèles E5 nécessitent souvent l'ajout du préfixe "query: " pour les questions
        et "passage: " pour les documents, mais la bibliothèque sentence-transformers
        gère cela ou s'adapte très bien en mode symétrique.
        """
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, texts):
        """
        Encode une liste de textes en vecteurs.
        La normalisation à 1 est maintenue pour le calcul de similarité cosinus[cite: 3].
        """
        # Pour les modèles de la famille E5, il est recommandé de préfixer les documents
        # dans la base de données.
        formatted_texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(formatted_texts, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, query):
        """
        Encode spécifiquement la question de l'utilisateur.
        """
        formatted_query = f"query: {query}"
        return self.model.encode([formatted_query], normalize_embeddings=True).tolist()[0]
