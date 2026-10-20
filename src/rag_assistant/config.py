"""Réglages du projet : chemins et paramètres partagés, en un seul endroit.

Les chemins partent de la racine du dépôt (et non du dossier courant) : les scripts
marchent quel que soit l'endroit d'où on les lance.
"""
from pathlib import Path

# --- Chemins ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # src/rag_assistant/config.py -> racine du dépôt
RAW_DIR = ROOT / "data" / "raw"
RAW_ZIP = RAW_DIR / "vosdroits-latest.zip"  # archive DILA téléchargée sur data.gouv.fr
CORPUS_PATH = RAW_DIR / "corpus.parquet"  # sortie de scripts/prepare_docs.py
CHROMA_DIR = ROOT / "chroma_db_final"  # index vectoriel (non versionné)
COLLECTION = "rag_sp_documents"
GOLDEN_PATH = ROOT / "eval" / "golden.jsonl"

# --- Modèles ---------------------------------------------------------------
EMBED_MODEL = "intfloat/multilingual-e5-small"
# Juge d'une autre famille que les générateurs Meta et Mistral.
JUDGE_MODEL = "google/gemini-2.5-flash"

# Découpage des passages, en mots.
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

# --- Prix, en USD par million de tokens : (entrée, sortie) --------------------
# Modèles locaux (Ollama) : gratuits en réalité. Comme dans le lab 06, on les compte « comme si »
# les mêmes tokens étaient payés au tarif de Claude Haiku 4.5 (1 USD en entrée, 5 USD en sortie),
# pour pouvoir comparer le coût d'une configuration locale et d'une configuration API.
LOCAL_PRICE = (1.0, 5.0)

# Modèles API (OpenRouter) : prix réels, relevés sur https://openrouter.ai/api/v1/models le 19/10/2026.
API_PRICES = {
    "mistralai/mistral-small-2603": (0.15, 0.60),  # agent API
    "deepseek/deepseek-v4.1-flash": (0.033, 0.080),  # générateur API (modèle du lab 08)
    "google/gemini-2.5-flash": (0.30, 2.50),  # juge (autre famille que les générateurs)
}


def price_for(provider: str, model: str) -> tuple[float, float]:
    if provider == "ollama":
        return LOCAL_PRICE
    if model not in API_PRICES:  # on n'invente pas de prix : il faut l'ajouter ci-dessus
        raise ValueError(f"Prix inconnu pour {model!r} : ajoute-le dans API_PRICES (config.py).")
    return API_PRICES[model]
