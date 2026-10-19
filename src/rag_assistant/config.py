"""Prix des modèles, en USD par million de tokens : (entrée, sortie)."""

# Modèles locaux (Ollama) : gratuits en réalité. Comme dans le lab 06, on les compte « comme si »
# les mêmes tokens étaient payés au tarif de Claude Haiku 4.5 (1 USD en entrée, 5 USD en sortie),
# pour pouvoir comparer le coût d'une configuration locale et d'une configuration API.
LOCAL_PRICE = (1.0, 5.0)

# Modèles API (OpenRouter) : prix réels, relevés sur https://openrouter.ai/api/v1/models le 19/10/2026.
API_PRICES = {
    "mistralai/mistral-small-2603": (0.15, 0.60),  # agent API
    "google/gemini-2.5-flash": (0.30, 2.50),  # juge (autre famille que les générateurs)
}


def price_for(provider: str, model: str) -> tuple[float, float]:
    if provider == "ollama":
        return LOCAL_PRICE
    if model not in API_PRICES:  # on n'invente pas de prix : il faut l'ajouter ci-dessus
        raise ValueError(f"Prix inconnu pour {model!r} : ajoute-le dans API_PRICES (config.py).")
    return API_PRICES[model]
