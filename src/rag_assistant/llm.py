"""Création du LLM : Ollama (local) ou OpenAI, choisi par variables d'environnement (.env).

    LLM_PROVIDER    ollama (défaut) ou openai   <- seule ligne à changer pour basculer
    OLLAMA_MODEL    modèle Ollama  (défaut : llama3.1:8b)
    OPENAI_MODEL    modèle OpenAI  (défaut : gpt-5.4-mini)
    OPENAI_API_KEY  clé OpenAI (seulement si LLM_PROVIDER=openai)
    LLM_NUM_CTX     fenêtre de contexte Ollama, tokens (défaut : 8192)
    LLM_TIMEOUT     délai max par appel, secondes (défaut : 300)
"""
import os

try:  # charge un fichier .env s'il existe (paquet python-dotenv, optionnel)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_MODELS = {"ollama": "llama3.1:8b", "openai": "gpt-5.4-mini"}


def get_llm(model: str | None = None, provider: str | None = None):
    provider = (provider or os.environ.get("LLM_PROVIDER") or "ollama").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER inconnu : {provider!r} (attendu : ollama ou openai)")
    model = model or os.environ.get(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    timeout = float(os.environ.get("LLM_TIMEOUT", 300))

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY n'est pas définie (dans le .env ou le terminal).")
        from llama_index.llms.openai import OpenAI
        try:
            return OpenAI(model=model, timeout=timeout, max_retries=3)
        except ValueError as err:  # nom de modèle inconnu de cette version du paquet
            raise RuntimeError(f"Modèle {model!r} non reconnu : mets à jour le paquet "
                               "(uv pip install -U llama-index-llms-openai).") from err

    from llama_index.llms.ollama import Ollama
    num_ctx = int(os.environ.get("LLM_NUM_CTX", 8192))
    # Sans num_ctx explicite, Ollama garde une fenêtre courte et tronque le prompt
    # sans prévenir : l'agent perdrait la question ou les règles.
    return Ollama(model=model, request_timeout=timeout, context_window=num_ctx,
                  additional_kwargs={"num_ctx": num_ctx})
