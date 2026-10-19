"""Création du LLM : Ollama (local) ou OpenRouter (API), choisi par variables d'environnement (.env).

    LLM_PROVIDER        ollama (défaut) ou openrouter   <- seule ligne à changer pour basculer
    OLLAMA_MODEL        modèle Ollama      (défaut : llama3.1:8b)
    OPENROUTER_MODEL    modèle OpenRouter  (défaut : mistralai/mistral-small-2603)
    OPENROUTER_API_KEY  clé OpenRouter (seulement si LLM_PROVIDER=openrouter)
    LLM_NUM_CTX         fenêtre de contexte Ollama, tokens (défaut : 8192)
    LLM_TIMEOUT         délai max par appel, secondes (défaut : 300)

Température 0 partout, comme dans les labs : à question égale, réponse (presque) identique.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # lit le fichier .env à la racine du projet, s'il existe

DEFAULT_MODELS = {"ollama": "llama3.1:8b", "openrouter": "mistralai/mistral-small-2603"}
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def get_llm(model: str | None = None, provider: str | None = None):
    provider = (provider or os.environ.get("LLM_PROVIDER") or "ollama").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"LLM_PROVIDER inconnu : {provider!r} (attendu : ollama ou openrouter)")
    model = model or os.environ.get(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]
    timeout = float(os.environ.get("LLM_TIMEOUT", 300))

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY n'est pas définie (dans le .env ou le terminal).")
        from llama_index.llms.openai_like import OpenAILike
        # OpenRouter parle le même protocole que l'API OpenAI : on change juste l'adresse.
        return OpenAILike(model=model, api_base=OPENROUTER_URL, api_key=key, temperature=0,
                          timeout=timeout, max_retries=3, context_window=128_000,
                          is_chat_model=True, is_function_calling_model=True)

    from llama_index.llms.ollama import Ollama
    num_ctx = int(os.environ.get("LLM_NUM_CTX", 8192))
    # Sans num_ctx explicite, llama-index envoie la fenêtre maximale du modèle (131 072 tokens) :
    # sur 16 Go de RAM, le modèle déborde du GPU et chaque appel dépasse le délai.
    return Ollama(model=model, temperature=0, request_timeout=timeout, context_window=num_ctx,
                  additional_kwargs={"num_ctx": num_ctx})


def provider_of(llm) -> str:
    """Nom du fournisseur d'un LLM créé par get_llm (pour le traçage et le calcul du coût)."""
    return "ollama" if type(llm).__name__ == "Ollama" else "openrouter"
