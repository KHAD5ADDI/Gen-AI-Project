"""LLM configuré par .env : LLM_PROVIDER, OLLAMA_MODEL, OPENROUTER_MODEL,
OPENROUTER_API_KEY, LLM_NUM_CTX et LLM_TIMEOUT.
"""

import os
import time

import ollama
from dotenv import load_dotenv
from openai import OpenAI

from rag_assistant.schemas import Usage

load_dotenv()

LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))


class OllamaLLM:
    """Modèle local appelé avec le protocole du lab 06."""

    def __init__(self, name):
        self.provider = "ollama"
        self.name = name
        self.client = ollama.Client(timeout=LLM_TIMEOUT)

    def _chat(self, system, user, max_new_tokens, stop):
        options = {"temperature": 0, "num_predict": max_new_tokens, "num_ctx": LLM_NUM_CTX}
        if stop:
            options["stop"] = ["Observation:"]
        # Sans num_ctx, Ollama utilise une fenêtre trop petite et coupe le prompt en silence.
        start = time.perf_counter()
        response = self.client.chat(
            model=self.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options=options,
        )
        seconds = time.perf_counter() - start
        content = response.message.content or ""
        text = content.split("Observation:")[0].strip() if stop else content.strip()
        return text, Usage(response.prompt_eval_count or 0, response.eval_count or 0, seconds)

    def next_step(self, system, user, max_new_tokens=300):
        """Produit la prochaine action ReAct."""
        return self._chat(system, user, max_new_tokens, stop=True)

    def generate(self, system, user, max_new_tokens=500):
        """Produit un texte pour le pipeline RAG classique."""
        return self._chat(system, user, max_new_tokens, stop=False)


class OpenRouterLLM:
    """Modèle API appelé avec le protocole OpenAI."""

    def __init__(self, name):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY manque dans l'environnement ou le fichier .env.")
        self.provider = "openrouter"
        self.name = name
        # Les limites temporaires du pool partagé justifient des tentatives supplémentaires.
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=LLM_TIMEOUT, max_retries=8)

    def _chat(self, system, user, max_new_tokens, stop):
        kwargs = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_new_tokens,
        }
        if stop:
            kwargs["stop"] = ["Observation:"]
        start = time.perf_counter()
        # Même protocole ReAct pour chaque modèle, sans tokens de raisonnement cachés (profondeur et coût).
        response = self.client.chat.completions.create(**kwargs, extra_body={"reasoning": {"enabled": False}})
        seconds = time.perf_counter() - start
        content = response.choices[0].message.content or ""
        text = content.split("Observation:")[0].strip() if stop else content.strip()
        usage = response.usage
        return text, Usage(usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0, seconds)

    def next_step(self, system, user, max_new_tokens=300):
        """Produit la prochaine action ReAct."""
        return self._chat(system, user, max_new_tokens, stop=True)

    def generate(self, system, user, max_new_tokens=500):
        """Produit un texte pour le pipeline RAG classique."""
        return self._chat(system, user, max_new_tokens, stop=False)


def get_llm(provider=None, model=None):
    """Choisit le modèle selon les arguments ou le fichier .env."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()
    if provider == "ollama":
        return OllamaLLM(model or os.getenv("OLLAMA_MODEL") or "llama3.1:8b")
    if provider == "openrouter":
        return OpenRouterLLM(model or os.getenv("OPENROUTER_MODEL") or "mistralai/mistral-small-2603")
    raise ValueError(f"Fournisseur LLM inconnu : {provider}")
