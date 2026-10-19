"""Traçage de chaque réponse : appels au LLM, tokens, latence, appels d'outils et coût."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import price_for
from .llm import provider_of


def _get(obj: Any, *path: str) -> Any:
    """Lit une valeur imbriquée dans un dictionnaire ou un objet, ou renvoie None si elle manque."""
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


def extract_usage(raw: Any) -> dict:
    """Lit tokens d'entrée, tokens de sortie et raison d'arrêt dans la réponse brute d'Ollama ou d'OpenAI."""
    if _get(raw, "prompt_eval_count") is not None:  # Ollama
        return {
            "tokens_in": _get(raw, "prompt_eval_count"),
            "tokens_out": _get(raw, "eval_count"),
            "stop_reason": _get(raw, "done_reason"),
        }
    choices = _get(raw, "choices") or [None]  # OpenAI
    return {
        "tokens_in": _get(raw, "usage", "prompt_tokens"),
        "tokens_out": _get(raw, "usage", "completion_tokens"),
        "stop_reason": _get(choices[0], "finish_reason"),
    }


@dataclass
class Trace:
    """Journal d'une question : appels au LLM, appels d'outils, durées et erreurs."""

    provider: str | None = None  # rempli au premier appel si absent (Trace() dans app.py)
    model: str | None = None
    llm_calls: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @contextmanager
    def timer(self, name: str):
        """Chronomètre une étape (with trace.timer("retrieval"): ...) et cumule sa durée."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(self.timings.get(name, 0.0) + time.perf_counter() - t0, 3)

    def record_llm(self, purpose: str, response: Any, latency: float) -> None:
        """Enregistre un appel au LLM avec son rôle, sa durée et ses tokens."""
        self.llm_calls.append({"purpose": purpose, "latency_s": round(latency, 3), **extract_usage(response.raw)})

    def record_tool(self, name: str, args: dict, latency: float, output_chars: int) -> None:
        """Enregistre un appel d'outil de l'agent avec ses arguments, sa durée et la taille du résultat."""
        self.tool_calls.append({"tool": name, "args": args, "latency_s": round(latency, 3), "output_chars": output_chars})

    def summary(self) -> dict:
        """Renvoie les totaux de la question, tels qu'écrits dans eval/results.jsonl."""
        tokens_in = sum(c["tokens_in"] or 0 for c in self.llm_calls)
        tokens_out = sum(c["tokens_out"] or 0 for c in self.llm_calls)
        price_in, price_out = price_for(self.provider, self.model) if self.llm_calls else (0, 0)
        return {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_missing": any(c["tokens_in"] is None for c in self.llm_calls),
            "cost_usd": round((tokens_in * price_in + tokens_out * price_out) / 1e6, 6),
            "n_llm_calls": len(self.llm_calls),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "stop_reason": self.llm_calls[-1]["stop_reason"] if self.llm_calls else None,
            "timings_s": self.timings,
            "errors": self.errors,
        }


def chat(llm, messages, trace: Trace, purpose: str, tools=None):
    """Appelle le LLM (avec outils en mode agent) et enregistre l'appel dans la trace."""
    if trace.model is None:
        trace.provider, trace.model = provider_of(llm), llm.model
    t0 = time.perf_counter()
    if tools:
        response = llm.chat_with_tools(tools, chat_history=messages, allow_parallel_tool_calls=False)
    else:
        response = llm.chat(messages)
    trace.record_llm(purpose, response, time.perf_counter() - t0)
    return response
