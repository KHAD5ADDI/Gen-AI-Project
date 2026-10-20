"""Trace minimale de la boucle ReAct."""

from rag_assistant.config import price_for


def cout_usd(tokens_in, tokens_out, provider, model):
    """Calcule le coût estimé en dollars."""
    price_in, price_out = price_for(provider, model)
    return round((tokens_in * price_in + tokens_out * price_out) / 1_000_000, 6)


class Trace:
    """Conserve les étapes, outils, tokens et durée d'une question."""

    def __init__(self):
        self.steps = []
        self.tool_calls = []
        self.status = None
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0

    def record(self, result):
        """Copie les compteurs et les vrais appels d'outils."""
        self.steps = result["steps"]
        self.status = result["status"]
        self.tokens_in = result["tokens_in"]
        self.tokens_out = result["tokens_out"]
        self.seconds = result["seconds"]
        self.tool_calls = [
            {
                "tool": step["action"][0],
                "input": step["action"][1],
                "seconds": step["tool_seconds"],
                "output_chars": step["output_chars"],
            }
            for step in self.steps
            if step["action"] is not None
            and step["action"][0] != "final_answer"
            and step["tool_seconds"] is not None
        ]
