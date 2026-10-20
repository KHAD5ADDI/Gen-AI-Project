"""Tests du juge sans appel réseau."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from judge import Verdict, grade, judge_one, sources_text


class FakeClient:
    def __init__(self, verdict=None, fail=False):
        self.calls = []
        self.verdict = verdict or Verdict(correct="yes", supported_by_sources=True, reason="Conforme à la référence.")
        self.fail = fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("indisponible")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self.verdict))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def result(answerable=True):
    return {
        "id": "q001", "type": "simple", "split": "dev", "question": "Question", "answerable": answerable,
        "generated_answer": "Réponse", "steps": [],
    }


def gold(docs=None):
    return {"gold_docs": ["F1"] if docs is None else docs, "answer": "Référence"}


def test_judge_one_retourne_verdict_et_usage():
    client = FakeClient()
    verdict, usage = judge_one(client, "Question", "Référence", "Source", "Réponse")
    assert verdict.correct == "yes"
    assert (usage.prompt_tokens, usage.completion_tokens) == (10, 5)
    assert client.calls[0]["response_format"] is Verdict
    assert client.calls[0]["temperature"] == 0


def test_judge_one_echec_compte_comme_non():
    verdict, usage = judge_one(FakeClient(fail=True), "Question", "Référence", "Source", "Réponse")
    assert (verdict.correct, verdict.reason, usage) == ("no", "judge call failed", None)


def test_grade_refus_attendu_sans_appel():
    client = FakeClient(fail=True)
    refused = grade(result(answerable=False), gold([]), client)
    answered = grade(result(), gold([]), client)
    assert (refused["correct"], refused["judged_by"], refused["score"]) == ("yes", "règle", 1.0)
    assert (answered["correct"], answered["judged_by"], answered["score"]) == ("no", "règle", 0.0)
    assert refused["supported_by_sources"] is True
    assert answered["supported_by_sources"] is False
    assert client.calls == []


def test_grade_reponse_attendue_absente_sans_appel():
    client = FakeClient(fail=True)
    row = grade(result(answerable=False), gold(), client)
    assert (row["correct"], row["judged_by"], row["reason"]) == (
        "no", "règle", "Pas de réponse alors qu'une réponse était attendue."
    )
    assert client.calls == []


def test_grade_reponse_normale_appelle_juge_une_fois():
    client = FakeClient()
    row = grade(result(), gold(), client)
    assert (row["correct"], row["judged_by"], row["judge_tokens_in"], row["judge_tokens_out"]) == (
        "yes", "juge", 10, 5
    )
    assert len(client.calls) == 1


def test_sources_text_filtre_et_limite():
    steps = [
        {"action": ["search", "x"], "observation": "abcd"},
        {"action": ["final_answer", "x"], "observation": "ignoré"},
        {"action": ["get_document", "F1"], "observation": "efgh"},
        {"action": None, "observation": "ignoré"},
    ]
    assert sources_text({"steps": steps}) == "abcd\n\nefgh"
    assert sources_text({"steps": steps}, max_chars=6) == "abcd\n\n"
