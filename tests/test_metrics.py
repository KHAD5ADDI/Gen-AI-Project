import math

from eval.metrics import fiches, mrr, ndcg_at_k, recall_at_k


def test_exemples_du_lab():
    ranking = ["C", "A", "B"]          # l'exemple du cours : seule A est pertinente
    assert recall_at_k(ranking, ["A"], 1) == 0 and recall_at_k(ranking, ["A"], 5) == 1
    assert mrr(ranking, ["A"]) == 0.5
    assert abs(ndcg_at_k(ranking, ["A"], 5) - 1 / math.log2(3)) < 1e-9
    assert recall_at_k(["A", "C", "B"], ["A", "B"], 2) == 0.5 and mrr(["C", "D"], ["A"]) == 0.0
    assert recall_at_k(ranking, [], 5) is None and mrr(ranking, []) is None and ndcg_at_k(ranking, [], 5) is None


def test_exemples_de_la_session_8():
    assert round(ndcg_at_k(["C", "A", "B"], ["A"], 5), 2) == 0.63
    assert sum(mrr(ranking, ["A"]) for ranking in (["C", "A"], ["A"], ["C", "D", "B", "A"])) / 3 == (1/2 + 1 + 1/4) / 3


def test_fiches_garde_le_premier_passage():
    assert fiches(["F1#0", "F2#3", "F1#5", "F3#1"]) == ["F1", "F2", "F3"]
