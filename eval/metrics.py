"""Métriques de recherche du lab 08 (Session 8)."""

import math


def recall_at_k(retrieved, gold, k=5):
    """Part des fiches attendues trouvées dans les k premières. None s'il n'y a pas de fiche attendue."""
    if not gold:
        return None
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def mrr(retrieved, gold):
    """1 / rang de la première fiche attendue, 0.0 si aucune. None s'il n'y a pas de fiche attendue."""
    if not gold:
        return None
    for rank, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved, gold, k=5):
    """Pertinence binaire : DCG des fiches attendues du top k, divisé par le DCG idéal.

    None s'il n'y a pas de fiche attendue.
    """
    if not gold:
        return None
    dcg = sum(1 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], start=1) if item in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def fiches(chunk_ids):
    """Garde chaque fiche une seule fois, dans l'ordre de son premier passage."""
    seen = set()
    result = []
    for chunk_id in chunk_ids:
        fiche = chunk_id.split("#", 1)[0]
        if fiche not in seen:
            seen.add(fiche)
            result.append(fiche)
    return result
