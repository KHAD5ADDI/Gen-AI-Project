"""Évalue la recherche seule (sans LLM) : la bonne fiche est-elle retrouvée à partir de la question brute ?

Usage (depuis la racine du repo, après run_ingest.py) :
    uv run python eval/eval_retrieval.py

Pour chaque question qui a une fiche attendue, donne le rang de cette fiche :
  - dans le pool de candidats (fusion RRF de la recherche dense et BM25, avant reranking),
  - dans la recherche dense seule et dans BM25 seul (30 premiers passages),
  - après reranking (c'est ce classement que voit l'agent : il reçoit les 4 premiers passages).
Si results.jsonl existe, compare avec ce que l'agent a réellement trouvé.
"""
import torch  # à charger en premier : évite un conflit de DLL sous Windows
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_assistant.ingest import IngestionPipeline
from src.rag_assistant.retrieval import HybridRetriever, rrf

PRE_K = 30  # même valeur que dans HybridRetriever.search


def rang_fiche(chunk_ids, docstore, gold):
    """Rang (1 = premier) de la première fiche attendue dans une liste de passages, en comptant par fiche."""
    fiches = []
    for cid in chunk_ids:
        d = docstore[cid]["doc_id"]
        if d not in fiches:
            fiches.append(d)
    rangs = [fiches.index(g) + 1 for g in gold if g in fiches]
    return min(rangs) if rangs else None


def rang_passage(chunk_ids, docstore, gold):
    """Rang du premier passage appartenant à une fiche attendue (l'agent voit les passages 1 à 4)."""
    for i, cid in enumerate(chunk_ids, start=1):
        if docstore[cid]["doc_id"] in gold:
            return i
    return None


def fmt(r):
    return "-" if r is None else str(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    p.add_argument("--results", default=str(ROOT / "eval" / "results.jsonl"))
    p.add_argument("--out", default=str(ROOT / "eval" / "retrieval_results.jsonl"))
    args = p.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        questions = [q for q in (json.loads(l) for l in f if l.strip()) if q.get("gold_docs")]

    agent = {}
    if Path(args.results).exists():
        with open(args.results, encoding="utf-8") as f:
            for r in (json.loads(l) for l in f if l.strip()):
                agent[r["id"]] = r

    pipeline = IngestionPipeline(persist_directory=str(ROOT / "chroma_db_final"))
    if pipeline.collection.count() == 0:
        sys.exit("Index vide : lance d'abord `uv run python run_ingest.py`.")
    ret = HybridRetriever(ingestion_pipeline=pipeline)
    ds = ret.docstore

    print(f"{len(questions)} questions avec fiche attendue\n")
    print(f"{'id':5} {'dense':>5} {'bm25':>5} {'pool':>5} {'rerank':>6} {'passage':>7}  {'agent':5}  question")
    rows = []
    for q in questions:
        gold = set(q["gold_docs"])
        dense = ret.dense_search(q["question"], top_n=PRE_K)
        bm25 = ret.bm25_search(q["question"], top_n=PRE_K)
        pool = rrf([dense, bm25], k=60, depth=PRE_K)
        scores = ret.reranker.predict([[q["question"], ds[c]["text"]] for c in pool]) if pool else []
        reranked = [c for _, c in sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)]

        row = {
            "id": q["id"],
            "gold_docs": q["gold_docs"],
            "rang_dense": rang_fiche(dense, ds, gold),
            "rang_bm25": rang_fiche(bm25, ds, gold),
            "rang_pool": rang_fiche(pool, ds, gold),
            "rang_rerank": rang_fiche(reranked, ds, gold),
            "rang_passage_rerank": rang_passage(reranked, ds, gold),
            "top4_fiches": [ds[c]["doc_id"] for c in reranked[:4]],
            "agent_a_trouve": (any(d in agent[q["id"]]["retrieved_document_IDs"] for d in gold)
                               if q["id"] in agent else None),
        }
        rows.append(row)
        ag = {True: "oui", False: "non", None: "?"}[row["agent_a_trouve"]]
        print(f"{q['id']:5} {fmt(row['rang_dense']):>5} {fmt(row['rang_bm25']):>5} {fmt(row['rang_pool']):>5} "
              f"{fmt(row['rang_rerank']):>6} {fmt(row['rang_passage_rerank']):>7}  {ag:5}  {q['question'][:60]}")

    n = len(rows)
    taux = lambda cle, k: sum(r[cle] is not None and r[cle] <= k for r in rows)
    print("\nRÉSUMÉ (rang de la fiche attendue, « - » = absente)")
    print(f"  Recherche dense seule, dans les 30 passages : {taux('rang_dense', 99)}/{n}")
    print(f"  BM25 seul, dans les 30 passages            : {taux('rang_bm25', 99)}/{n}")
    print(f"  Pool de candidats avant reranking          : {taux('rang_pool', 99)}/{n}")
    print(f"  Après reranking, fiche n°1                 : {taux('rang_rerank', 1)}/{n}")
    print(f"  Après reranking, dans les 4 premiers passages (ce que voit l'agent) : {taux('rang_passage_rerank', 4)}/{n}")
    print(f"  Après reranking, dans les 10 premiers passages : {taux('rang_passage_rerank', 10)}/{n}")
    if agent:
        vus = [r for r in rows if r["agent_a_trouve"] is not None]
        print(f"  L'agent (ses propres requêtes) a trouvé la fiche : {sum(r['agent_a_trouve'] for r in vus)}/{len(vus)}")
        perdu = [r["id"] for r in vus if not r["agent_a_trouve"] and r["rang_passage_rerank"] and r["rang_passage_rerank"] <= 4]
        gagne = [r["id"] for r in vus if r["agent_a_trouve"] and not (r["rang_passage_rerank"] and r["rang_passage_rerank"] <= 4)]
        print(f"  Trouvée par la question brute mais ratée par l'agent : {', '.join(perdu) or 'aucune'}")
        print(f"  Ratée par la question brute mais trouvée par l'agent : {', '.join(gagne) or 'aucune'}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nDétail : {args.out}")


if __name__ == "__main__":
    main()
