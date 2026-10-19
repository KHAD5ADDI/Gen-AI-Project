"""Exporte des fiches tirées au hasard dans un fichier texte, pour écrire le golden set.

    uv run python export_sample.py                 # 12 fiches Logement + 8 Papiers
    uv run python export_sample.py 15 10 42        # 15 + 10 fiches, graine 42

Le tirage est reproductible : avec la même graine, on obtient les mêmes fiches.
Les fiches déjà utilisées dans eval/golden.jsonl (champ gold_docs) sont exclues du tirage.
Le résultat est écrit dans eval/golden_sources_<graine>.md (un seul fichier à partager).
"""
import json
import sys
from pathlib import Path

import pandas as pd

N_LOGEMENT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_PAPIERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
MAX_CHARS = 12000  # au-delà, le texte est tronqué (les fiches géantes dépassent 200 000 caractères)
OUT = Path(f"eval/golden_sources_{SEED}.md")
GOLDEN = Path("eval/golden.jsonl")

df = pd.read_parquet("data/raw/corpus.parquet")
if GOLDEN.exists():  # ne pas retirer des fiches déjà couvertes par le golden set
    used = {d for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()
            for d in json.loads(line).get("gold_docs", [])}
    df = df[~df["doc_id"].isin(used)]
    print(f"{len(used)} fiches déjà dans {GOLDEN} exclues du tirage")
sample = pd.concat([
    df[df["theme"] == "Logement"].sample(N_LOGEMENT, random_state=SEED),
    df[df["theme"] == "Papiers - Citoyenneté - Élections"].sample(N_PAPIERS, random_state=SEED),
])

parts = [f"# Fiches pour le golden set (graine {SEED})\n"]
for r in sample.itertuples():
    truncated = len(r.text) > MAX_CHARS
    parts.append(
        f"\n\n========== {r.doc_id} ==========\n"
        f"Titre : {r.title}\nFil d'Ariane : {r.fil_ariane}\nURL : {r.url}\n"
        f"Mise à jour : {r.date} · {len(r.text)} caractères"
        + (f" (tronqué à {MAX_CHARS})" if truncated else "") + "\n\n"
        + r.text[:MAX_CHARS] + ("\n[... suite tronquée ...]" if truncated else "")
    )

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("".join(parts), encoding="utf-8")
print(f"{len(sample)} fiches écrites dans {OUT} ({OUT.stat().st_size // 1024} Ko) :")
for r in sample.itertuples():
    print(f"  {r.doc_id:>7}  {r.title}")
