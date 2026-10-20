"""Aperçu du contenu du corpus préparé.

    uv run python scripts/inspect_corpus.py
"""
import pandas as pd

from rag_assistant.config import CORPUS_PATH

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 70)

df = pd.read_parquet(CORPUS_PATH)
lengths = df["text"].str.len()

print("=== Structure du tableau ===")
df.info(memory_usage="deep")

print("\n=== 10 premières lignes (sans le texte) ===")
print(df[["doc_id", "title", "theme", "date", "type"]].head(10).to_string(index=False))

print("\n=== Répartition par thème et par type ===")
print(pd.crosstab(df["theme"], df["type"]).to_string())

print("\n=== 15 sous-thèmes les plus fournis ===")
sub = df["fil_ariane"].str.split(" > ").str[:2].str.join(" > ")
print(sub.value_counts().head(15).to_string())

print("\n=== Longueur des textes (caractères) ===")
print(lengths.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round().to_string())

print("\n=== Année de dernière mise à jour ===")
print(df["date"].str[:4].value_counts().sort_index().to_string())

print("\n=== Titres en double ===")
dup = df[df["title"].duplicated(keep=False)].sort_values("title")
print(dup[["doc_id", "title"]].to_string(index=False) if len(dup) else "Aucun")

print("\n=== Début d'une fiche tirée au hasard ===")
row = df.sample(1, random_state=7).iloc[0]
print(f"{row.doc_id} · {row.title} · {row.date}\n{row.url}\n{'-' * 80}\n{row.text[:1500]}")
