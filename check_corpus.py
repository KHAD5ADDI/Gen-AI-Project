"""Contrôles automatiques du corpus produit par prepare_simple.py.

    uv run python check_corpus.py                      # tous les contrôles
    uv run python check_corpus.py --find "carte d'identité"  # chercher des fiches par titre
    uv run python check_corpus.py --show F36798        # afficher une fiche en entier

Chaque contrôle affiche OK, ATTENTION (à regarder, pas forcément une erreur) ou ERREUR.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:  # mêmes réglages que la préparation (prepare_docs.py ou prepare_simple.py)
    from prepare_docs import OUT, THEMES
except ImportError:
    from prepare_simple import OUT, THEMES

XML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*[ >/]")  # reste de balise XML, ex. « <Paragraphe> »
problems = 0


def report(ok, label, detail="", warn_only=False):
    """Affiche le résultat d'un contrôle ; compte les erreurs bloquantes."""
    global problems
    status = "OK       " if ok else ("ATTENTION" if warn_only else "ERREUR   ")
    if not ok and not warn_only:
        problems += 1
    print(f"[{status}] {label}" + (f" : {detail}" if detail else ""))


def duplicated_titles(text):
    """Titre de section répété sur la ligne suivante (« ## A > B » puis « B »)."""
    lines = text.splitlines()
    return [l for l, nxt in zip(lines, lines[1:])
            if l.startswith("## ") and nxt.strip() == l.split(" > ")[-1].removeprefix("## ").strip()]


def orphan_colons(text):
    """Phrase finissant par « : » suivie d'un titre ou de la fin : contenu peut-être perdu."""
    lines = [l for l in text.splitlines() if l.strip()]
    return [l for l, nxt in zip(lines, lines[1:] + ["## fin"])
            if l.rstrip().endswith(":") and nxt.startswith("## ")]


def main():
    df = pd.read_parquet(OUT)

    if "--show" in sys.argv:  # affichage complet d'une fiche, à comparer avec le site
        row = df[df["doc_id"] == sys.argv[sys.argv.index("--show") + 1]].iloc[0]
        print(f"{row.doc_id} · {row.title}\n{row.fil_ariane}\n{row.url}\n{'-' * 80}\n{row.text}")
        return
    if "--find" in sys.argv:  # recherche de fiches par mot du titre
        needle = sys.argv[sys.argv.index("--find") + 1].lower()
        hits = df[df["title"].str.lower().str.contains(needle, regex=False)]
        print(hits[["doc_id", "title"]].to_string(index=False))
        return

    print(f"{len(df)} fiches dans {OUT}\n")

    # 1. Structure du tableau
    expected = {"doc_id", "title", "text", "theme", "fil_ariane", "date", "url", "type"}
    report(expected <= set(df.columns), "Colonnes attendues", ", ".join(sorted(expected - set(df.columns))))

    # 2. Identifiants : uniques et au format F + chiffres
    report(df["doc_id"].is_unique, "Identifiants uniques", f"{df['doc_id'].duplicated().sum()} doublons")
    bad_ids = df.loc[~df["doc_id"].str.fullmatch(r"F\d+"), "doc_id"].tolist()
    report(not bad_ids, "Identifiants au format F1234", str(bad_ids[:5]))

    # 3. Thèmes : uniquement ceux demandés, et tous présents
    themes = set(df["theme"])
    report(themes == THEMES, "Thèmes = THEMES", f"en trop {themes - THEMES}, absents {THEMES - themes}")

    # 4. Champs vides
    for col in ("title", "text", "date", "url"):
        empty = (df[col].fillna("").str.strip() == "").sum()
        report(empty == 0, f"Aucun champ '{col}' vide", f"{empty} vides", warn_only=(col == "date"))

    # 5. Formats des dates et des URL
    bad_dates = df.loc[(df["date"] != "") & ~df["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}"), "doc_id"]
    report(bad_dates.empty, "Dates au format AAAA-MM-JJ", str(bad_dates.tolist()[:5]))
    bad_urls = df.loc[~df["url"].str.startswith("https://www.service-public"), "doc_id"]
    report(bad_urls.empty, "URL Service-Public", str(bad_urls.tolist()[:5]))

    # 6. Nettoyage : plus aucune balise XML dans le texte
    with_tags = df[df["text"].str.contains(XML_TAG)]
    report(with_tags.empty, "Aucune balise XML restante", f"{len(with_tags)} fiches, ex. {with_tags['doc_id'].tolist()[:3]}")

    # 7. Titres en double (le bug « ## Cas général / Cas général »)
    dups = df["text"].map(duplicated_titles)
    n_dups = dups.map(len).sum()
    report(n_dups == 0, "Aucun titre de section répété", f"{n_dups} cas, ex. {dups[dups.map(len) > 0].head(1).tolist()}")

    # 8. Contenu perdu après « : » (le bug des formulaires manquants)
    orphans = df["text"].map(orphan_colons)
    n_orph = orphans.map(len).sum()
    report(n_orph == 0, "Aucune phrase « : » sans suite",
           f"{n_orph} cas dans {(orphans.map(len) > 0).sum()} fiches", warn_only=True)

    # 9. Structure en sections
    n_sections = df["text"].str.count(r"(?m)^## ")
    report((n_sections > 0).mean() > 0.9, "Fiches découpées en sections",
           f"{(n_sections == 0).sum()} fiches sans aucune section", warn_only=True)

    # 10. Vérification ponctuelle sur une fiche connue (si elle est dans le corpus)
    if "F36798" in set(df["doc_id"]):
        txt = df.loc[df["doc_id"] == "F36798", "text"].iloc[0]
        report("16702" in txt and "Où s'adresser : Mairie" in txt, "F36798 : cerfa et interlocuteur présents")

    # Aperçu pour la data card
    lengths = df["text"].str.len()
    print(f"\nLongueur : médiane {lengths.median():.0f}, moyenne {lengths.mean():.0f}, max {lengths.max()} car.")
    print(f"Sections par fiche : médiane {n_sections.median():.0f}, max {n_sections.max()}")
    print("5 fiches les plus longues :")
    for _, r in df.assign(n=lengths).nlargest(5, "n").iterrows():
        print(f"   {r.doc_id:>7}  {r.n:7d} car.  {r.title}")
    if n_orph:
        print("\nExemples de « : » sans suite (à vérifier avec --show et sur le site) :")
        examples = [(d, items[0]) for d, items in zip(df["doc_id"], orphans) if items]
        for doc_id, line in examples[:10]:
            print(f"   {doc_id}  …{line[-90:]}")

    print(f"\n{'Aucune erreur bloquante.' if problems == 0 else f'{problems} erreur(s) à corriger.'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
