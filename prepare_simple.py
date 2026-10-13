"""Construit data/raw/corpus.parquet à partir de l'archive Service-Public.

    uv run python scripts/prepare_simple.py          # génère le corpus
    uv run python scripts/prepare_simple.py --list   # affiche les thèmes disponibles
"""
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ZIP = Path("data/raw/vosdroits-latest.zip")
OUT = Path("data/raw/corpus.parquet")
THEMES = {"Logement", "Papiers - Citoyenneté - Élections"}  # thèmes de 1er niveau retenus

SECTIONS = {"Chapitre", "SousChapitre", "Situation", "Cas"}  # leur <Titre> ouvre une section
IGNORED = {"TitreRiche", "PourEnSavoirPlus", "VoirAussi", "QuestionReponse",
           "NoticeLiee", "ServiceEnLigneAnnexe"}


def tag(e):
    return e.tag.rsplit("}", 1)[-1]  # retire l'espace de noms (dc:title -> title)


def text(e):
    return " ".join("".join(e.itertext()).split()) if e is not None else ""


def child(e, name):
    return next((c for c in e if tag(c) == name), None)


def render(e, path, out):
    """Transforme le XML en lignes de texte ; chaque section devient « ## Chapitre > Cas »."""
    t = tag(e)
    if t in IGNORED:
        return
    if t in SECTIONS:
        title = text(child(e, "Titre"))
        if title:
            path = path + [title]
            out.append("\n## " + " > ".join(path))
        for c in e:
            if tag(c) != "Titre":
                render(c, path, out)
    elif t in ("Paragraphe", "Titre"):
        out.append(text(e))
    elif t == "Item":
        out.append("- " + text(e))
    elif t == "Rangee":  # ligne de tableau
        out.append(" | ".join(text(c) for c in e))
    elif t == "OuSAdresser":
        out.append("Où s'adresser : " + text(child(e, "Titre")))
    elif t == "ServiceEnLigne":
        cerfa = e.get("numerocerfa")
        out.append(f"{e.get('type', 'Service en ligne')} : {text(child(e, 'Titre'))}"
                   + (f" (cerfa n°{cerfa})" if cerfa else ""))
    else:
        for c in e:
            render(c, path, out)


def parse(xml):
    root = ET.fromstring(xml)
    meta = {tag(e): text(e) for e in root if tag(e) in ("title", "description", "date")}
    ariane = child(root, "FilDAriane")
    levels = [text(n) for n in ariane][1:] if ariane is not None else []  # sans « Accueil »

    out = [meta.get("description", "")]
    for c in root:
        if tag(c) in ("Introduction", "Texte", "ListeSituations"):
            render(c, [], out)
    # Blocs de fin de fiche : interlocuteurs et textes de loi.
    for name, label in (("QuiPeutMAider", "Qui peut m'aider ?"), ("Reference", "Textes de référence")):
        items = [c for c in root if tag(c) == name]
        if items:
            out.append(f"\n## {label}")
            for i in items:
                detail = text(child(i, "Complement"))
                out.append(f"- {text(child(i, 'Titre'))}" + (f" ({detail})" if detail else ""))

    date = re.search(r"\d{4}-\d{2}-\d{2}", meta.get("date", ""))
    return {
        "doc_id": root.get("ID"),
        "title": meta.get("title", ""),
        "text": "\n".join(line for line in out if line.strip()).strip(),
        "theme": levels[0] if levels else "",
        "fil_ariane": " > ".join(levels),
        "date": date.group(0) if date else "",
        "url": root.get("spUrl", ""),
        "type": root.get("type", ""),
    }


def main():
    with zipfile.ZipFile(ZIP) as z:
        fiches = [parse(z.read(n)) for n in z.namelist() if re.search(r"(^|/)F\d+\.xml$", n)]

    if "--list" in sys.argv:
        for theme, n in Counter(f["theme"] for f in fiches).most_common():
            print(f"{n:5d}  {theme}")
        return

    df = pd.DataFrame([f for f in fiches if f["theme"] in THEMES and len(f["text"]) >= 200])
    df = df.sort_values("doc_id")
    df.to_parquet(OUT, index=False)

    lengths = df["text"].str.len()
    print(f"{len(df)} fiches écrites dans {OUT}\n")
    print(df["theme"].value_counts().to_string(), "\n")
    print(f"Longueur : médiane {lengths.median():.0f} car., moyenne {lengths.mean():.0f}, max {lengths.max()}")
    print(f"Dates de mise à jour : {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    main()
