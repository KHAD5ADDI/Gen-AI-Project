
"""Construit le corpus de l'assistant à partir de l'archive Service-Public.

Source : vosdroits-latest.zip, publié sur data.gouv.fr par la DILA. L'archive
contient 5 573 fichiers XML, un par page du site service-public.gouv.fr
(rubrique Particuliers) :
- 3 003 fiches pratiques (F...), le contenu utilisé ici ;
- 2 318 ressources (R...) : formulaires, téléservices, adresses, définitions ;
- 243 pages de navigation (N...) : thèmes, sous-thèmes et dossiers ;
- 9 autres fichiers techniques.
Les ressources et pages de navigation ne sont pas lues directement : les
informations utiles des ressources (titre d'un formulaire, numéro cerfa,
interlocuteur) figurent déjà dans les fiches qui les citent.

Fiches extraites, en trois filtres successifs :
1. uniquement les fiches pratiques (fichiers F*.xml) : 3 003 fiches ;
2. dont le thème principal (1er niveau du fil d'Ariane) est « Logement » ou
   « Papiers - Citoyenneté - Élections » : 549 fiches (350 + 199). Les deux
   thèmes sont pris en entier, pour que toute question sur ces sujets trouve
   sa réponse dans le corpus ;
3. dont le texte converti fait au moins 200 caractères (en dessous, ce sont
   des pages quasi vides) : 23 fiches retirées.
Résultat : 526 fiches (333 Logement et 193 Papiers), mises à jour entre le
22/06/2021 et le 09/10/2026.

Pour chaque fiche, on garde le résumé, le texte principal découpé en sections
(« ## Chapitre > Cas »), les formulaires et interlocuteurs cités, et les textes
de loi de référence. Les renvois vers d'autres pages, les URL et les documents
annexes sont retirés.

Sortie : data/raw/corpus.parquet, une ligne par fiche, avec les colonnes
doc_id, title, text, theme, fil_ariane, date, url et type.

Utilisation :
    uv run python prepare_docs.py          # génère le corpus
    uv run python prepare_docs.py --list   # nombre de fiches par thème
"""

import re  # expressions régulières : noms de fichiers et dates
import sys  # lecture de l'option --list
import xml.etree.ElementTree as ET  # lecture du XML (bibliothèque standard)
import zipfile  # lecture de l'archive sans la décompresser sur le disque
from collections import Counter  # comptage des fiches par thème
from pathlib import Path  # chemins de fichiers indépendants du système

import pandas as pd  # tableau final et écriture en Parquet

# ---------------------------------------------------------------------------
# Réglages : c'est la seule partie à modifier pour changer de corpus.
# ---------------------------------------------------------------------------
ZIP = Path("data/raw/vosdroits-latest.zip")
OUT = Path("data/raw/corpus.parquet")
# Thèmes de 1er niveau du fil d'Ariane, écrits exactement comme sur le site
# (la commande --list les affiche). Un ensemble {...} rend le test « in » rapide.
THEMES = {"Logement", "Papiers - Citoyenneté - Élections"}

# Balises dont le <Titre> ouvre une nouvelle section du texte.
SECTIONS = {"Chapitre", "SousChapitre", "Situation", "Cas"}
# Balises sautées avec tout leur contenu :
# - TitreRiche répète le <Titre> des cas (cause des titres en double) ;
# - les autres renvoient vers d'autres pages ou des documents annexes.
IGNORED = {"TitreRiche", "PourEnSavoirPlus", "VoirAussi", "QuestionReponse",
           "NoticeLiee", "ServiceEnLigneAnnexe"}


# ---------------------------------------------------------------------------
# Petites fonctions utilitaires
# ---------------------------------------------------------------------------
def tag(e):
    """Nom de la balise sans espace de noms : '{http://purl.org/dc/...}title' -> 'title'."""
    return e.tag.rsplit("}", 1)[-1]


def text(e):
    """Tout le texte d'une balise, balises imbriquées comprises, espaces normalisés.

    <Paragraphe>Des panneaux <LienIntra>photovoltaïques</LienIntra></Paragraphe>
    -> 'Des panneaux photovoltaïques'. Renvoie '' si la balise n'existe pas (None).
    """
    return " ".join("".join(e.itertext()).split()) if e is not None else ""


def child(e, name):
    """Premier enfant DIRECT de e nommé `name`, ou None s'il n'y en a pas.

    On ne cherche pas dans toute la descendance : pour un <Cas>, on veut son propre
    <Titre>, pas celui d'un cas imbriqué plus bas.
    """
    return next((c for c in e if tag(c) == name), None)


# ---------------------------------------------------------------------------
# Conversion du contenu XML en texte
# ---------------------------------------------------------------------------
def render(e, path, out):
    """Transforme la balise e (et ses enfants) en lignes de texte ajoutées à `out`.

    Fonction récursive : elle s'appelle elle-même sur les enfants des balises
    « conteneurs ». `path` contient les titres des sections parentes, pour écrire
    des titres complets comme « ## Cas général > Formulaire ».
    """
    t = tag(e)

    # Balise inutile : on l'abandonne avec tout son contenu.
    if t in IGNORED:
        return

    # Section : on écrit son titre, puis on traite son contenu.
    if t in SECTIONS:
        title = text(child(e, "Titre"))
        if title:
            # Nouvelle liste (et non path.append) : en sortant de cette section,
            # les sections sœurs retrouvent le chemin du parent, intact.
            path = path + [title]
            out.append("\n## " + " > ".join(path))  # ligne vide puis titre
        for c in e:
            if tag(c) != "Titre":  # le titre vient d'être écrit
                render(c, path, out)

    # Balises qui produisent du texte. text(e) récupère déjà le contenu des
    # balises imbriquées : on ne descend donc pas plus loin.
    elif t in ("Paragraphe", "Titre"):  # Titre hors section (ex. titre de tableau)
        out.append(text(e))
    elif t == "Item":  # élément de liste à puces
        out.append("- " + text(e))
    elif t.startswith("Rang"):  # ligne de tableau : cellules séparées par « | »
        out.append(" | ".join(text(c) for c in e))

    # Blocs techniques (URL, identifiants...) résumés en une ligne.
    elif t == "OuSAdresser":  # interlocuteur : « Où s'adresser : Mairie »
        out.append("Où s'adresser : " + text(child(e, "Titre")))
    elif t == "ServiceEnLigne":  # formulaire ou téléservice cité dans le texte
        cerfa = e.get("numerocerfa")  # e.get lit un ATTRIBUT de la balise
        out.append(f"{e.get('type', 'Service en ligne')} : {text(child(e, 'Titre'))}"
                   + (f" (cerfa n°{cerfa})" if cerfa else ""))

    # Toute autre balise (Texte, BlocCas, Liste, Tableau...) est un conteneur :
    # elle ne produit rien elle-même, on traite ses enfants.
    else:
        for c in e:
            render(c, path, out)


# ---------------------------------------------------------------------------
# Une fiche XML -> un dictionnaire (une future ligne du tableau)
# ---------------------------------------------------------------------------
def parse(xml):
    root = ET.fromstring(xml)  # racine <Publication>

    # Métadonnées Dublin Core (dc:title, dc:description, dc:date), enfants directs.
    meta = {tag(e): text(e) for e in root if tag(e) in ("title", "description", "date")}

    # Fil d'Ariane : « Accueil particuliers > Logement > Autorisations d'urbanisme > ... ».
    # [1:] retire « Accueil particuliers » : levels[0] est alors le thème.
    ariane = child(root, "FilDAriane")
    levels = [text(n) for n in ariane][1:] if ariane is not None else []

    # Texte : le résumé, puis les trois blocs de contenu principal. On ne convertit
    # pas toute la racine, qui contient aussi des métadonnées et des listes de liens.
    out = [meta.get("description", "")]
    for c in root:
        if tag(c) in ("Introduction", "Texte", "ListeSituations"):
            render(c, [], out)

    # Blocs de fin de fiche, hors du texte principal mais utiles pour répondre :
    # qui contacter, et sur quels textes de loi repose la règle.
    for name, label in (("QuiPeutMAider", "Qui peut m'aider ?"), ("Reference", "Textes de référence")):
        items = [c for c in root if tag(c) == name]
        if items:
            out.append(f"\n## {label}")
            for i in items:
                detail = text(child(i, "Complement"))
                out.append(f"- {text(child(i, 'Titre'))}" + (f" ({detail})" if detail else ""))

    # La date est écrite « modified 2025-02-07 » : on extrait la partie AAAA-MM-JJ.
    date = re.search(r"\d{4}-\d{2}-\d{2}", meta.get("date", ""))
    return {
        "doc_id": root.get("ID"),  # attribut de <Publication>, ex. « F36798 »
        "title": meta.get("title", ""),
        # Lignes recollées, sans les lignes vides (paragraphe vide, fiche sans résumé).
        "text": "\n".join(line for line in out if line.strip()).strip(),
        "theme": levels[0] if levels else "",
        "fil_ariane": " > ".join(levels),
        "date": date.group(0) if date else "",
        "url": root.get("spUrl", ""),  # adresse de la fiche sur service-public.gouv.fr
        "type": root.get("type", ""),  # fiche d'information, question-réponse...
    }


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------
def main():
    # Lecture de toutes les fiches : fichiers « F » + chiffres + « .xml », à la
    # racine de l'archive ou dans un dossier. Les pages N... et R... sont ignorées.
    with zipfile.ZipFile(ZIP) as z:
        fiches = [parse(z.read(n)) for n in z.namelist() if re.search(r"(^|/)F\d+\.xml$", n)]

    # Option --list : nombre de fiches par thème, du plus grand au plus petit.
    if "--list" in sys.argv:
        for theme, n in Counter(f["theme"] for f in fiches).most_common():
            print(f"{n:5d}  {theme}")  # :5d aligne les nombres sur 5 caractères
        return

    # Sélection : thèmes choisis, et au moins 200 caractères de texte (en dessous,
    # ce sont des pages quasi vides). Tri par identifiant : fichier identique à
    # chaque exécution.
    df = pd.DataFrame([f for f in fiches if f["theme"] in THEMES and len(f["text"]) >= 200])
    df = df.sort_values("doc_id")
    df.to_parquet(OUT, index=False)

    # Statistiques de contrôle, à reporter dans la data card.
    lengths = df["text"].str.len()
    print(f"{len(df)} fiches écrites dans {OUT}\n")
    print(df["theme"].value_counts().to_string(), "\n")
    print(f"Longueur : médiane {lengths.median():.0f} car., moyenne {lengths.mean():.0f}, max {lengths.max()}")
    # Les dates AAAA-MM-JJ se trient correctement comme du texte.
    print(f"Dates de mise à jour : {df['date'].min()} → {df['date'].max()}")


if __name__ == "__main__":
    main()