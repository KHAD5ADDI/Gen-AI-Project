"""Vérifie que le corpus contient bien tout le texte des fiches d'origine.

    uv run python check_coverage.py

Pour chaque fiche du corpus, on relit son XML dans l'archive, on liste tous ses
paragraphes (<Paragraphe>) et on vérifie que chacun se retrouve, mot pour mot,
dans le texte nettoyé. Sont exclus du calcul les blocs que la préparation retire
ou résume volontairement (renvois, annexes, formulaires, adresses).
"""
import zipfile

import pandas as pd

try:  # le script de préparation peut s'appeler prepare_docs.py ou prepare_simple.py
    import prepare_docs as prep
except ImportError:
    import prepare_simple as prep

# Blocs dont les paragraphes ne doivent pas se retrouver tels quels dans le texte :
# ceux que la préparation ignore, et ceux qu'elle résume en une ligne.
NOT_EXPECTED = prep.IGNORED | {"OuSAdresser", "ServiceEnLigne"}


def paragraphs(e, found):
    """Ajoute à `found` le texte de chaque <Paragraphe> attendu dans le corpus."""
    if prep.tag(e) in NOT_EXPECTED:
        return
    if prep.tag(e) == "Paragraphe":
        if prep.text(e):
            found.append(prep.text(e))
        return
    for c in e:
        paragraphs(c, found)


def main():
    df = pd.read_parquet(prep.OUT)
    rows = []
    with zipfile.ZipFile(prep.ZIP) as z:
        names = {n.split("/")[-1][:-4]: n for n in z.namelist() if n.endswith(".xml")}
        for doc_id, text in zip(df["doc_id"], df["text"]):
            root = prep.ET.fromstring(z.read(names[doc_id]))
            expected = []
            for c in root:
                if prep.tag(c) in ("Introduction", "Texte", "ListeSituations"):
                    paragraphs(c, expected)
            missing = [p for p in expected if p not in text]
            rows.append({"doc_id": doc_id, "attendus": len(expected), "manquants": len(missing),
                         "exemple": missing[0][:120] if missing else ""})

    res = pd.DataFrame(rows)
    total, lost = res["attendus"].sum(), res["manquants"].sum()
    print(f"{len(res)} fiches vérifiées · {total} paragraphes attendus · "
          f"{total - lost} retrouvés ({100 * (total - lost) / max(total, 1):.2f} %)")

    incomplete = res[res["manquants"] > 0].sort_values("manquants", ascending=False)
    if incomplete.empty:
        print("Tous les paragraphes des fiches se retrouvent dans le corpus.")
        return
    print(f"\n{len(incomplete)} fiches avec des paragraphes manquants (les 15 premières) :")
    for _, r in incomplete.head(15).iterrows():
        print(f"  {r.doc_id:>7}  {r.manquants:3d}/{r.attendus:<4d} ex. « {r.exemple} »")


if __name__ == "__main__":
    main()
