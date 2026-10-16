SIZE = 150
OVERLAP = 30

def split_words(n_words, size=SIZE, overlap=OVERLAP):
    """
    Retourne une liste de positions (start, end) pour les chunks de mots.
    """
    step = size - overlap
    return [(i, min(i + size, n_words)) for i in range(0, max(1, n_words), step)]

def chunk_document(doc_row):
    """
    Découpe une fiche issue du fichier Parquet en chunks.
    """
    text = doc_row.get("text", "")
    if not text:
        return []

    words = text.split()
    chunks = []

    # En-tête enrichi avec le thème pour donner un maximum de contexte administratif
    theme = doc_row.get("theme", "")
    title = doc_row.get("title", "Sans titre")
    date = doc_row.get("date", "Date inconnue")
    header = f"[{theme}] {title} (MàJ: {date})"

    for k, (start, end) in enumerate(split_words(len(words))):
        body = " ".join(words[start:end])

        chunks.append({
            "chunk_id": f"{doc_row['doc_id']}#{k}",
            "doc_id": doc_row["doc_id"],
            "date": date,
            "title": title,
            "theme": theme,
            "start": start,
            "end": end,
            "body": body,
            "text": f"{header}\n{body}" # Le texte final qui sera encodé
        })
    return chunks
