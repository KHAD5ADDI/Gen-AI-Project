"""Découpe les fiches en passages : d'abord par section, puis en fenêtres de mots comme dans le lab 05."""

from .config import CHUNK_OVERLAP, CHUNK_SIZE


def split_words(n_words, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Fonction du lab 05 : arrête les fenêtres dès que l'une atteint le dernier mot."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Il faut 0 <= overlap < size.")

    windows = []
    step = size - overlap
    for start in range(0, n_words, step):
        end = min(start + size, n_words)
        windows.append((start, end))
        if end == n_words:
            break
    return windows


def sections(text):
    """Sépare une fiche sur ses titres de section."""
    result = []
    title = ""
    words = []
    for line in text.splitlines():
        if line.startswith("## "):
            if words:
                result.append((title, words))
            title = line[3:].strip()
            words = []
        else:
            words.extend(line.split())
    if words:
        result.append((title, words))
    return result


def chunk_document(doc_row, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Découpe chaque section en passages avec le contexte de la fiche."""
    theme = doc_row.get("theme", "")
    title = doc_row.get("title", "Sans titre")
    date = doc_row.get("date", "Date inconnue")
    header = f"[{theme}] {title} (MàJ: {date})"
    chunks = []

    for section, words in sections(doc_row.get("text", "")):
        for start, end in split_words(len(words), size=size, overlap=overlap):
            body = " ".join(words[start:end])
            section_header = "## " + section + "\n" if section else ""
            chunks.append({
                "chunk_id": f"{doc_row['doc_id']}#{len(chunks)}",
                "doc_id": doc_row["doc_id"],
                "date": date,
                "title": title,
                "theme": theme,
                "url": doc_row.get("url", ""),
                "type": doc_row.get("type", ""),
                "fil_ariane": doc_row.get("fil_ariane", ""),
                "section": section,
                "start": start,
                "end": end,
                "body": body,
                "text": header + "\n" + section_header + body,
            })
    return chunks
