SIZE = 40
OVERLAP = 10

def split_words(n_words, size=SIZE, overlap=OVERLAP):
    """
    Retourne une liste de positions (start, end) pour les chunks de mots.
    """
    step = size - overlap
    return [(i, min(i + size, n_words)) for i in range(0, max(1, n_words), step)]

def chunk_document(doc_row):
    """
    Découpe une fiche issue du fichier Parquet en chunks. 
    Chaque chunk reçoit un en-tête contextuel combinant le titre et la date.
    """
    text = doc_row.get("text", "")
    if not text:
        return []

    words = text.split()
    chunks = []
    
    for k, (start, end) in enumerate(split_words(len(words))):
        body = " ".join(words[start:end])
        header = f"{doc_row.get('title', 'Sans titre')} ({doc_row.get('date', 'Date inconnue')})"
        
        chunks.append({
            "chunk_id": f"{doc_row['doc_id']}#{k}",
            "doc_id": doc_row["doc_id"],
            "date": doc_row.get("date"),
            "title": doc_row.get("title"),
            "start": start,
            "end": end,
            "body": body,
            "text": f"{header}: {body}" # En-tête + corps du texte
        })
    return chunks
