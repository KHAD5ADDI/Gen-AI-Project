import xml.etree.ElementTree as ET
import re
from pathlib import Path

SIZE = 40
OVERLAP = 10

def parse_xml_document(file_path):
    """
    Extrait les métadonnées et le texte d'un fichier XML de Service-Public.
    À adapter selon les balises exactes de votre dataset.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extraction générique du titre (ex: <dc:title> ou <Titre>)
    title_elem = root.find('.//dc:title') or root.find('.//Titre')
    title = title_elem.text if title_elem is not None else Path(file_path).stem
    
    # Extraction générique de la date (ex: <dc:date> ou <Date>)
    date_elem = root.find('.//dc:date') or root.find('.//Date')
    doc_date = date_elem.text if date_elem is not None else "Date inconnue"
    
    # Extraction de tout le texte contenu dans les paragraphes
    text_elements = root.findall('.//Paragraphe')
    if not text_elements:
        text_blocks = [elem.text for elem in root.iter() if elem.text and elem.text.strip()]
    else:
        text_blocks = [elem.text for elem in text_elements if elem.text and elem.text.strip()]
        
    full_text = " ".join(text_blocks)
    
    return {
        "doc_id": Path(file_path).stem,
        "date": doc_date,
        "title": title,
        "text": full_text
    }

def split_words(n_words, size=SIZE, overlap=OVERLAP):
    """
    Retourne une liste de positions (start, end) pour les chunks de mots.
    Une fenêtre contient au maximum `size` mots, la suivante commence `size - overlap` mots plus tard[cite: 3].
    """
    step = size - overlap
    return [(i, min(i + size, n_words)) for i in range(0, max(1, n_words), step)]

def chunk_document(doc):
    """
    Découpe un document parsé en chunks. 
    Chaque chunk reçoit un en-tête contextuel combinant le titre et la date[cite: 3].
    """
    words = doc["text"].split()
    chunks = []
    
    for k, (start, end) in enumerate(split_words(len(words))):
        body = " ".join(words[start:end])
        header = f"{doc.get('title', 'Sans titre')} ({doc.get('date', '')})"
        
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#{k}",
            "doc_id": doc["doc_id"],
            "date": doc.get("date"),
            "title": doc.get("title"),
            "start": start,
            "end": end,
            "body": body,
            "text": f"{header}: {body}" # En-tête + corps du texte[cite: 3]
        })
    return chunks
