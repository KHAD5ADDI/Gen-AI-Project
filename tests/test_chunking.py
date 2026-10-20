import chromadb
import pandas as pd
import pytest

from rag_assistant.chunking import chunk_document, sections, split_words
from rag_assistant.config import CHUNK_SIZE, COLLECTION, CORPUS_PATH
from rag_assistant.ingest import IngestionPipeline


def test_split_words_comme_dans_le_lab():
    assert split_words(10, size=4, overlap=1) == [(0, 4), (3, 7), (6, 10)]
    assert split_words(5, size=10, overlap=2) == [(0, 5)]
    assert split_words(100, size=40, overlap=10)[1] == (30, 70)


def test_sections():
    text = """Introduction courte.
Encore une phrase.
## Droits et démarches
Premier paragraphe.
## Section vide
## Cas particuliers
Dernier paragraphe.
"""
    assert sections(text) == [
        ("", ["Introduction", "courte.", "Encore", "une", "phrase."]),
        ("Droits et démarches", ["Premier", "paragraphe."]),
        ("Cas particuliers", ["Dernier", "paragraphe."]),
    ]


def test_fenetres_couvrent_tout_le_texte():
    for n in range(1, 400):
        windows = split_words(n)
        assert windows[0][0] == 0
        assert windows[-1][1] == n
        for k in range(1, len(windows)):
            previous = windows[k - 1]
            current = windows[k]
            assert current[0] <= previous[1]
            assert not (previous[0] <= current[0] and current[1] <= previous[1])


def test_chaque_section_est_decoupee_comme_dans_le_lab():
    corpus = pd.read_parquet(CORPUS_PATH)
    for row in corpus.to_dict(orient="records"):
        chunks = iter(chunk_document(row))
        for section, words in sections(row["text"]):
            for start, end in split_words(len(words)):
                chunk = next(chunks)
                assert (chunk["section"], chunk["start"], chunk["end"]) == (section, start, end)
                assert chunk["body"] == " ".join(words[start:end])
        assert next(chunks, None) is None


def test_identifiants_taille_et_titre_de_section():
    corpus = pd.read_parquet(CORPUS_PATH)
    for row in corpus.to_dict(orient="records"):
        chunks = chunk_document(row)
        assert [c["chunk_id"] for c in chunks] == [f"{row['doc_id']}#{k}" for k in range(len(chunks))]
        for chunk in chunks:
            assert len(chunk["body"].split()) <= CHUNK_SIZE
            if chunk["section"]:
                assert "## " + chunk["section"] in chunk["text"]


def test_ancien_index_refuse(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(name=COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    collection.add(
        ids=["F1#0"],
        embeddings=[[1.0, 0.0]],
        documents=["Ancien passage"],
        metadatas=[{"doc_id": "F1", "date": "2026-01-01", "title": "Ancienne fiche", "theme": "Logement"}],
    )
    with pytest.raises(RuntimeError, match="uv run python scripts/run_ingest.py"):
        IngestionPipeline(persist_directory=tmp_path)


def test_reconstruction_sur_un_ancien_index(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(name=COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    collection.add(
        ids=["F1#0"],
        embeddings=[[1.0, 0.0]],
        documents=["Ancien passage"],
        metadatas=[{"doc_id": "F1", "date": "2026-01-01", "title": "Ancienne fiche", "theme": "Logement"}],
    )
    tiny_parquet = tmp_path / "deux_fiches.parquet"
    pd.read_parquet(CORPUS_PATH).head(2).to_parquet(tiny_parquet, index=False)

    pipeline = IngestionPipeline(persist_directory=tmp_path, load_docstore=False)
    assert pipeline.docstore == {}
    pipeline.ingest_parquet(parquet_path=tiny_parquet)

    reopened = IngestionPipeline(persist_directory=tmp_path)
    assert reopened.docstore
    assert "body" in next(iter(reopened.docstore.values()))
    metadata = reopened.collection.get(limit=1, include=["metadatas"])["metadatas"][0]
    assert "body" in metadata and "embed_model" in metadata
