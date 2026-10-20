"""Contrats entre les modules : les formats que chaque étape donne à la suivante.

Chaque module peut être réécrit librement tant qu'il respecte ces formats.
"""
from collections import namedtuple
from typing import TypedDict

from pydantic import BaseModel, Field


class Passage(TypedDict):
    """Un passage renvoyé par la recherche (retrieval.search)."""

    chunk_id: str  # « F1234#3 » : fiche F1234, 4e passage
    doc_id: str  # identifiant de la fiche, celui qu'on cite
    title: str
    date: str  # AAAA-MM-JJ, dernière mise à jour de la fiche
    text: str  # texte du passage, tel que l'a vu l'embedder


# Coût d'un appel au LLM, comme dans les labs 04 et 06.
Usage = namedtuple("Usage", ["input_tokens", "output_tokens", "seconds"])


class Reponse(BaseModel):
    """Réponse finale de l'assistant : sortie structurée et validée (lab 04)."""

    answer: str
    citations: list[str] = Field(default_factory=list)  # doc_id des fiches qui justifient la réponse
    answerable: bool = True  # False : l'assistant refuse (information absente ou demande hors sujet)
