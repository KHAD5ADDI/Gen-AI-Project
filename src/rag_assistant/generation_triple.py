"""Pipeline triple confirmation : combine les 3 variantes pour une réponse plus fiable."""
from pydantic import BaseModel, Field
from .tracing import Trace
from .generation import generate, REFUS_TECHNIQUE
from .generation_confidence import generate_with_confidence
from .generation_crossval import generate_with_crossval


class ReponseTriple(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    answerable: bool = True
    confidence: int = Field(default=0, ge=0, le=100)
    confirmed_by: int = Field(default=0, description="Nombre de pipelines (sur 3) ayant répondu answerable=True")


def generate_triple(llm, question: str, chunks, trace: Trace) -> ReponseTriple:
    """Lance les 3 pipelines et prend une décision par vote majoritaire.

    Règle de vote :
    - answerable = True si au moins 2/3 des pipelines disent True
    - confidence : score issu du pipeline confiance (0 si non disponible)
    - réponse principale : crossval (vérifié) > base > confidence
    - citations : union de toutes les citations des pipelines answerable
    """
    r_base = generate(llm, question, chunks, trace)
    r_conf = generate_with_confidence(llm, question, chunks, trace)
    r_cross = generate_with_crossval(llm, question, chunks, trace)

    votes = sum([r_base.answerable, r_conf.answerable, r_cross.answerable])
    answerable = votes >= 2

    if not answerable:
        return ReponseTriple(
            answer=(
                f"Les 3 pipelines ont conclu qu'il n'y a pas de réponse fiable dans le corpus "
                f"({votes}/3 seulement ont trouvé une réponse)."
                if votes > 0
                else "Aucun des 3 pipelines n'a trouvé de réponse dans le corpus."
            ),
            citations=[],
            answerable=False,
            confidence=0,
            confirmed_by=votes,
        )

    # Score de confiance : issu du pipeline dédié
    confidence = r_conf.confidence if r_conf.answerable else 0

    # Réponse principale : ordre de préférence crossval > base > confidence
    primary = (
        r_cross if r_cross.answerable
        else r_base if r_base.answerable
        else r_conf
    )

    # Citations : union dédupliquée de tous les pipelines ayant répondu
    seen: set[str] = set()
    all_citations: list[str] = []
    for resp in (r_base, r_conf, r_cross):
        if resp.answerable:
            for c in resp.citations:
                if c not in seen:
                    seen.add(c)
                    all_citations.append(c)

    return ReponseTriple(
        answer=primary.answer,
        citations=all_citations,
        answerable=True,
        confidence=confidence,
        confirmed_by=votes,
    )
