# RAG assistant

Assistant de questions-réponses sur un corpus de documents publics de data.gouv.fr.

## Installation

    uv sync
    cp .env.example .env

Compléter ensuite le fichier `.env` (modèles, clés API).

## Évaluation

    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

## Modèles

| Rôle | Modèle | Version |
|---|---|---|
| Embedding | | |
| LLM | | |
