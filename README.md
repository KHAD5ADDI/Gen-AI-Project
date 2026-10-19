# Gen-AI-Project

Assistant RAG sur les fiches pratiques Service-Public.fr (logement).

## Installation

```bash
uv pip install -r requirements.txt
```

## Lancement

Dans un terminal séparé, démarrer Ollama :

```bash
ollama run llama3.2:3b
```

Ingestion du corpus :

```bash
uv run python run_ingest.py
```

Lancer l'application Streamlit :

```bash
uv run streamlit run app.py
```

Évaluation :

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

---

## Changements par fichier (branche khadidja)

### `src/rag_assistant/retrieval.py`
- Ajout des variables d'environnement Windows en début de fichier : `KMP_DUPLICATE_LIB_OK=TRUE` et `TOKENIZERS_PARALLELISM=false` pour éviter le conflit DLL OpenMP entre torch et ChromaDB.
- Import de `torch` avant toute opération ChromaDB (nécessaire sur Windows).
- Chargement du **CrossEncoder reranker** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) avec fallback BM25-only si indisponible.
- Pipeline `search()` : BM25 (30 candidats) → CrossEncoder reranker → top_k=6.
  - La recherche dense (`dense_search`) est conservée en méthode mais non appelée depuis `search()` : torch + ChromaDB dans le même processus provoque un crash au niveau OS sur Windows.
- Correction de `get_document()` : utilise `c["text"]` au lieu de `c["body"]`.
- `top_k` augmenté à 6 (au lieu de 4) pour inclure tous les passages pertinents après reranking.

### `src/rag_assistant/generation.py`
- Ajout de la déduplication des citations dans `parse_and_validate()` : supprime les `doc_id` en double avant de retourner la réponse.

### `src/rag_assistant/tracing.py`
- Réécriture complète du module de traçage.
- Classe `Trace` avec attributs : `tool_calls`, `errors`, `llm_calls`, `retrieved_passages`.
- Méthodes : `record_llm(purpose, duration, input_tokens, output_tokens, stop_reason)` et `record_passages(chunks)`.
- Propriétés : `total_tokens` et `total_llm_latency`.
- Fonction `chat(llm, messages, trace, purpose, tools)` qui enregistre la durée, les tokens et la raison d'arrêt depuis `response.raw`.

### `src/rag_assistant/generation_confidence.py` *(nouveau)*
- Variante du pipeline avec **score de confiance** (0–100) ajouté au schéma de réponse.
- Modèle `ReponseConfidence` : champ `confidence: int` en plus des champs standard.
- Correction : si `answerable=True` et `confidence=0`, la valeur par défaut passe à 50.
- Fonction `generate_with_confidence(llm, question, chunks, trace, max_attempts)`.

### `src/rag_assistant/generation_crossval.py` *(nouveau)*
- Variante du pipeline avec **cross-validation** : un second appel LLM vérifie la cohérence de la première réponse.
- Prompt vérificateur rendu permissif : rejette uniquement les hallucinations numériques claires, pas les résumés partiels.
- Fonction `generate_with_crossval(llm, question, chunks, trace, max_attempts)`.

### `app.py`
- Modèle changé de `llama3.1:8b` à `llama3.2:3b` pour la démo (temps de réponse réduit : ~1–2 min au lieu de 5–6 min).
- Ajout de `context_window=4096` pour éviter un OOM (le contexte par défaut 128K nécessite 16 Go de cache KV).
- Ajout de `request_timeout=300.0`.

### `requirements.txt`
- Ajout de `rank-bm25` (recherche lexicale BM25).
- Ajout de `sentence-transformers` (CrossEncoder reranker).

### `eval/run_eval.py`
- Ajout de `context_window=4096` pour éviter l'OOM avec `llama3.1:8b`.
- Ajout de `request_timeout=600.0`.
