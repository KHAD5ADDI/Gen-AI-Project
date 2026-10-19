# Gen-AI-Project

Ouvrir le terminal à la racine du projet Git et installer les paquets requis : uv pip install -r requirements.txt

Dans un terminal séparé, assurez-vous qu'Ollama tourne en arrière-plan avec le modèle requis : ollama run llama3.1:8b

uv run python run_ingest.py

uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

uv run streamlit run app.py

enjoy !

## Modèles et coût

Le LLM se choisit dans le fichier `.env` (copier `.env.example`), sans toucher au code : `LLM_PROVIDER=ollama` (défaut, local) ou `LLM_PROVIDER=openrouter` (API, clé `OPENROUTER_API_KEY`). Température 0 dans tous les cas.

| Rôle | Modèle | Où |
|---|---|---|
| LLM local (défaut) | `llama3.1:8b` (Q4_K_M), fenêtre de contexte 8 192 tokens | Ollama |
| LLM API (comparaison) | `mistralai/mistral-small-2603` | OpenRouter |

Coût : chaque réponse enregistre ses tokens d'entrée et de sortie. Pour les modèles API, on applique le prix réel (relevé sur OpenRouter le 19/10/2026, voir `src/rag_assistant/config.py`). Un modèle local est gratuit : comme dans le lab 06, on le compte « comme si » les mêmes tokens étaient payés au tarif de Claude Haiku 4.5 (1 USD par million de tokens en entrée, 5 USD en sortie), pour comparer les configurations entre elles.
