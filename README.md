# Gen-AI-Project
Ouvrir le terminal à la racine du projet Git et installer les paquets requis : uv pip install -r requirements.txt
Dans un terminal séparé, assurez-vous qu'Ollama tourne en arrière-plan avec le modèle requis : ollama run llama3.1:8b
uv run python run_ingest.py
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
uv run streamlit run app.py
enjoy !
