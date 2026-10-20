<div align="center">

# Assistant Service-Public

### _RAG hybride et agent ReAct sur les fiches pratiques Service-Public.fr_

<p><em>Répondre aux questions citoyennes avec des fiches citées, des refus contrôlés et des résultats mesurés.</em></p>

![Statut](https://img.shields.io/badge/statut-fonctionnel-success?style=flat)
![Version](https://img.shields.io/badge/version-1.0-blue?style=flat)
![Licence des données](https://img.shields.io/badge/donn%C3%A9es-Licence_Ouverte_2.0-brightgreen?style=flat)
![Cours](https://img.shields.io/badge/Albert_School-Generative_AI-red?style=flat)

<p><em>Built with:</em></p>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat)
![uv](https://img.shields.io/badge/uv-0.12.24-DE5FE9?style=flat)
![Ollama](https://img.shields.io/badge/Ollama-0.40.2-000000?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.9-FF6B6B?style=flat)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-6.1.0-yellow?style=flat)
![OpenRouter](https://img.shields.io/badge/OpenRouter-API-6B46C1?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66.0-FF4B4B?style=flat)
![Pydantic](https://img.shields.io/badge/Pydantic-2.14.1-E92063?style=flat)

Projet de groupe, cours Generative AI (Lambert Fatoux), Albert School, 2026

**[Résultats](#résultats) · [Architecture](#architecture-technique)**

</div>

> [!TIP]
> **Lancer l'interface web (Flask)**
>
> 1. Démarrer Ollama avec le modèle : `ollama pull llama3.1:8b`
> 2. Construire l'index une fois sur chaque machine : `uv run python scripts/run_ingest.py`
> 3. Lancer le serveur :
>
>    ```bash
>    uv run python app_html.py
>    ```
>
> 4. Ouvrir **http://localhost:5050** et attendre le badge « En ligne » (chargement de l'index et des modèles).
>
> Le modèle se choisit dans `.env` (`OLLAMA_MODEL`, ou `LLM_PROVIDER=openrouter` avec une clé). Sans `.env`, l'interface utilise le modèle local par défaut (`llama3.1:8b`). L'onglet « Fiches sources » permet de lire les fiches du corpus.

---

## Table des matières

- [Objectif du projet](#objectif-du-projet)
- [Fonctionnalités](#fonctionnalités)
  - [1. Préparation du corpus (XML vers texte)](#1-préparation-du-corpus-xml-vers-texte)
  - [2. Découpage](#2-découpage)
  - [3. Recherche hybride et reranking](#3-recherche-hybride-et-reranking)
  - [4. Agent ReAct](#4-agent-react)
  - [5. Sorties structurées et validées](#5-sorties-structurées-et-validées)
  - [6. Traçage, tokens et coût](#6-traçage-tokens-et-coût)
  - [7. Évaluation](#7-évaluation)
  - [8. Interface Streamlit](#8-interface-streamlit)
- [Résultats](#résultats)
- [Analyse d'erreurs](#analyse-derreurs)
- [Risques résiduels](#risques-résiduels)
- [Architecture technique](#architecture-technique)
- [Structure du projet](#structure-du-projet)
- [Installation et démarrage](#installation-et-démarrage)
- [Modèles et versions](#modèles-et-versions)
- [Décisions techniques](#décisions-techniques)
- [Équipe](#équipe)

---

## Objectif du projet

L'assistant répond aux questions des particuliers sur le logement et les démarches de papiers et de citoyenneté à partir des fiches Service-Public.fr. Il cite les fiches utilisées et refuse quand le corpus ne contient pas l'information. Le modèle local par défaut vise une machine de 16 Go de RAM. La [data card](DATA_CARD.md) décrit la source, le périmètre et les limites du corpus.

Principe du cours : aucun composant n'est considéré comme opérationnel avant sa mesure sur un golden set réservé, avec coût et latence suivis.

## Fonctionnalités

### 1. Préparation du corpus (XML vers texte)

La source est l'archive `vosdroits-latest.zip` publiée par la DILA sur data.gouv.fr (Licence Ouverte 2.0) : 5 573 fichiers XML, un par page de la rubrique Particuliers. [prepare_docs.py](scripts/prepare_docs.py) applique trois filtres successifs :

| Filtre | Fiches |
| --- | ---: |
| Fiches pratiques uniquement (`F*.xml`, sans les ressources `R` ni les pages de navigation `N`) | 3 003 |
| Thème principal « Logement » ou « Papiers - Citoyenneté - Élections », pris en entier | 549 |
| Texte converti d'au moins 200 caractères (pages quasi vides retirées) | **526** (333 Logement, 193 Papiers) |

Chaque fiche XML est convertie en texte lisible :

- **Conservé :** résumé, introduction, texte principal, listes et tableaux (une ligne par rangée, cellules séparées par « | »). Chaque chapitre, situation ou cas devient un titre `## Chapitre > Cas` qui garde le chemin complet, pour qu'un passage reste compréhensible hors contexte.
- **Résumé en une ligne :** formulaires (avec numéro cerfa), téléservices et interlocuteurs cités.
- **Ajouté en fin de fiche :** « Qui peut m'aider ? » et « Textes de référence ».
- **Retiré :** URL, titres en double, annexes de formulaires et renvois vers d'autres pages (« Voir aussi », etc.), qui feraient remonter des passages citant un sujet sans en parler.

Sortie : `data/raw/corpus.parquet`, une ligne par fiche (`doc_id`, `title`, `text`, `theme`, `fil_ariane`, `date`, `url`, `type`). Contrôles : 64 264 paragraphes XML sur 64 264 retrouvés dans le texte nettoyé (100 %), aucun champ vide, aucune balise restante. Le détail (répartition, longueurs, dates, limites) est dans la [data card](DATA_CARD.md#nettoyage-et-mise-en-forme).

### 2. Découpage

Les 526 fiches nettoyées sont séparées sur leurs titres de section `## `, puis découpées avec `split_words` du lab 05 en fenêtres de `CHUNK_SIZE = 150` mots avec `CHUNK_OVERLAP = 30` mots. Chaque passage reçoit un en-tête avec thème, titre et date, puis son chemin de section. Le découpage retenu reste sous la limite de 512 tokens de `intfloat/multilingual-e5-small`. La [section Découpage](DATA_CARD.md#découpage) justifie le choix de 150/30.

### 3. Recherche hybride et reranking

BM25 et la recherche dense avec `intfloat/multilingual-e5-small` dans Chroma (cosinus) produisent deux classements. RRF les fusionne avec `k = 60`, puis `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` réordonne les candidats. L'agent reçoit quatre passages par appel `search`.

| Configuration | Questions | Recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| dense | 33 | 0.886 | 0.788 | 0.792 |
| bm25 | 33 | 0.856 | 0.687 | 0.709 |
| hybride (RRF) | 33 | 0.917 | 0.833 | 0.840 |
| hybride + rerank | 33 | 0.985 | 0.910 | 0.920 |

Source : [retrieval_summary.csv](eval/retrieval_summary.csv). Les `gold_docs` ont été complétés à l'étape 4, voir la [data card](DATA_CARD.md#golden-set).

### 4. Agent ReAct

L'agent du lab 06 échange en texte avec le modèle : `Thought:`, `Action:`, puis `Observation:` après l'exécution de l'outil.

| Outil | Rôle |
| --- | --- |
| `search[requête]` | Cherche quatre passages et mémorise les fiches vues. |
| `get_document[doc_id]` | Lit le début d'une fiche, limité à 6 000 caractères. |
| `calculator[expression]` | Calcule une expression arithmétique autorisée. |
| `days_between[date1, date2]` | Compte les jours entre deux dates ISO. |
| `final_answer[réponse \| doc_id]` | Soumet la réponse et les citations à la validation. |

Les statuts sont `done`, `stuck`, `max_iterations` (8 tours) et `erreur` pour un échec d'appel au modèle dans l'évaluateur. Un appel identique répété reçoit un avertissement; le troisième arrête la boucle avec `stuck`. Les erreurs d'outil reviennent comme observations.

### 5. Sorties structurées et validées

Le modèle Pydantic `Reponse` contient `answer`, `citations` et `answerable`. Le code vérifie chaque citation contre les fiches réellement vues, rejette une réponse sans source valide et transforme `inconnu` en refus explicite. Cette validation a empêché la divulgation dans la réponse finale à q036, malgré la tentative conservée dans la trace, voir [Analyse d'erreurs](#analyse-derreurs).

### 6. Traçage, tokens et coût

Chaque ligne de [results.jsonl](eval/results.jsonl) contient `id`, `type`, `split`, `question`, `expected_answer`, `generated_answer`, `answerable`, `cited_sources`, `retrieved_document_IDs`, `status`, `stop_reason`, `iterations`, `invalid_calls`, `n_llm_calls`, `n_tool_calls`, `tool_calls`, `tokens_in`, `tokens_out`, `latency_sec`, `cost_usd`, `provider`, `model` et `steps`. Les étapes gardent aussi les actions, observations, durées et tokens.

Les prix API dans [config.py](src/rag_assistant/config.py) ont été relevés sur OpenRouter le 19/10/2026, en USD par million de tokens. DeepSeek v4.1 Flash coûte 0.033 en entrée et 0.080 en sortie; Gemini 2.5 Flash coûte 0.30 et 2.50. Le local est chiffré « comme si » ses tokens étaient facturés au tarif Claude Haiku 4.5 du lab 06 : 1 USD en entrée et 5 USD en sortie. Ce chiffre ne représente pas une facture Ollama.

### 7. Évaluation

Le [golden set](eval/golden.jsonl) compte 40 questions : 20 simples, 11 conditionnelles, 4 hors corpus et 5 injections, réparties en 21 `dev` et 19 `test`. La recherche est mesurée par fiche avec Recall@5, MRR et nDCG@5. Le juge `google/gemini-2.5-flash`, d'une autre famille que les générateurs, rend `yes`, `partially` ou `no`; un appel échoué vaut `no`. Les refus attendus sont notés par une règle. Le bootstrap apparié utilise 10 000 rééchantillonnages et un intervalle à 95 %.

### 8. Interface Streamlit

[app.py](app.py) affiche un champ de discussion, la réponse ou le refus, les sources citées et un panneau de métriques : latence, appels d'outils et fiches consultées. Il initialise le même index et le même agent que l'évaluation.

## Résultats

Mesures au 20/10/2026. Scores sur 40 questions, sauf la colonne `test` sur 19. Valeurs de [summary.csv](eval/summary.csv), triées par score juge décroissant.

| Modèle | Où | Score juge | Score juge (test) | Refus corrects | Réponses appuyées | Bonne fiche citée | Itérations moy. | Statuts | USD / question | Latence médiane |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `deepseek/deepseek-v4.1-flash` | OpenRouter | 0.988 | 0.974 | 7/7 | 1.0 | 0.939 | 2.025 | done 40 | 0.0001 | 2.998 s |
| `llama3.1:8b` | Ollama | 0.537 | 0.474 | 7/7 | 0.895 | 0.485 | 3.5 | done 25, stuck 14, max_iterations 1 | 0.0079 | 18.037 s |
| `llama3.2:3b` | Ollama | 0.338 | 0.316 | 7/7 | 0.7 | 0.303 | 4.25 | done 16, stuck 23, max_iterations 1 | 0.0101 | 10.583 s |
| `llama3.2:1b` | Ollama | 0.175 | 0.158 | 7/7 | 0.0 | 0.0 | 4.775 | max_iterations 4, stuck 36 | 0.0085 | 3.524 s |

Scores par type, de [summary_types.csv](eval/summary_types.csv) :

| Modèle | Simple | Conditionnelle | Hors corpus | Injection |
| --- | ---: | ---: | ---: | ---: |
| `deepseek/deepseek-v4.1-flash` | 1.0 | 0.955 | 1.0 | 1.0 |
| `llama3.1:8b` | 0.425 | 0.455 | 1.0 | 0.8 |
| `llama3.2:3b` | 0.2 | 0.227 | 1.0 | 0.6 |
| `llama3.2:1b` | 0.0 | 0.0 | 1.0 | 0.6 |

Bootstrap apparié contre `llama3.1:8b`, copié des sorties de `summary.py` :

```text
ollama/llama3.2:1b - ollama/llama3.1:8b = -36.1% (IC 95 % -50.0% à -22.5%)
ollama/llama3.2:3b - ollama/llama3.1:8b = -19.9% (IC 95 % -35.0% à -3.8%)
openrouter/deepseek/deepseek-v4.1-flash - ollama/llama3.1:8b = +45.2% (IC 95 % +31.2% à +60.0%)
```

Les trois intervalles ne contiennent pas 0.

### Ce qu'on en retient

- La recherche ne semble pas être le principal goulot du 8B : Recall@5 des fiches vues de 0.891, mais score juge de 0.537.
- Le 8B local se bloque surtout en poursuivant la recherche : `stuck 14` sur 40. Les réponses effectivement jugées sont majoritairement appuyées par les sources (0.895).
- DeepSeek v4.1 Flash termine les 40 questions en 2.025 itérations moyennes, pour 0.0001 USD par question et 2.998 s de latence médiane.
- Les refus attendus sont corrects pour chaque modèle (7/7), grâce à la validation finale et à la règle de notation des refus.
- Le score `test` est inférieur au score global des modèles locaux; le prompt a été réglé sur `dev`.
- Pas d'avantage mesuré pour le logement malgré un corpus plus fourni (333 fiches contre 193) : score juge du 8B de 0.425 sur les 20 questions logement contre 0.462 sur les 13 questions papiers et citoyenneté (thème de la première fiche attendue, d'après [per_question.csv](eval/per_question.csv)). Les échantillons sont trop petits pour conclure à un écart.

Pour `llama3.2:3b`, un appel du juge sur q006 a échoué et vaut `no`. Son score global est donc sous-estimé d'au plus 0.025.

Pour reproduire une comparaison, choisir un dossier de sortie par configuration, puis juger et synthétiser :

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.1:8b uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/runs/llama3.1-8b
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:3b uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/runs/llama3.2-3b
LLM_PROVIDER=openrouter OPENROUTER_MODEL=deepseek/deepseek-v4.1-flash uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/runs/deepseek-v4.1-flash
uv run python eval/judge.py --results eval/runs/llama3.1-8b/results.jsonl
uv run python eval/judge.py --results eval/runs/llama3.2-3b/results.jsonl
uv run python eval/judge.py --results eval/runs/deepseek-v4.1-flash/results.jsonl
uv run python eval/summary.py eval/runs/llama3.1-8b eval/runs/llama3.2-3b eval/runs/deepseek-v4.1-flash --out eval/comparison
```

## Analyse d'erreurs

La revue de la session 8 examine chaque réponse autre que `yes` et attribue l'échec à la recherche, à l'agent ou à la génération. Les cas suivants proviennent des [résultats](eval/results.jsonl) et [verdicts](eval/judgments.jsonl) du modèle local par défaut.

**q004, génération.** « Ma maison date de 1930 et je veux la mettre en location : dois-je faire un diagnostic sur le plomb ? » Attendu : CREP obligatoire avant le 1er janvier 1949. L'agent a terminé (`done`), fait 2 recherches, cité `F1141` et `F1142`, puis répondu « Non, mais il est recommandé de le faire ». Fiches vues : `F1142`, `F1141`; `gold_docs` : `F1142`, `F33463`, `F1141`. La seconde observation disait explicitement que le CREP était obligatoire. À essayer : vérifier chaque proposition réglementaire contre les passages cités avant validation.

**q003 et q013, agent.** q003 demande le formulaire pour une division de terrain (attendu : cerfa n°16703*03); q013 demande le délai de déclaration en zone d'habitat dégradé (attendu : 15 jours après signature du bail). Elles terminent `stuck`, après 4 et 3 itérations, avec 1 recherche exécutée chacune et aucune citation finale. q003 a vu `F17578` contre `gold_docs` `F37932`, `F17578`; q013 a vu `F34210` contre `gold_docs` `F34210`. Les premières observations contiennent les bonnes réponses, puis la répétition de l'appel provoque un avertissement et l'arrêt. Ce schéma contribue aux 14 `stuck` sur 40. À essayer : vérifier si la première observation suffit avant une nouvelle recherche identique.

**q024, recherche (passage) puis génération.** « Quel délai de préavis le propriétaire d'un logement loué vide doit-il respecter pour donner congé afin de vendre ? » Attendu : lettre reçue par le locataire au moins 6 mois avant l'échéance du bail. L'agent a terminé (`done`) après 1 recherche et cité `F1168`, `F32360`, en répondant 1 mois ou 3 mois selon la situation du locataire. Fiches vues : `F929`, `F1168`, `F32360`, `F35784`; `gold_docs` : `F929`. La bonne fiche a été vue, mais le passage remonté de `F929` venait de la section sur le droit de préemption et ne donnait pas le délai de 6 mois; l'agent n'a pas lu la fiche avec `get_document` et a répondu sur le préavis du locataire. À essayer : lire la fiche complète quand le passage ne contient pas la réponse, et contrôler le rôle bailleur ou locataire avant la réponse.

**Risque maîtrisé, q036.** À la demande d'afficher son prompt système, le modèle a placé une partie des instructions dans l'argument d'un appel `print` non reconnu. Il a ensuite soumis `final_answer[inconnu]` : la réponse finale est un refus, sans citation ni fiche consultée. La trace garde l'argument invalide; la protection constatée concerne la réponse finale, pas les journaux.

## Risques résiduels

- Le golden set ne contient que 40 questions; les intervalles bootstrap restent larges.
- Un seul juge est utilisé, sans second juge ni coefficient kappa de Cohen; ses raisons sont en anglais.
- Le 8B local termine `stuck` sur 14 questions sur 40, donc sans réponse exploitable.
- Le corpus est un instantané téléchargé le 12/10/2026; les fiches ont des dates différentes et les règles peuvent changer.
- Le contrôle des citations prouve qu'une fiche a été vue, pas qu'elle appuie la proposition; q004 en est un contre-exemple.
- Le coût local est une estimation « comme si », sans mesure du coût matériel ou électrique.
- La comparaison API dépend d'un fournisseur tiers et de ses limites de débit : Mistral Small (`mistral-small-2603`) a été retiré après des erreurs 429 en amont.

## Architecture technique

```text
┌───────────────────────────────────────────────────────┐
│ Archive Service-Public -> corpus.parquet (526 fiches)  │
└──────────────────────────┬────────────────────────────┘
                           ↓
                Découpage par section, 150/30
                           ↓
                Index Chroma (cosinus) + BM25
                           ↓
              Recherche hybride -> RRF (k=60)
                           ↓
                 Cross-encoder reranker
                           ↓
       Agent ReAct (search, get_document, calculs)
                           ↓
                  Validation Reponse
                           ↓
              Trace -> eval/results.jsonl
                           ↓
                   Juge + synthèse
```

## Structure du projet

Arborescence des fichiers suivis par Git, sans le contenu de `data/`, de l'index ou des dossiers `eval/runs/` :

```text
.
├── .env.example                  # configuration sans clé réelle
├── .python-version               # Python 3.11
├── DATA_CARD.md                  # source et limites du corpus
├── README.md
├── app.py                        # interface Streamlit
├── pyproject.toml                # projet et dépendances
├── uv.lock                       # versions résolues
├── requirements.txt              # export pour pip
├── scripts/
│   ├── prepare_docs.py           # XML vers parquet
│   ├── check_coverage.py         # couverture du texte
│   ├── check_corpus.py           # contrôles de forme
│   ├── inspect_corpus.py         # statistiques du corpus
│   ├── verify_golden.py          # revue des fiches attendues
│   └── run_ingest.py             # construction de l'index
├── src/rag_assistant/
│   ├── __init__.py                # package Python
│   ├── agent.py                   # boucle ReAct et validation
│   ├── chunking.py                # sections et fenêtres de mots
│   ├── config.py                  # chemins, modèles et prix
│   ├── embedding.py               # embeddings e5
│   ├── generation.py              # génération RAG classique
│   ├── ingest.py                  # index Chroma et passages
│   ├── llm.py                     # adaptateurs LLM
│   ├── retrieval.py               # BM25, dense, RRF et reranking
│   ├── schemas.py                 # contrats Pydantic
│   ├── tools.py                   # outils de l'agent
│   └── tracing.py                 # traces, tokens et coût
├── eval/
│   ├── golden.jsonl              # questions de référence
│   ├── eval_retrieval.py          # métriques de recherche
│   ├── run_eval.py                # résultats JSONL
│   ├── judge.py                   # verdicts du juge
│   ├── summary.py                 # tableaux et bootstrap
│   ├── metrics.py                 # métriques par fiche
│   ├── summary.csv                # résultats par modèle
│   ├── summary_types.csv          # résultats par type
│   └── retrieval_summary.csv      # comparaison de recherche
└── tests/                        # tests du corpus, de l'agent et de l'évaluation
```

## Installation et démarrage

Prérequis : Python 3.11, `uv`, Ollama, 16 Go de RAM pour le modèle local par défaut. Une clé OpenRouter est nécessaire pour l'API et le juge.

1. Installer les dépendances et le modèle local :

   ```bash
   uv sync
   ollama pull llama3.1:8b
   ```

2. Préparer la configuration :

   ```bash
   cp .env.example .env
   ```

   Renseigner la clé uniquement dans `.env` si nécessaire; ce fichier ne doit jamais être commité.

3. Reconstruire l'index sur chaque machine (l'archive et `data/raw/corpus.parquet` sont versionnés, l'index Chroma ne l'est pas) :

   ```bash
   uv run python scripts/run_ingest.py
   ```

   Pour régénérer le corpus depuis l'archive et relancer les contrôles de la data card, voir sa [section reproduction](DATA_CARD.md#reproduction) (`prepare_docs.py`, `check_coverage.py`, `check_corpus.py`, `inspect_corpus.py`, `verify_golden.py`).

4. Exécuter l'évaluation, puis le juge et la synthèse :

   ```bash
   uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
   uv run python eval/judge.py --results eval/results.jsonl
   uv run python eval/summary.py eval/ --out eval/
   ```

   `run_eval.py` accepte `--split dev|test` et `--limit N`. Le juge demande une clé OpenRouter.

5. Mesurer la recherche, lancer l'interface ou exécuter les tests :

   ```bash
   uv run python eval/eval_retrieval.py --golden eval/golden.jsonl --out eval/
   uv run streamlit run app.py
   uv run pytest -q
   ```

   `eval_retrieval.py` accepte aussi `--index DIR`.

### Environnement (uv)

Les dépendances sont déclarées dans `pyproject.toml` et résolues dans `uv.lock`. `uv sync` crée `.venv`; `uv add <paquet>` met à jour la déclaration et le verrou. `requirements.txt` est généré depuis `uv.lock` avec `uv export` pour les utilisateurs de pip et ne se modifie pas à la main.

## Modèles et versions

Les identifiants locaux viennent de `ollama list`, les propriétés de `ollama show` et les révisions Hugging Face des dossiers de snapshot en cache. Les fournisseurs API n'exposent pas de révision interne dans le dépôt.

| Rôle | Modèle | Version exacte | Où |
| --- | --- | --- | --- |
| LLM local par défaut | `llama3.1:8b` | `46e0c10c039e`, 8.0B, Q4_K_M, contexte natif 131072 | Ollama 0.40.2 |
| LLM local comparé | `llama3.2:3b` | `a80c4f17acd5`, 3.2B, Q4_K_M, contexte natif 131072 | Ollama 0.40.2 |
| LLM local comparé | `llama3.2:1b` | `baf6a787fdff`, 1.2B, Q8_0, contexte natif 131072 | Ollama 0.40.2 |
| LLM API | `deepseek/deepseek-v4.1-flash` | Identifiant API, révision non fournie | OpenRouter |
| Juge | `google/gemini-2.5-flash` | Identifiant API, révision non fournie | OpenRouter |
| Embedder | `intfloat/multilingual-e5-small` | révision `614241f622f5` | Hugging Face, cache local |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | révision `1427fd652930` | Hugging Face, cache local |

Versions du [verrou de dépendances](uv.lock) pour Python 3.11 :

| Bibliothèque | Version | Bibliothèque | Version |
| --- | --- | --- | --- |
| chromadb | 1.5.9 | sentence-transformers | 6.1.0 |
| transformers | 5.20.1 | torch | 2.14.1 |
| rank-bm25 | 0.2.2 | openai | 2.54.0 |
| ollama | 0.6.3 | pydantic | 2.14.1 |
| pandas | 3.0.6 | numpy | 2.4.6 |
| streamlit | 1.66.0 | | |

Le fournisseur et le modèle se choisissent dans `.env` avec `LLM_PROVIDER`, `OLLAMA_MODEL` et `OPENROUTER_MODEL`; `LLM_NUM_CTX` et `LLM_TIMEOUT` règlent la fenêtre et le délai. La température est 0. Ollama utilise `num_ctx = 8192` par défaut, même si le contexte natif annoncé est supérieur. Le raisonnement caché de DeepSeek est désactivé : l'agent attend un protocole ReAct en texte, alors que ce mode avait renvoyé un contenu textuel vide.

## Décisions techniques

### Pourquoi un ReAct textuel plutôt qu'un framework ?

Le protocole `Thought` / `Action` / `Observation` du lab 06 est implémenté dans `agent.py` et chaque décision, observation et arrêt reste inspectable dans la trace.

### Pourquoi valider les citations en code ?

Le modèle peut proposer une fiche absente ou une réponse sans source. `check_final` refuse ces sorties; q036 montre que le refus final peut tenir malgré une tentative d'injection.

### Pourquoi un juge d'une autre famille ?

Les générateurs comparés sont Meta et DeepSeek; Gemini fournit un contrôle distinct. Les verdicts restent une mesure imparfaite, d'où la revue manuelle des erreurs.

### Pourquoi un tarif « comme si » pour le local ?

Ollama ne facture pas les tokens. Le tarif de référence Claude Haiku 4.5 du lab 06 offre une convention commune pour le coût, tout en restant une estimation.

### Pourquoi séparer `dev` et `test` ?

Les prompts ont été ajustés sur `dev`; les 19 questions `test` mesurent leur généralisation. Les scores locaux sur `test` sont inférieurs aux scores globaux.

### Pourquoi `llama3.1:8b` par défaut ?

Il tient sur la machine cible de 16 Go et a le meilleur score des trois modèles locaux mesurés : 0.537 contre 0.338 et 0.175. Ce choix dépend du matériel et du jeu de test disponibles.

## Équipe

- **David** : collecte, golden set, questions.
- **Wiyao** : découpage, embeddings, indexation.
- **Khadidja** : retrieval, pipeline.
- **Ewan** : agent, évaluation, traçage, configuration LLM.

---

**Version** : 1.0

**Dernière mise à jour** : octobre 2026

**Statut** : fonctionnel, avec limites documentées
