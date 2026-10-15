
"""Golden set : eval/golden.jsonl (40 questions, format du sujet : id, question, answer, gold_docs)

Questions     Type                   Nombre  Ce qui est testé
------------  ---------------------  ------  ----------------------------------------------------
q001 à q020   Simples                20      Un fait précis dans une seule fiche : 2 reprennent mot
                                             pour mot le titre de la fiche, 9 sont proches du
                                             texte, 9 sont formulées comme le ferait un usager
q021 à q031   Conditionnelles        11      La réponse dépend d'une précision donnée dans la
                                             question
q032 à q035   Hors corpus            4       Sujets hors du domaine, que le modèle connaît par
                                             son entraînement
q036 à q040   Injections de prompt   5       Tentatives de détourner les règles de l'assistant

Détail des questions et des pièges testés : eval/golden_notes.md
"""
