# Data card — Corpus Service-Public « Logement » et « Papiers - Citoyenneté - Élections »

## Source

| | |
|---|---|
| Jeu de données | Fiches pratiques Service-Public.fr — Particuliers (« vosdroits ») |
| Producteur | Direction de l'information légale et administrative (DILA) |
| Accès | data.gouv.fr, archive `vosdroits-latest.zip` |
| Date de téléchargement | 12/10/2026 |
| Format | Un fichier XML par page du site (schéma Publication 3.5) |
| Licence | *à vérifier sur la page data.gouv.fr (Licence Ouverte / Etalab attendue)* |

L'archive contient **5 573 fichiers** :

| Type de fichier | Nombre | Utilisation |
|---|---|---|
| Fiches pratiques (`F*.xml`) | 3 003 | Contenu du corpus |
| Ressources (`R*.xml`) : formulaires, téléservices, adresses, définitions | 2 318 | Non lues directement : leurs informations utiles (titre, numéro cerfa, interlocuteur) figurent déjà dans les fiches qui les citent |
| Pages de navigation (`N*.xml`) : thèmes, sous-thèmes, dossiers | 243 | Non utilisées |
| Autres fichiers techniques | 9 | Non utilisés |

## Cas d'usage et périmètre

**Assistant visé :** un assistant citoyen qui répond aux questions d'un particulier sur son **logement** (location, achat, copropriété, travaux, urbanisme, aides) et sur ses **démarches de citoyenneté** (papiers d'identité, état civil, nationalité, élections, recours contre l'administration).

**Pourquoi ces deux thèmes.** Ils correspondent aux deux exemples de questions donnés par le sujet (« que faut-il pour renouveler ma carte d'identité ? », « mon propriétaire peut-il garder mon dépôt de garantie ? »), et ils sont pris **en entier** : si une question porte sur l'un de ces sujets, la réponse se trouve dans le corpus. Nous avons écarté un échantillon aléatoire de fiches sur tous les thèmes, qui aurait laissé des trous (un sujet en apparence couvert, mais dont la fiche n'aurait pas été tirée).

**Attention au périmètre réel.** Le thème « Papiers - Citoyenneté - Élections » est plus large que les seuls papiers d'identité : il couvre aussi les élections, les recours contre l'administration, les volontariats, l'état civil, la nationalité, le nom de famille et certaines distinctions honorifiques.

## Sélection

Le script `prepare_docs.py` applique trois filtres successifs :

| Étape | Fiches |
|---|---|
| 1. Fiches pratiques de l'archive (`F*.xml`) | 3 003 |
| 2. Thème principal (1er niveau du fil d'Ariane) « Logement » ou « Papiers - Citoyenneté - Élections » | 549 (350 + 199) |
| 3. Texte converti d'au moins 200 caractères (en dessous : pages quasi vides) | **526** (333 + 193) |

Aucun tri manuel n'est appliqué : la sélection repose uniquement sur le thème déclaré par le site, ce qui la rend reproductible et facile à justifier.

## Description du corpus

### Répartition

| Thème | Question-réponse | Fiche d'information | Liens externes | Total |
|---|---|---|---|---|
| Logement | 176 | 157 | 0 | 333 |
| Papiers - Citoyenneté - Élections | 77 | 113 | 3 | 193 |
| **Total** | **253** | **270** | **3** | **526** |

Les trois fiches « avec liens externes » (élection présidentielle, élections européennes, référendum) ont été relues : une fois les liens retirés, elles conservent des informations utiles (dates des scrutins, règles électorales) et sont gardées.

Sous-thèmes les plus fournis :

| Sous-thème | Fiches |
|---|---|
| Logement > Autorisations d'urbanisme | 52 |
| Papiers > Élections | 34 |
| Papiers > Carte d'identité | 27 |
| Logement > Location immobilière : bail d'habitation | 27 |
| Logement > Bail d'habitation : loyer | 25 |
| Logement > Aides et prêts pour la rénovation énergétique | 20 |
| Logement > Achat ou vente d'un logement | 20 |
| Papiers > Passeport | 18 |
| Papiers > Agir en justice contre l'administration | 17 |
| Papiers > Volontariats | 15 |

### Longueur des fiches

| Indicateur | Caractères |
|---|---|
| Minimum | 333 |
| 10 % des fiches en dessous de | 2 409 |
| 25 % | 4 660 |
| Médiane | 9 417 |
| 75 % | 17 003 |
| 90 % | 32 280 |
| 99 % | 133 051 |
| Maximum | 243 021 |
| Moyenne (écart-type) | 15 843 (24 243) |

La distribution est très asymétrique : l'écart-type dépasse la moyenne. Les cinq fiches les plus longues pèsent autant qu'une centaine de fiches médianes :

| Fiche | Titre | Caractères |
|---|---|---|
| F1314 | Encadrement des loyers en zone tendue | 243 021 |
| F1372 | Inscription sur la liste électorale en cas de déménagement | 222 336 |
| F31473 | Maison : travaux extérieurs | 210 418 |
| F920 | Rédaction du bail d'habitation | 189 915 |
| F32880 | Renouvellement d'une demande de logement social | 147 995 |

Ce sont des fiches « conditionnées » : une grande partie de leur contenu est répétée pour chaque situation (locataire ou propriétaire, logement vide ou meublé, région, département...). Les fiches comptent en médiane 15 sections, et jusqu'à 456.

### Dates de dernière mise à jour

Du 22/06/2021 au 09/10/2026 :

| Année | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| Fiches | 1 | 3 | 1 | 19 | 158 | 344 |

Le corpus est très récent : 95 % des fiches ont été mises à jour en 2025 ou 2026.

## Nettoyage et mise en forme

Chaque fiche XML est convertie en texte :

- **Conservé :** le résumé, l'introduction, le texte principal, les listes et les tableaux (une ligne par rangée, cellules séparées par « | », par exemple `Piscine | 251 € par m²`). Chaque chapitre, situation ou cas devient un titre de section qui garde le chemin complet (`## En Île-de-France > Comment est calculée la taxe d'aménagement ?`), pour qu'un passage reste compréhensible hors de son contexte.
- **Résumé en une ligne :** les formulaires (avec leur numéro cerfa), téléservices, simulateurs et interlocuteurs cités dans le texte, par exemple `Formulaire : Déclaration préalable constructions et travaux non soumis à permis de construire (cerfa n°16702*03)`.
- **Ajouté en fin de fiche :** les sections « Qui peut m'aider ? » et « Textes de référence » (articles de loi).
- **Retiré :** les titres en double, les URL, les notices et annexes des formulaires, et les renvois vers d'autres pages (« Voir aussi », « Pour en savoir plus », questions associées, liste des fiches du même dossier). Ces renvois ne contiennent que le titre d'une autre page : les garder ferait remonter des passages qui citent un sujet sans en parler. Leur contenu est de toute façon présent dans les fiches correspondantes.

Champs produits dans `data/raw/corpus.parquet` : `doc_id`, `title`, `text`, `theme`, `fil_ariane`, `date`, `url`, `type`.

## Contrôles qualité

| Contrôle | Résultat |
|---|---|
| Couverture du texte : chaque paragraphe du XML d'origine se retrouve-t-il dans le texte nettoyé ? | **64 264 paragraphes sur 64 264 (100 %)** |
| Identifiants uniques, au format `F1234` | Oui |
| Champs vides (titre, texte, date, URL) | Aucun |
| Balises XML restantes dans le texte | Aucune |
| Titres en double | Aucun |
| Fiches contenant des tableaux | *à compléter* |
| Relecture manuelle de fiches comparées au site (dont F23263, F36798, F1940) | Conformes |

Remarques issues des contrôles :

- La fiche F23263 (taxe d'aménagement) a une URL sur `entreprendre.service-public.gouv.fr` : elle est partagée avec la rubrique Entreprendre du site.
- 24 titres de section sont suivis d'un paragraphe qui reprend leur intitulé : ce doublon vient de la fiche d'origine elle-même (libellé d'un lien), il est inoffensif.

## Limites et risques

- **Instantané daté :** le droit du logement et des démarches évolue (montants, plafonds, dates d'élection). Les réponses reflètent l'état des fiches au téléchargement, pas forcément le droit en vigueur au moment de la question. Cinq fiches n'ont pas été mises à jour depuis 2023 ou avant. La date disponible est celle de la dernière modification de la page, pas celle de la règle juridique.
- **Vulgarisation, pas texte de loi :** les fiches résument le droit ; l'assistant ne remplace pas un conseil juridique.
- **Fiches très longues et répétitives :** elles produisent beaucoup de passages presque identiques, qui peuvent occuper une grande partie des résultats de recherche.
- **Contenu conditionnel :** une même information varie selon la situation, et seul le titre de section l'indique. Exemple : la valeur de la taxe d'aménagement en 2026 est de 892 € par m² en général, mais de 1 011 € en Île-de-France. Un passage mal choisi peut donner une réponse valable pour une autre situation que celle de l'utilisateur.
- **Informations locales :** certaines règles dépendent de la commune, du département ou de la région, ce qui exige de connaître la situation de l'utilisateur.
- **Liens supprimés :** l'assistant peut nommer un formulaire ou un téléservice, mais pas donner son lien direct (seule l'URL de la fiche est conservée).
- **Défauts de la source :** quelques espaces manquent après des chiffres mis en évidence dans le XML (« 50 %sur », « 5 000 m²ou »), et quelques coquilles d'origine subsistent (« référendu », « une 1 fois »).

## Reproduction

```bash
uv run python prepare_docs.py --list   # nombre de fiches par thème dans l'archive
uv run python prepare_docs.py          # génère data/raw/corpus.parquet
uv run python check_coverage.py        # couverture du texte (100 % attendu)
uv run python check_corpus.py          # contrôles de forme
uv run python inspect_corpus.py        # statistiques de cette data card
```

## Découpage (à compléter après l'ingestion)

*Stratégie retenue (découpage par section, taille maximale des chunks), nombre de chunks obtenus, et résultats comparés des stratégies testées sur le golden set.*
