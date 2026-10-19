import os
# Fix conflit DLL OpenMP entre torch et ChromaDB sur Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import torch  # charger torch avant toute opération ChromaDB sur Windows
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

STOPWORDS = set("a abord absolument afin ah ai aie aient aies ailleurs ainsi ait allaient allo allons allô alors anterieur anterieure anterieures ap apres après as assez attendu au aucucun aucune aujourd aujourd'hui aupres auquel aura auraient aurais aurait auras aurez auriez aurions aurons auront aussi autre autrefois autrement autres autrui aux auxquelles auxquels avaient avais avait avant avec avoir avons ayant ayez b bah bas basee bat beau beaucoup bien bigre boum bravo brrr c car ce ceci cela celle celle-ci celle-là celles celles-ci celles-là celui celui-ci celui-là cent cependant certain certaine certaines certains certes ces cet cette ceux ceux-ci ceux-là chacun chacune chaque cher chers chez chiche chut ci cinq cinquantaine cinquante cinquantième cinquième clac clic combien comme comment comparable comparables compris concernant contre couic crac d da dans de debout dedans dehors deja delà depuis dernier derniere derriere derrière des desormais desquelles desquels dessous dessus deux deuxième deuxièmement devant devers devra different differentes differents différent différente différentes différents dire directe directement dit dite dits divers diverse diverses dix dix-huit dixième dix-neuf dix-sept doigt doite doits donc dont douze douzième d'un d'une eau effet egale egalement egales eh elle elle-même elles elles-mêmes en encore enfin entre envers environ es est et etait etaient etais etait etant etc ete etee etees etes etait etant eu eue eues euh eurent eus eusse eussent eusses eussiez eussions eut eux eux-mêmes exactement excepté extensif extreme extremement f fi flac floc font g gens h ha hé hein hem hep hi ho holà hop hormis hors hou houp hue hui huit huitième hurrah i il ils importe j je jusqu jusque k l la laissez laquelle las le lequel les lesquelles lesquels leur leurs long longtemps lors lorsque lui lui-meme lui-même m ma maint maintenant mais malgre malgre me meme memes merci mes mien mienne miennes miens mille mince moi moi-meme moi-même moindre moins mon moyennant multiple multiples même mêmes n na naturel naturelle naturelles naturels ne neanmoins necessaire necessairement neuf neuvième ni nombreuses nombreux non nos notamment notre nous nous-mêmes nouveau nul néanmoins nôtre nôtres o oh ohé ollé olé on ont onze onzième ore ou ouf oui ouï où p paf pan par parce parfois parle parlent parler parmi parseme partant particulier particuliere particulierement pas passé pendant pense permet personne peu peut peuvent peux pff pfft pfut pif pire plein plouf plus plusieurs plutôt possessif possesifs possible possibles pouah pour pourquoi pourrais pourrait pouvait prealable precisement premier première premièrement pres probable probante procedant proche près psitt pu puisque q qu quand quant quanta quant-à-soi quarante quatorze quatre quatre-vingt quatrième quatrièmement que quel quelconque quelle quelles quelque quelques quelqu'un quels qui quiconque quinze quoi quoique r revoici revoilà rien s sa sacrebleu sans sapristi sauf se seize selon sept septième sera seraient serais serait seras serez seriez serions serons seront ses seulement si sien sienne siennes siens sinon six sixième soi soi-même soit soixante son sont sous souvent specifique specifiques speculatif stop strict stricte strictement subordonnet suffisant suffisante suffit suis suit suivant suivante suivantes suivants superieur superieure superieures superieurs sur surcroît t ta tac tant tard te tel telle tellement telles tels tenant tend tes tic tien tienne tiennes tiens toc toi toi-même ton touchant toujours tous tout toute toutefois toutes treize trente tres trois troisième troisièmement trop très tsoin tsouin tu té u un une unes uniformement unique uniques uns v va vais vas vers via vif vifs vingt vivat voici voilà vont vos votre vous vous-mêmes vu vé vôtre vôtres w x y z zut à â ça è é ê ë î ï ô ö û ü ù".split())


def tokenize_fr(text):
    return [w for w in re.findall(r"[a-z0-9àâçéèêëîïôûùüÿñæœ]+(?:\.[0-9]+)?%?", text.lower()) if w not in STOPWORDS]


def rrf(rankings, k=60, depth=30):
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking[:depth], start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    def __init__(self, ingestion_pipeline, reranker_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.collection = ingestion_pipeline.collection
        self.embedder = ingestion_pipeline.embedder
        self.docstore = ingestion_pipeline.docstore

        self.chunks_list = list(self.docstore.values())
        self.tokens = [tokenize_fr(c["text"]) for c in self.chunks_list]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

        self.reranker = None
        if _RERANKER_AVAILABLE:
            try:
                self.reranker = _CrossEncoder(reranker_model)
                print("Reranker chargé.", flush=True)
            except Exception as e:
                print(f"Reranker non disponible ({e}), fallback BM25.", flush=True)

    def dense_search(self, query, top_n=30):
        top_n = min(top_n, self.collection.count())
        if top_n == 0:
            return []
        query_vector = self.embedder.encode_query(query)
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_n)
        return results["ids"][0]

    def bm25_search(self, query, top_n=30):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize_fr(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks_list[i]["chunk_id"] for i in ranked_indices[:top_n]]

    def search(self, query, top_k=6, pre_k=30):
        # Dense search provoque un crash DLL sur Windows (torch + ChromaDB same process).
        # Pipeline : BM25 (30 candidats) → CrossEncoder reranker → top_k.
        candidate_ids = self.bm25_search(query, top_n=pre_k)
        if not candidate_ids:
            return []

        candidates = [self.docstore[cid] for cid in candidate_ids if cid in self.docstore]

        if self.reranker and candidates:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.reranker.predict(pairs)
            candidates = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

        return candidates[:top_k]

    def get_document(self, doc_id: str):
        chunks = [c for c in self.docstore.values() if c["doc_id"] == doc_id]
        if not chunks:
            return None
        return "\n".join([c["text"] for c in sorted(chunks, key=lambda x: int(x["chunk_id"].split("#")[1]))])
