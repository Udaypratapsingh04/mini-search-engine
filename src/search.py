import re
from typing import NamedTuple

from src.bm25 import bm25_scores
from src.tfidf import tfidf_scores
from src.topk import top_k_nlargest
from src.boolean_query import boolean_search, OPERATORS
from src.phrase import phrase_search

PHRASE_RE = re.compile(r'"([^"]+)"')


class Result(NamedTuple):
    doc_id: int
    score: float
    title: str
    snippet: str


class SearchEngine:
    def __init__(self, index, docs, preprocess_fn=None):
        if preprocess_fn is None:
            from src.preprocess import preprocess
            preprocess_fn = preprocess
        self.index = index
        self.docs = {d["id"]: d for d in docs}
        self.preprocess = preprocess_fn

    def _score(self, terms, model, k1, b):
        if model == "tfidf":
            return tfidf_scores(self.index, terms)
        return bm25_scores(self.index, terms, k1=k1, b=b)

    def _format(self, ranked):
        results = []
        for doc_id, score in ranked:
            doc = self.docs[doc_id]
            snippet = doc["text"][:150].replace("\n", " ")
            results.append(Result(doc_id, score, doc["title"], snippet))
        return results

    def search(self, query, k=10, model="bm25", k1=1.5, b=0.75):
        query = query.strip()
        if not query:
            return []

        phrase = PHRASE_RE.fullmatch(query)
        if phrase:                                   # "machine learning"
            terms = self.preprocess(phrase.group(1))
            matched = set(phrase_search(self.index, terms))
            rank_terms = terms
        elif any(tok in OPERATORS for tok in query.split()):   # boolean
            ids, rank_terms = boolean_search(query, self.index, self.preprocess)
            matched = set(ids)
        else:                                        # plain ranked query
            terms = self.preprocess(query)
            return self._format(top_k_nlargest(self._score(terms, model, k1, b), k))

        if not matched:
            return []
        scores = self._score(rank_terms, model, k1, b)
        scores = {d: scores.get(d, 0.0) for d in matched}
        return self._format(top_k_nlargest(scores, k))
