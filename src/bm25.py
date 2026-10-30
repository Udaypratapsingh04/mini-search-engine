import math


def bm25_idf(n_docs: int, df: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_scores(index, terms, k1: float = 1.5, b: float = 0.75):
    """Return {doc_id: score} using Okapi BM25."""
    scores = {}
    n_docs = index.num_docs
    avgdl = index.avg_doc_length or 1.0
    for term in dict.fromkeys(terms):
        postings = index.postings(term)
        df = len(postings)
        if df == 0:
            continue
        idf = bm25_idf(n_docs, df)
        for doc_id, f in postings.items():
            norm = f + k1 * (1 - b + b * index.doc_lengths[doc_id] / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * f * (k1 + 1) / norm
    return scores
