import math


def tfidf_scores(index, terms):
    """Return {doc_id: score}. tf = 1 + log(freq), idf = log(N / df)."""
    scores = {}
    n_docs = index.num_docs
    for term in dict.fromkeys(terms):          # unique terms, keep order
        postings = index.postings(term)
        df = len(postings)
        if df == 0:
            continue
        idf = math.log(n_docs / df)
        for doc_id, freq in postings.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + (1 + math.log(freq)) * idf
    return scores
