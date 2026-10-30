"""Boolean search: AND, OR, NOT (uppercase). No brackets.

Precedence: NOT, then AND, then OR.
Adjacent terms without an operator are treated as AND.
"""

OPERATORS = {"AND", "OR", "NOT"}


# ---- two-pointer merges on sorted lists ----
def intersect_sorted(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union_sorted(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference_sorted(a, b):
    """Elements of a that are not in b."""
    i = j = 0
    out = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i]); i += 1
        elif a[i] == b[j]:
            i += 1; j += 1
        else:
            j += 1
    return out


# ---- parsing ----
def parse(query: str):
    """Split into OR-clauses. Each clause = (positive_words, negative_words)."""
    clauses = []
    pos, neg, negate = [], [], False
    for tok in query.split():
        if tok == "OR":
            clauses.append((pos, neg))
            pos, neg, negate = [], [], False
        elif tok == "AND":
            continue
        elif tok == "NOT":
            negate = not negate
        else:
            (neg if negate else pos).append(tok)
            negate = False
    clauses.append((pos, neg))
    return [c for c in clauses if c[0] or c[1]]


def _tokens(words, preprocess_fn):
    out = []
    for w in words:
        out.extend(preprocess_fn(w))
    return out


def _postings_list(index, term):
    return sorted(index.positions(term))


def boolean_search(query: str, index, preprocess_fn):
    """Return (sorted doc_ids, positive terms to use for ranking)."""
    result = []
    rank_terms = []
    for pos_words, neg_words in parse(query):
        pos = _tokens(pos_words, preprocess_fn)
        neg = _tokens(neg_words, preprocess_fn)
        if not pos and not neg:                 # clause was only stopwords
            continue
        if pos:
            docs = _postings_list(index, pos[0])
            for term in pos[1:]:
                docs = intersect_sorted(docs, _postings_list(index, term))
        else:                                   # only NOT: all docs minus ...
            docs = sorted(index.doc_lengths)
        for term in neg:
            docs = difference_sorted(docs, _postings_list(index, term))
        result = union_sorted(result, docs)
        rank_terms.extend(pos)
    return result, rank_terms
