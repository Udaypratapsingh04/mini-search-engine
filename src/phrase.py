def phrase_search(index, terms):
    """Doc ids (sorted) where terms appear consecutively, in order.

    Note: stopwords are removed before indexing, so "state of the art"
    behaves like "state art". Mention this in the README.
    """
    if not terms:
        return []
    pos_maps = [index.positions(t) for t in terms]
    if any(not m for m in pos_maps):
        return []

    candidates = set(pos_maps[0])
    for m in pos_maps[1:]:
        candidates &= set(m)

    hits = []
    for doc_id in sorted(candidates):
        later = [set(m[doc_id]) for m in pos_maps[1:]]
        for p in pos_maps[0][doc_id]:
            if all((p + i + 1) in s for i, s in enumerate(later)):
                hits.append(doc_id)
                break
    return hits
