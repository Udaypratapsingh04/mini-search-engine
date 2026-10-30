"""Three ways to get the best K of {doc_id: score}.

Full sort:   O(n log n)
heapq:       O(n log k)
Manual heap: O(n log k)
Ties are broken by lower doc_id first, so all three agree.
"""
import heapq


def _key(item):
    return (item[1], -item[0])


def top_k_sorted(scores: dict, k: int):
    return sorted(scores.items(), key=_key, reverse=True)[:k]


def top_k_nlargest(scores: dict, k: int):
    return heapq.nlargest(k, scores.items(), key=_key)


def top_k_minheap(scores: dict, k: int):
    """Keep a min-heap of size k; the smallest of the best k sits at heap[0]."""
    if k <= 0:
        return []
    heap = []                                   # entries: (score, -doc_id)
    for doc_id, score in scores.items():
        entry = (score, -doc_id)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)      # pop smallest + push new
    heap.sort(reverse=True)
    return [(-neg_id, score) for score, neg_id in heap]
