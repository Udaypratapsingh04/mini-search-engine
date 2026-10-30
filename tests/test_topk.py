import random
from src.topk import top_k_sorted, top_k_nlargest, top_k_minheap


def test_all_methods_agree():
    random.seed(1)
    scores = {i: random.random() for i in range(500)}
    a = top_k_sorted(scores, 10)
    assert top_k_nlargest(scores, 10) == a
    assert top_k_minheap(scores, 10) == a


def test_ties_prefer_lower_doc_id():
    scores = {5: 1.0, 2: 1.0, 9: 1.0}
    assert [d for d, _ in top_k_minheap(scores, 2)] == [2, 5]
    assert [d for d, _ in top_k_nlargest(scores, 2)] == [2, 5]


def test_k_larger_than_n():
    scores = {1: 0.5, 2: 0.7}
    assert top_k_minheap(scores, 10) == [(2, 0.7), (1, 0.5)]


def test_empty_and_zero_k():
    assert top_k_minheap({}, 5) == []
    assert top_k_minheap({1: 1.0}, 0) == []
