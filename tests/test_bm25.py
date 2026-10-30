import math
from src.index import InvertedIndex
from src.bm25 import bm25_scores, bm25_idf


def toy():
    idx = InvertedIndex()
    idx.add_document(0, ["python", "python", "fun"])
    idx.add_document(1, ["snake", "long"])
    idx.add_document(2, ["python", "snake"])
    return idx


def test_matches_formula():
    idx = toy()
    k1, b = 1.5, 0.75
    avgdl = idx.avg_doc_length            # (3 + 2 + 2) / 3
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expected = idf * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 3 / avgdl))
    assert math.isclose(bm25_scores(idx, ["python"])[0], expected)
    assert math.isclose(bm25_idf(3, 2), idf)


def test_shorter_doc_wins_with_same_tf():
    idx = InvertedIndex()
    idx.add_document(0, ["cat"] + ["x"] * 20)
    idx.add_document(1, ["cat", "y"])
    scores = bm25_scores(idx, ["cat"])
    assert scores[1] > scores[0]


def test_b_zero_ignores_length():
    idx = InvertedIndex()
    idx.add_document(0, ["cat"] + ["x"] * 20)
    idx.add_document(1, ["cat", "y"])
    scores = bm25_scores(idx, ["cat"], b=0.0)
    assert math.isclose(scores[0], scores[1])


def test_unknown_term():
    assert bm25_scores(toy(), ["zzz"]) == {}
