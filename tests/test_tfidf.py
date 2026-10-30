import math
from src.index import InvertedIndex
from src.tfidf import tfidf_scores


def toy():
    idx = InvertedIndex()
    idx.add_document(0, ["python", "python", "fun"])
    idx.add_document(1, ["snake", "long"])
    idx.add_document(2, ["python", "snake"])
    return idx


def test_matches_hand_calculation():
    scores = tfidf_scores(toy(), ["python"])
    idf = math.log(3 / 2)
    assert math.isclose(scores[0], (1 + math.log(2)) * idf)
    assert math.isclose(scores[2], 1 * idf)
    assert 1 not in scores


def test_multiple_terms_sum():
    scores = tfidf_scores(toy(), ["python", "snake"])
    idf = math.log(3 / 2)
    assert math.isclose(scores[2], 2 * idf)


def test_unknown_term():
    assert tfidf_scores(toy(), ["zzz"]) == {}
