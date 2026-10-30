from src.index import InvertedIndex
from src.spellcheck import SpellChecker, did_you_mean
from tests.conftest import pp

VOCAB = {"computer": 100, "complete": 40, "graphics": 60, "graphic": 10,
         "science": 30, "space": 80, "spec": 5, "the": 500}


def test_correct_simple_typos():
    sc = SpellChecker(VOCAB)
    assert sc.correct("comptuer") == "computer"      # transposition
    assert sc.correct("graphcis") == "graphics"
    assert sc.correct("scence") == "science"         # deletion


def test_known_word_unchanged():
    assert SpellChecker(VOCAB).correct("space") == "space"


def test_tie_prefers_higher_frequency():
    # "spce" is distance 1 from both "space" (80) and "spec" (5)
    assert SpellChecker(VOCAB).correct("spce") == "space"


def test_too_far_or_unsuitable_returns_none():
    sc = SpellChecker(VOCAB)
    assert sc.correct("qqqqqqq") is None
    assert sc.correct("ab") is None                  # too short
    assert sc.correct("abc123") is None              # not alphabetic


def test_did_you_mean():
    idx = InvertedIndex()
    idx.add_document(0, ["computer", "graphics", "science", "space"])
    sc = SpellChecker(VOCAB)
    assert did_you_mean("graphcis card", idx, sc, pp) == "graphics card"
    assert did_you_mean("comptuer science", idx, sc, pp) == "computer science"
    assert did_you_mean("computer science", idx, sc, pp) is None
    assert did_you_mean("comptuer AND NOT scence", idx, sc, pp) == "computer AND NOT science"
    assert did_you_mean('"comptuer science"', idx, sc, pp) == '"computer science"'
    assert did_you_mean("", idx, sc, pp) is None
