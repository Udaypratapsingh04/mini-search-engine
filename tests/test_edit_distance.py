import random
from src.edit_distance import levenshtein, damerau_levenshtein


def test_known_values():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("same", "same") == 0
    assert levenshtein("", "") == 0


def test_symmetric():
    assert levenshtein("flaw", "lawn") == levenshtein("lawn", "flaw") == 2


def test_transposition():
    assert levenshtein("teh", "the") == 2
    assert damerau_levenshtein("teh", "the") == 1
    assert damerau_levenshtein("kitten", "sitting") == 3
    assert damerau_levenshtein("", "abc") == 3


def test_max_dist_early_exit_agrees_with_full():
    rng = random.Random(7)
    letters = "abcd"
    for _ in range(300):
        a = "".join(rng.choice(letters) for _ in range(rng.randint(0, 7)))
        b = "".join(rng.choice(letters) for _ in range(rng.randint(0, 7)))
        for limit in (1, 2, 3):
            assert levenshtein(a, b, limit) == min(levenshtein(a, b), limit + 1)
            assert damerau_levenshtein(a, b, limit) == min(damerau_levenshtein(a, b), limit + 1)
