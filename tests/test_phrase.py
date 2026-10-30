from src.phrase import phrase_search


def test_phrase_match(index):
    assert phrase_search(index, ["machine", "learning"]) == [3]


def test_reversed_phrase_matches_only_reversed_doc(index):
    assert phrase_search(index, ["learning", "machine"]) == [4]


def test_three_word_phrase(index):
    assert phrase_search(index, ["python", "is", "fun"]) == [0]


def test_words_present_but_not_adjacent(index):
    assert phrase_search(index, ["python", "fun"]) == []


def test_single_term_phrase(index):
    assert phrase_search(index, ["snake"]) == [1, 2]


def test_unknown_and_empty(index):
    assert phrase_search(index, ["machine", "zzz"]) == []
    assert phrase_search(index, []) == []
