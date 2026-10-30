from src.tokenizer import tokenize


def test_basic():
    assert tokenize("The Quick, Brown Fox!") == ["quick", "brown", "fox"]

def test_empty():
    assert tokenize("") == []

def test_punctuation_only():
    assert tokenize("!!! ... ,,, ???") == []

def test_numbers_kept():
    assert tokenize("Python 3 released in 2008") == ["python", "3", "released", "2008"]

def test_capital_letters():
    assert tokenize("PYTHON Python python") == ["python", "python", "python"]

def test_only_stopwords():
    assert tokenize("the and is of to") == []
