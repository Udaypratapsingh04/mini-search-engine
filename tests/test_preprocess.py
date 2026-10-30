from src.preprocess import preprocess


def test_running_and_runs_same_stem():
    assert preprocess("running") == preprocess("runs")

def test_stopwords_removed_then_stemmed():
    assert preprocess("The cats are running") == ["cat", "run"]

def test_empty():
    assert preprocess("") == []

def test_over_stemming_limitation():
    # Known limitation to mention in the README
    assert preprocess("university") == preprocess("universe")
