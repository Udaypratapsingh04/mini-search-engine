from src.vocab import build_vocab, save_vocab, load_vocab

DOCS = [
    {"id": 0, "title": "python snake", "text": "python"},
    {"id": 1, "title": "snake", "text": "rocks"},
    {"id": 2, "title": "python", "text": "123 x"},
]


def test_build_vocab_counts_document_frequency_and_filters():
    vocab = build_vocab(DOCS, min_df=2)
    assert vocab == {"python": 2, "snake": 2}     # rocks: df 1; "123": not alpha; "x": too short


def test_min_df_one_keeps_rare_words():
    assert "rocks" in build_vocab(DOCS, min_df=1)


def test_save_load_round_trip(tmp_path):
    vocab = {"python": 2, "snake": 2}
    path = str(tmp_path / "v.pkl")
    save_vocab(vocab, path)
    assert load_vocab(path) == vocab
