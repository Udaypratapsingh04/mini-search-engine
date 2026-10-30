import pytest
from src.index import InvertedIndex


def make_index():
    idx = InvertedIndex()
    idx.add_document(0, ["python", "is", "fun", "python"])
    idx.add_document(1, ["snake", "is", "long"])
    idx.add_document(2, ["python", "snake"])
    return idx


def test_postings_frequencies():
    idx = make_index()
    assert idx.postings("python") == {0: 2, 2: 1}
    assert idx.postings("snake") == {1: 1, 2: 1}

def test_positions():
    idx = make_index()
    assert idx.positions("python")[0] == [0, 3]

def test_unknown_term():
    idx = make_index()
    assert idx.postings("zzz") == {}
    assert "zzz" not in idx.index        # lookup must not pollute the index

def test_stats():
    idx = make_index()
    assert idx.num_docs == 3
    assert idx.doc_lengths == {0: 4, 1: 3, 2: 2}
    assert idx.avg_doc_length == 3.0
    assert idx.doc_freq("python") == 2

def test_duplicate_doc_id_rejected():
    idx = make_index()
    with pytest.raises(ValueError):
        idx.add_document(0, ["x"])

def test_save_load_round_trip(tmp_path):
    idx = make_index()
    path = tmp_path / "idx.pkl"
    idx.save(str(path))
    loaded = InvertedIndex.load(str(path))
    assert loaded.postings("python") == idx.postings("python")
    assert loaded.positions("snake") == idx.positions("snake")
    assert loaded.doc_lengths == idx.doc_lengths
    assert loaded.num_docs == idx.num_docs
    assert loaded.avg_doc_length == idx.avg_doc_length
