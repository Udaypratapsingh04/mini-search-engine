import os
import pickle

import pytest

from app.service import SearchService, resolve_data_dir
from src.vocab import save_vocab
from tests.conftest import DOCS, pp


def make_vocab():
    vocab = {}
    for d in DOCS:
        for w in set(pp(d["text"])):
            vocab[w] = vocab.get(w, 0) + 1
    return vocab


def make_service(index, docs):
    return SearchService(index, docs, make_vocab(), preprocess_fn=pp)


def test_search_payload(index, docs):
    out = make_service(index, docs).search("python", k=5)
    assert out["query"] == "python"
    assert out["k"] == 5
    assert out["count"] == len(out["results"]) == 2
    assert out["took_ms"] >= 0
    assert out["did_you_mean"] is None
    first = out["results"][0]
    assert set(first) == {"doc_id", "score", "title", "snippet"}
    assert first["doc_id"] == 0


def test_query_is_stripped(index, docs):
    assert make_service(index, docs).search("   python  ")["query"] == "python"


def test_boolean_and_phrase(index, docs):
    svc = make_service(index, docs)
    assert [r["doc_id"] for r in svc.search("python AND NOT snake")["results"]] == [0]
    assert [r["doc_id"] for r in svc.search('"machine learning"')["results"]] == [3]


def test_did_you_mean(index, docs):
    out = make_service(index, docs).search("pythn")
    assert out["did_you_mean"] == "python"
    assert out["count"] == 0


def test_empty_query_returns_nothing(index, docs):
    out = make_service(index, docs).search("   ")
    assert out["count"] == 0 and out["results"] == []


def test_suggest(index, docs):
    svc = make_service(index, docs)
    assert svc.suggest("py")["suggestions"] == ["python"]
    assert svc.suggest("p")["suggestions"] == []        # shorter than 2 chars
    assert svc.suggest("zz")["suggestions"] == []


def test_health(index, docs):
    h = make_service(index, docs).health()
    assert h == {"status": "ok", "docs": 5, "vocab_words": len(make_vocab())}


def test_from_data_dir_round_trip(index, docs, tmp_path):
    index.save(str(tmp_path / "index.pkl"))
    save_vocab(make_vocab(), str(tmp_path / "vocab.pkl"))
    with open(tmp_path / "docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    svc = SearchService.from_data_dir(str(tmp_path), preprocess_fn=pp)
    assert svc.health()["docs"] == 5
    assert svc.search("python")["results"][0]["doc_id"] == 0


def test_from_data_dir_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        SearchService.from_data_dir(str(tmp_path), preprocess_fn=pp)
    assert "build_index" in str(err.value)


def test_resolve_data_dir_env_override():
    old = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = "somewhere"
    try:
        assert resolve_data_dir() == "somewhere"
    finally:
        if old is None:
            del os.environ["DATA_DIR"]
        else:
            os.environ["DATA_DIR"] = old
