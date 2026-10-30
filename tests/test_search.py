from src.search import SearchEngine
from tests.conftest import pp


def engine(index, docs):
    return SearchEngine(index, docs, preprocess_fn=pp)


def test_ranked_query(index, docs):
    res = engine(index, docs).search("python")
    assert [r.doc_id for r in res] == [0, 2]      # doc 0 has higher tf
    assert res[0].title == "Python basics"
    assert len(res[0].snippet) <= 150


def test_k_limits_results(index, docs):
    assert len(engine(index, docs).search("is fun python snake", k=2)) == 2


def test_boolean_through_engine(index, docs):
    res = engine(index, docs).search("python AND NOT snake")
    assert [r.doc_id for r in res] == [0]


def test_phrase_through_engine(index, docs):
    res = engine(index, docs).search('"machine learning"')
    assert [r.doc_id for r in res] == [3]
    assert engine(index, docs).search('"learning machine"')[0].doc_id == 4


def test_empty_and_unknown(index, docs):
    e = engine(index, docs)
    assert e.search("") == []
    assert e.search("zzz") == []
    assert e.search('"zzz qqq"') == []


def test_tfidf_model_option(index, docs):
    res = engine(index, docs).search("python", model="tfidf")
    assert res[0].doc_id == 0
