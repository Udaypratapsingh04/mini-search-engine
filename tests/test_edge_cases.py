"""Day 18: edge cases. None of these may crash."""
import pytest
from src.search import SearchEngine
from tests.conftest import pp


def engine(index, docs):
    return SearchEngine(index, docs, preprocess_fn=pp)


def test_empty_and_whitespace_query(index, docs):
    e = engine(index, docs)
    assert e.search("") == []
    assert e.search("     ") == []
    assert e.search("\t\n") == []


def test_very_long_query(index, docs):
    e = engine(index, docs)
    res = e.search("python " * 5000)
    assert [r.doc_id for r in res] == [0, 2]
    assert e.search("zzz " * 5000) == []


def test_very_long_single_token(index, docs):
    assert engine(index, docs).search("a" * 100_000) == []


@pytest.mark.parametrize("q", [
    "!@#$%^&*()",
    '"',
    '""',
    '"" ""',
    "'; DROP TABLE docs;--",
    "<script>alert(1)</script>",
    "\\ / | ? * + [ ] { }",
    "日本語 café naïve",
    "AND OR NOT",
    "NOT",
    "AND",
    '"python',
    'python "snake',
    "((python))",
])
def test_special_characters_never_crash(index, docs, q):
    res = engine(index, docs).search(q)
    assert isinstance(res, list)


def test_dangling_operators(index, docs):
    e = engine(index, docs)
    assert {r.doc_id for r in e.search("python AND")} == {0, 2}
    assert {r.doc_id for r in e.search("AND python")} == {0, 2}
    assert {r.doc_id for r in e.search("python OR")} == {0, 2}
    assert e.search("AND OR NOT") == []


def test_k_edge_values(index, docs):
    e = engine(index, docs)
    assert e.search("python", k=0) == []
    assert len(e.search("is fun python snake machine", k=1000)) <= len(docs)


def test_duplicate_terms_do_not_double_count(index, docs):
    e = engine(index, docs)
    once = e.search("python")[0].score
    many = e.search("python python python")[0].score
    assert once == many
