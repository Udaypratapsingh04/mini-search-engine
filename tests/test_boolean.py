from src.boolean_query import (
    boolean_search, intersect_sorted, union_sorted, difference_sorted, parse,
)
from tests.conftest import pp


def run(q, index):
    return boolean_search(q, index, pp)[0]


def test_merges():
    assert intersect_sorted([1, 3, 5, 7], [3, 4, 5, 8]) == [3, 5]
    assert union_sorted([1, 3], [2, 3, 4]) == [1, 2, 3, 4]
    assert difference_sorted([1, 2, 3, 4], [2, 4]) == [1, 3]
    assert intersect_sorted([], [1]) == []


def test_and(index):
    assert run("python AND snake", index) == [2]


def test_or(index):
    assert run("python OR snake", index) == [0, 1, 2]


def test_and_not(index):
    assert run("python AND NOT snake", index) == [0]


def test_only_not(index):
    assert run("NOT python", index) == [1, 3, 4]


def test_precedence_and_before_or(index):
    # fun AND python OR snake  ==  (fun AND python) OR snake
    assert run("fun AND python OR snake", index) == [0, 1, 2]


def test_implicit_and(index):
    assert run("python snake", index) == [2]


def test_unknown_term(index):
    assert run("zzz", index) == []
    assert run("python AND zzz", index) == []
    assert run("python OR zzz", index) == [0, 2]


def test_empty_query(index):
    assert run("", index) == []
    assert run("   ", index) == []


def test_parse_structure():
    assert parse("a AND NOT b OR c") == [(["a"], ["b"]), (["c"], [])]
