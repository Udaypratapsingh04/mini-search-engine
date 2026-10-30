"""API tests. Needs: pip install fastapi httpx (both are in requirements.txt)."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app, MAX_QUERY_LEN
from app.service import SearchService
from tests.conftest import DOCS, pp
from tests.test_service import make_vocab


@pytest.fixture
def client(index, docs):
    service = SearchService(index, docs, make_vocab(), preprocess_fn=pp)
    with TestClient(create_app(service=service)) as c:     # `with` runs startup
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["docs"] == len(DOCS)


def test_search_ok(client):
    r = client.get("/search", params={"q": "python", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["took_ms"] >= 0
    assert body["results"][0]["title"] == "Python basics"
    assert set(body["results"][0]) == {"doc_id", "score", "title", "snippet"}


def test_search_boolean_and_phrase(client):
    r = client.get("/search", params={"q": "python AND NOT snake"})
    assert [h["doc_id"] for h in r.json()["results"]] == [0]
    r = client.get("/search", params={"q": '"machine learning"'})
    assert [h["doc_id"] for h in r.json()["results"]] == [3]


def test_did_you_mean(client):
    assert client.get("/search", params={"q": "pythn"}).json()["did_you_mean"] == "python"


def test_validation_errors(client):
    assert client.get("/search").status_code == 422                          # q missing
    assert client.get("/search", params={"q": ""}).status_code == 422        # empty
    assert client.get("/search", params={"q": "   "}).status_code == 422     # blank
    assert client.get("/search", params={"q": "x", "k": 51}).status_code == 422
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "a" * (MAX_QUERY_LEN + 1)}).status_code == 422


def test_k_at_limit_and_long_query_ok(client):
    assert client.get("/search", params={"q": "python", "k": 50}).status_code == 200
    assert client.get("/search", params={"q": "python " * 28}).status_code == 200   # 196 chars


def test_special_characters(client):
    for q in ['"', "<script>alert(1)</script>", "'; DROP TABLE x;--", "日本語", "AND OR NOT"]:
        r = client.get("/search", params={"q": q})
        assert r.status_code == 200, q
        assert r.json()["results"] == [] or isinstance(r.json()["results"], list)


def test_suggest(client):
    r = client.get("/suggest", params={"prefix": "py"})
    assert r.status_code == 200
    assert r.json()["suggestions"] == ["python"]
    assert client.get("/suggest", params={"prefix": "p"}).json()["suggestions"] == []
    assert client.get("/suggest").status_code == 422
    assert client.get("/suggest", params={"prefix": "py", "limit": 21}).status_code == 422


def test_home_page_is_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Mini Search Engine" in r.text


def test_503_if_service_not_loaded(index, docs):
    client = TestClient(create_app(service=None))          # no `with`: startup NOT run
    assert client.get("/health").status_code == 503
    assert client.get("/search", params={"q": "x"}).status_code == 503
