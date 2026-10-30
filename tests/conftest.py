import pytest
from src.index import InvertedIndex


def pp(text):
    """Simple stand-in for preprocess so tests do not need NLTK."""
    return text.lower().split()


DOCS = [
    {"id": 0, "title": "Python basics",  "text": "python is fun python rocks"},
    {"id": 1, "title": "Snakes",         "text": "snake is long and scary"},
    {"id": 2, "title": "Python vs snake","text": "python snake comparison"},
    {"id": 3, "title": "ML intro",       "text": "machine learning is fun"},
    {"id": 4, "title": "Learning machine","text": "learning machine order reversed"},
]


@pytest.fixture
def index():
    idx = InvertedIndex()
    for d in DOCS:
        idx.add_document(d["id"], pp(d["text"]))
    return idx


@pytest.fixture
def docs():
    return DOCS
