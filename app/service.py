"""Search service: everything the API needs, with no web-framework code.

Kept separate from FastAPI so it is easy to test and reuse.
"""
import os
import pickle
import time

from src.autocomplete import Autocomplete
from src.index import InvertedIndex
from src.search import SearchEngine
from src.spellcheck import SpellChecker, did_you_mean
from src.vocab import load_vocab


def resolve_data_dir() -> str:
    """DATA_DIR env var > data/ (full local build) > deploy_data/ (bundled subset)."""
    env = os.environ.get("DATA_DIR")
    if env:
        return env
    if os.path.exists(os.path.join("data", "index.pkl")):
        return "data"
    return "deploy_data"


class SearchService:
    def __init__(self, index, docs, vocab, preprocess_fn=None):
        self.index = index
        self.vocab = vocab
        self.engine = SearchEngine(index, docs, preprocess_fn)
        self.autocomplete = Autocomplete(vocab)
        self.checker = SpellChecker(vocab)

    @classmethod
    def from_data_dir(cls, data_dir=None, preprocess_fn=None):
        data_dir = data_dir or resolve_data_dir()
        index_path = os.path.join(data_dir, "index.pkl")
        vocab_path = os.path.join(data_dir, "vocab.pkl")
        docs_path = os.path.join(data_dir, "docs.pkl")

        for path in (index_path, vocab_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Build it with: python -m src.build_index and "
                    f"python -m src.vocab (full dataset), or use the bundled "
                    f"deploy_data/ folder (python -m scripts.make_deploy_data)."
                )

        index = InvertedIndex.load(index_path)
        vocab = load_vocab(vocab_path)
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                docs = pickle.load(f)
        else:                                   # full local build: read the dataset
            from src.load_data import load_documents
            docs = load_documents()
        return cls(index, docs, vocab, preprocess_fn)

    # ---- API payloads ----
    def search(self, q: str, k: int = 10) -> dict:
        t0 = time.perf_counter()
        q = q.strip()
        results = self.engine.search(q, k=k)
        fix = did_you_mean(q, self.index, self.checker, self.engine.preprocess)
        took = (time.perf_counter() - t0) * 1000
        return {
            "query": q,
            "k": k,
            "count": len(results),
            "took_ms": round(took, 2),
            "did_you_mean": fix,
            "results": [
                {"doc_id": r.doc_id, "score": round(r.score, 4),
                 "title": r.title, "snippet": r.snippet}
                for r in results
            ],
        }

    def suggest(self, prefix: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()
        words = self.autocomplete.suggest(prefix, limit)
        took = (time.perf_counter() - t0) * 1000
        return {"prefix": prefix, "suggestions": words, "took_ms": round(took, 3)}

    def health(self) -> dict:
        return {"status": "ok", "docs": self.index.num_docs,
                "vocab_words": len(self.vocab)}
