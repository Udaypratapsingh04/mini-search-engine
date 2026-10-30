import pickle
from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)   # term -> {doc_id: [positions]}
        self.doc_lengths = {}            # doc_id -> number of tokens
        self.num_docs = 0
        self.total_length = 0

    @property
    def avg_doc_length(self) -> float:
        return self.total_length / self.num_docs if self.num_docs else 0.0

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        if doc_id in self.doc_lengths:
            raise ValueError(f"doc_id {doc_id} already indexed")
        for pos, term in enumerate(tokens):
            self.index[term].setdefault(doc_id, []).append(pos)
        self.doc_lengths[doc_id] = len(tokens)
        self.num_docs += 1
        self.total_length += len(tokens)

    def postings(self, term: str) -> dict[int, int]:
        """doc_id -> term frequency."""
        return {d: len(p) for d, p in self.index.get(term, {}).items()}

    def positions(self, term: str) -> dict[int, list[int]]:
        """doc_id -> list of positions (needed for phrase search)."""
        return self.index.get(term, {})

    def doc_freq(self, term: str) -> int:
        return len(self.index.get(term, {}))

    def vocab_size(self) -> int:
        return len(self.index)

    # ---- Day 5: persistence ----
    def save(self, path: str) -> None:
        state = {
            "index": dict(self.index),
            "doc_lengths": self.doc_lengths,
            "num_docs": self.num_docs,
            "total_length": self.total_length,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "InvertedIndex":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.index = defaultdict(dict, state["index"])
        obj.doc_lengths = state["doc_lengths"]
        obj.num_docs = state["num_docs"]
        obj.total_length = state["total_length"]
        return obj
