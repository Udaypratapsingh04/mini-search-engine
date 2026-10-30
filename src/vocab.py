"""Autocomplete / spellcheck vocabulary: original lowercase words (NOT stemmed)
with their document frequency.

Run once after Week 1 setup:  python -m src.vocab
"""
import os
import pickle
from collections import Counter


VOCAB_PATH = "data/vocab.pkl"


def build_vocab(docs, min_df: int = 2) -> dict:
    """Return {word: document_frequency}. Keeps alphabetic words, len >= 2."""
    from src.tokenizer import tokenize      # lazy: loading a vocab needs no NLTK
    df = Counter()
    for d in docs:
        words = set(tokenize(d["title"] + " " + d["text"]))
        df.update(w for w in words if w.isalpha() and len(w) >= 2)
    return {w: c for w, c in df.items() if c >= min_df}


def save_vocab(vocab: dict, path: str = VOCAB_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_vocab(path: str = VOCAB_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import time
    from src.load_data import load_documents

    docs = load_documents()
    t0 = time.perf_counter()
    vocab = build_vocab(docs)
    print(f"Vocabulary: {len(vocab)} words (built in {time.perf_counter() - t0:.1f}s)")
    save_vocab(vocab)
    print(f"Saved to {VOCAB_PATH}")
    top = sorted(vocab.items(), key=lambda kv: -kv[1])[:10]
    print("Most common:", ", ".join(f"{w}({c})" for w, c in top))
