"""Compare TF-IDF vs BM25 and sweep k1 / b. Needs data/index.pkl."""
from src.index import InvertedIndex
from src.load_data import load_documents
from src.search import SearchEngine

QUERIES = [
    "space shuttle launch",
    "graphics card driver",
    "baseball world series",
    "gun control law",
    "windows operating system",
    "encryption clipper chip",
    "israel palestine conflict",
    "hockey playoffs",
    "car engine oil",
    "medical treatment diet",
]


def top_ids(engine, query, **kw):
    return [r.doc_id for r in engine.search(query, k=5, **kw)]


if __name__ == "__main__":
    index = InvertedIndex.load("data/index.pkl")
    engine = SearchEngine(index, load_documents())

    print("=== TF-IDF vs BM25 (top 5 titles) ===")
    for q in QUERIES:
        print(f"\nQuery: {q}")
        for model in ("tfidf", "bm25"):
            print(f"  {model}:")
            for r in engine.search(q, k=5, model=model):
                print(f"    {r.score:7.3f}  {r.title[:70]}")

    print("\n=== Parameter sweep (overlap of top-5 with default k1=1.5, b=0.75) ===")
    default = {q: top_ids(engine, q) for q in QUERIES}
    print(f"{'k1':>5} {'b':>5} {'avg overlap /5':>15}")
    for k1 in (1.2, 1.5, 2.0):
        for b in (0.5, 0.75, 0.9):
            total = 0
            for q in QUERIES:
                ids = top_ids(engine, q, k1=k1, b=b)
                total += len(set(ids) & set(default[q]))
            print(f"{k1:>5} {b:>5} {total / len(QUERIES):>15.2f}")
