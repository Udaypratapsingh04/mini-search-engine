import time
from src.index import InvertedIndex
from src.load_data import load_documents
from src.search import SearchEngine

INDEX_PATH = "data/index.pkl"


def main():
    print("Loading index...")
    index = InvertedIndex.load(INDEX_PATH)
    engine = SearchEngine(index, load_documents())
    print(f"Ready: {index.num_docs} docs. Examples:")
    print('  space shuttle | python AND NOT snake | "machine learning"')
    print("Type 'quit' to exit.\n")

    while True:
        query = input("search> ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        t0 = time.perf_counter()
        results = engine.search(query, k=10)
        ms = (time.perf_counter() - t0) * 1000
        for i, r in enumerate(results, 1):
            print(f"{i:2}. [{r.score:6.3f}] {r.title}")
            print(f"      {r.snippet[:100]}")
        print(f"-- {len(results)} results in {ms:.1f} ms\n")


if __name__ == "__main__":
    main()
