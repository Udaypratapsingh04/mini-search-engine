import os
import time

from src.autocomplete import Autocomplete
from src.index import InvertedIndex
from src.load_data import load_documents
from src.search import SearchEngine
from src.spellcheck import SpellChecker, did_you_mean
from src.vocab import VOCAB_PATH, load_vocab

INDEX_PATH = "data/index.pkl"


def main():
    for path, cmd in ((INDEX_PATH, "python -m src.build_index"),
                      (VOCAB_PATH, "python -m src.vocab")):
        if not os.path.exists(path):
            print(f"Missing {path}. Run:  {cmd}")
            return

    print("Loading...")
    index = InvertedIndex.load(INDEX_PATH)
    vocab = load_vocab()
    engine = SearchEngine(index, load_documents())
    auto = Autocomplete(vocab)
    checker = SpellChecker(vocab)

    print(f"Ready: {index.num_docs} docs, {len(vocab)} vocabulary words.")
    print('Try:  space shuttle | python AND NOT snake | "machine learning"')
    print("      suggest mach     (autocomplete)      quit")
    print("      graphcis card    (typo -> Did you mean)\n")

    while True:
        query = input("search> ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if query.lower().startswith("suggest "):
            t0 = time.perf_counter()
            words = auto.suggest(query[8:])
            ms = (time.perf_counter() - t0) * 1000
            print(f"  {words}  ({ms:.2f} ms)\n")
            continue

        t0 = time.perf_counter()
        results = engine.search(query, k=10)
        ms = (time.perf_counter() - t0) * 1000

        fix = did_you_mean(query, index, checker, engine.preprocess)
        if fix:
            print(f"Did you mean: {fix}")
        for i, r in enumerate(results, 1):
            print(f"{i:2}. [{r.score:6.3f}] {r.title}")
            print(f"      {r.snippet[:100]}")
        print(f"-- {len(results)} results in {ms:.1f} ms\n")


if __name__ == "__main__":
    main()
