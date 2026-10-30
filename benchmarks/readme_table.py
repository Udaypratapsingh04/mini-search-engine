"""Print the benchmark table for the README.

    python -m benchmarks.readme_table --build-time 142.3

Use the "Build time" number printed by `python -m src.build_index`.
Needs data/index.pkl and data/vocab.pkl (run src.build_index and src.vocab first).
"""
import argparse
import os
import random
import statistics
import time

from benchmarks.bench import percentile, time_calls
from src.index import InvertedIndex
from src.load_data import load_documents
from src.search import SearchEngine
from src.vocab import load_vocab

INDEX_PATH = "data/index.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-time", type=float, default=None,
                    help="seconds, from the output of python -m src.build_index")
    args = ap.parse_args()

    t0 = time.perf_counter()
    index = InvertedIndex.load(INDEX_PATH)
    load_s = time.perf_counter() - t0
    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)

    engine = SearchEngine(index, load_documents())
    words = sorted(w for w in load_vocab() if len(w) >= 5)
    rng = random.Random(42)
    queries = [" ".join(rng.sample(words, rng.randint(1, 3))) for _ in range(200)]
    for q in queries[:20]:                       # warm-up
        engine.search(q)
    times = time_calls(lambda q: engine.search(q, 10), queries)

    build = f"{args.build_time:.1f} s" if args.build_time is not None else "(pass --build-time)"
    table = "\n".join([
        "| Metric | Value |",
        "|---|---|",
        f"| Documents | {index.num_docs:,} |",
        f"| Unique terms | {index.vocab_size():,} |",
        f"| Index size | {size_mb:.1f} MB |",
        f"| Build time | {build} |",
        f"| Index load time | {load_s:.2f} s |",
        f"| Avg search latency | {statistics.mean(times):.2f} ms |",
        f"| p95 search latency | {percentile(times, 95):.2f} ms |",
    ])
    print(table)
    os.makedirs("docs", exist_ok=True)
    with open("docs/benchmark_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nSaved docs/benchmark_table.md. Paste it into README.md.")


if __name__ == "__main__":
    main()
