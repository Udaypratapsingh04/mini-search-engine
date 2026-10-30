"""Day 15: latency benchmark. Writes docs/benchmark.md.

    python -m benchmarks.bench            # run benchmark
    python -m benchmarks.bench --profile  # also show cProfile hot spots
"""
import argparse
import cProfile
import math
import os
import platform
import pstats
import random
import statistics
import sys
import time

from src.autocomplete import Autocomplete
from src.index import InvertedIndex
from src.load_data import load_documents
from src.search import SearchEngine
from src.spellcheck import SpellChecker
from src.vocab import load_vocab

INDEX_PATH = "data/index.pkl"
N_QUERIES = 200


def percentile(sorted_vals, p):
    i = max(0, min(len(sorted_vals) - 1, math.ceil(p / 100 * len(sorted_vals)) - 1))
    return sorted_vals[i]


def time_calls(fn, items):
    """Run fn on each item, return sorted list of times in ms."""
    out = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        out.append((time.perf_counter() - t0) * 1000)
    return sorted(out)


def stats(times):
    return (statistics.mean(times), statistics.median(times),
            percentile(times, 95), percentile(times, 99))


def corrupt(word, rng):
    """Make a typo: swap two letters or drop one."""
    i = rng.randrange(1, len(word) - 1)
    if rng.random() < 0.5:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + word[i + 1:]


def main(profile=False):
    rng = random.Random(42)

    t0 = time.perf_counter()
    index = InvertedIndex.load(INDEX_PATH)
    index_load_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    vocab = load_vocab()
    vocab_load_ms = (time.perf_counter() - t0) * 1000

    engine = SearchEngine(index, load_documents())
    auto = Autocomplete(vocab)
    checker = SpellChecker(vocab)

    words = sorted(w for w in vocab if len(w) >= 5)
    queries = [" ".join(rng.sample(words, rng.randint(1, 3))) for _ in range(N_QUERIES)]
    prefixes = [rng.choice(words)[:rng.choice([2, 3])] for _ in range(N_QUERIES)]
    typos = [corrupt(rng.choice(words), rng) for _ in range(N_QUERIES)]

    for q in queries[:20]:                       # warm-up (stem cache etc.)
        engine.search(q)

    rows = [
        ("search (BM25, k=10)", stats(time_calls(lambda q: engine.search(q, 10), queries))),
        ("suggest (autocomplete)", stats(time_calls(auto.suggest, prefixes))),
        ("typo correction", stats(time_calls(checker.correct, typos))),
    ]

    lines = [
        "# Benchmark results", "",
        f"- Documents: {index.num_docs}",
        f"- Unique index terms: {index.vocab_size()}",
        f"- Vocabulary (autocomplete): {len(vocab)} words",
        f"- Queries per test: {N_QUERIES} (1-3 random terms)",
        f"- Machine: Python {platform.python_version()}, {platform.system()} {platform.machine()}",
        "", f"Index load time: {index_load_ms:.0f} ms | Vocab load time: {vocab_load_ms:.0f} ms", "",
        "| Operation | avg (ms) | median (ms) | p95 (ms) | p99 (ms) |",
        "|---|---|---|---|---|",
    ]
    for name, (avg, med, p95, p99) in rows:
        lines.append(f"| {name} | {avg:.3f} | {med:.3f} | {p95:.3f} | {p99:.3f} |")
    report = "\n".join(lines)
    print(report)

    os.makedirs("docs", exist_ok=True)
    with open("docs/benchmark.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nSaved docs/benchmark.md")

    if profile:
        print("\n=== cProfile: search loop (top 15 by cumulative time) ===")
        pr = cProfile.Profile()
        pr.enable()
        for q in queries:
            engine.search(q, 10)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main(profile="--profile" in sys.argv)
