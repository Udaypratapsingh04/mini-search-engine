"""Day 12: autocomplete latency with and without the per-node top-k cache."""
import random
import statistics
import time

from src.trie import Trie
from src.vocab import load_vocab


def measure(trie, prefixes, use_cache):
    times = []
    for p in prefixes:
        t0 = time.perf_counter()
        trie.autocomplete(p, 5, use_cache=use_cache)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return statistics.mean(times), times[int(0.95 * len(times)) - 1]


if __name__ == "__main__":
    random.seed(42)
    vocab = load_vocab()
    words = [w for w in vocab if len(w) >= 3]

    trie = Trie()
    t0 = time.perf_counter()
    for w, f in vocab.items():
        trie.insert(w, f)
    print(f"Trie built: {trie.size} words in {time.perf_counter() - t0:.2f}s")
    t0 = time.perf_counter()
    trie.build_cache(5)
    print(f"Cache built in {time.perf_counter() - t0:.2f}s\n")

    print("Latency in ms (200 random prefixes per length)")
    print(f"{'prefix len':>10} | {'no cache avg':>12} {'p95':>8} | {'cache avg':>10} {'p95':>8}")
    for n in (1, 2, 3):
        prefixes = [random.choice(words)[:n] for _ in range(200)]
        a1, p1 = measure(trie, prefixes, use_cache=False)
        a2, p2 = measure(trie, prefixes, use_cache=True)
        print(f"{n:>10} | {a1:>12.4f} {p1:>8.4f} | {a2:>10.4f} {p2:>8.4f}")
    print("\nExamples:", trie.autocomplete("mach"), trie.autocomplete("co"))
