import random
import time
from src.topk import top_k_sorted, top_k_nlargest, top_k_minheap

K = 10
METHODS = [
    ("sorted()", top_k_sorted),
    ("heapq.nlargest", top_k_nlargest),
    ("manual min-heap", top_k_minheap),
]


def best_of(fn, scores, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(scores, K)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


if __name__ == "__main__":
    random.seed(42)
    print(f"Top-{K} selection time (ms, best of 3)")
    print(f"{'n':>10} | " + " | ".join(f"{name:>16}" for name, _ in METHODS))
    for n in (10_000, 100_000, 1_000_000):
        scores = {i: random.random() for i in range(n)}
        row = [best_of(fn, scores) for _, fn in METHODS]
        print(f"{n:>10} | " + " | ".join(f"{t:>16.2f}" for t in row))
