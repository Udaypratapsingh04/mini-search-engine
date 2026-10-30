# Ranking notes: TF-IDF vs BM25

Fill this in after running `python -m benchmarks.compare_ranking`.

## TF-IDF vs BM25
- Which model gave more sensible top results? Give 2 example queries.
- Did TF-IDF favour very long documents? Did BM25 fix that?

## Parameter sweep (k1, b)
| k1  | b    | avg top-5 overlap with default | Notes |
|-----|------|-------------------------------|-------|
| 1.2 | 0.5  |                               |       |
| 1.2 | 0.75 |                               |       |
| 1.2 | 0.9  |                               |       |
| 1.5 | 0.5  |                               |       |
| 1.5 | 0.75 | 5.00 (default)                |       |
| 1.5 | 0.9  |                               |       |
| 2.0 | 0.5  |                               |       |
| 2.0 | 0.75 |                               |       |
| 2.0 | 0.9  |                               |       |

## Conclusions
- b controls length normalisation: higher b penalises long docs more.
- k1 controls term-frequency saturation: higher k1 lets repeated terms count longer.
- Chosen defaults and why:

## Top-K complexity
- Full sort: O(n log n). heapq / min-heap: O(n log k).
- Paste the output of `python -m benchmarks.bench_topk` here.
