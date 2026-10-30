# Benchmark results

- Documents: 17547
- Unique index terms: 96190
- Vocabulary (autocomplete): 42193 words
- Queries per test: 200 (1-3 random terms)
- Machine: Python 3.14.0, Windows AMD64

Index load time: 305 ms | Vocab load time: 7 ms

| Operation | avg (ms) | median (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| search (BM25, k=10) | 0.109 | 0.062 | 0.337 | 0.740 |
| suggest (autocomplete) | 0.001 | 0.001 | 0.002 | 0.004 |
| typo correction | 10.579 | 9.707 | 21.855 | 25.127 |
