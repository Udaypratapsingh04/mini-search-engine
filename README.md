# Mini Search Engine

A search engine built from scratch in Python: inverted index, TF-IDF and BM25
ranking, boolean and phrase queries, autocomplete, and typo correction.

## Setup (Windows PowerShell)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('stopwords')"

## Build (once)

    python -m src.build_index     # inverted index -> data/index.pkl
    python -m src.vocab           # autocomplete vocabulary -> data/vocab.pkl

## Search from the terminal

    python -m src.cli

- Ranked (BM25): `space shuttle launch`
- Boolean (uppercase operators, no brackets): `python AND NOT snake`, `cat OR dog`
- Phrase: `"machine learning"` (the whole query inside quotes)
- Autocomplete: `suggest mach`
- Typos: `graphcis card` prints `Did you mean: graphics card`

## Run tests

    python -m pytest -v

## Benchmarks

    python -m benchmarks.bench_topk
    python -m benchmarks.compare_ranking
    python -m benchmarks.bench_suggest
    python -m benchmarks.typo_accuracy
    python -m benchmarks.bench          # writes docs/benchmark.md

## Known limitations

- Porter stemming over-stems (university and universe both become "univers").
- Stopwords are removed before indexing, so the phrase "state of the art"
  behaves like "state art".
- Boolean queries do not support brackets. Phrase queries must be the whole query.
- Typo correction does not fix a wrong FIRST letter, and only looks at words
  that appear in at least 2 documents.
