# Mini Search Engine

A search engine built from scratch in Python: inverted index, TF-IDF and BM25
ranking, boolean and phrase queries, autocomplete, and typo correction.

## Setup (Windows PowerShell)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('stopwords')"

## Build the index

    python -m src.build_index

## Search from the terminal

    python -m src.cli

Query types:

- Ranked (BM25): `space shuttle launch`
- Boolean (uppercase operators, no brackets): `python AND NOT snake`, `cat OR dog`
- Phrase: `"machine learning"` (the whole query inside quotes)

## Run tests

    python -m pytest -v

## Benchmarks

    python -m benchmarks.bench_topk
    python -m benchmarks.compare_ranking

## Known limitations

- Porter stemming over-stems (university and universe both become "univers").
- Stopwords are removed before indexing, so the phrase "state of the art"
  behaves like "state art".
- Boolean queries do not support brackets. Phrase queries must be the whole query.
