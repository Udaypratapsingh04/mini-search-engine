# Mini Search Engine

A search engine built from scratch in Python: inverted index, BM25 ranking,
boolean and phrase queries, autocomplete, and typo correction.

## Setup (Windows PowerShell)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('stopwords')"

## Build the index

    python -m src.build_index

## Run tests

    python -m pytest -v

## Known limitations

- Porter stemming over-stems (university and universe both become "univers").
