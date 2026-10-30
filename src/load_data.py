import re


def _split_header(raw: str):
    """Pull the Subject line out as the title; the rest is the body."""
    head, _, body = raw.partition("\n\n")
    match = re.search(r"^Subject:\s*(.+)$", head, re.MULTILINE)
    title = match.group(1).strip() if match else "(no title)"
    return title, body.strip()


def load_documents():
    """Return a list of {id, title, text} dicts."""
    # Imported here so the deployed web app does not need scikit-learn.
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="all",
        remove=("footers", "quotes"),
        data_home="data/",          # cache inside the gitignored data/ folder
    )
    docs = []
    for raw in data.data:
        title, text = _split_header(raw)
        if not text:                # skip empty documents
            continue
        docs.append({"id": len(docs), "title": title, "text": text})
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} docs")
    print(docs[0]["title"])
