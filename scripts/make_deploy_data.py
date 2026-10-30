"""Build a small index that fits Render's free tier (512 MB RAM).

    python -m scripts.make_deploy_data                 # 6000 docs -> deploy_data/
    python -m scripts.make_deploy_data --max-docs 4000
"""
import argparse
import os
import pickle
import time

from src.build_index import build_index
from src.load_data import load_documents
from src.vocab import build_vocab, save_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=6000)
    ap.add_argument("--out", default="deploy_data")
    ap.add_argument("--snippet-chars", type=int, default=300)
    args = ap.parse_args()

    docs = load_documents()[: args.max_docs]
    print(f"Using {len(docs)} docs")

    t0 = time.perf_counter()
    index = build_index(docs)
    vocab = build_vocab(docs)
    print(f"Built index + vocab in {time.perf_counter() - t0:.1f}s "
          f"({index.vocab_size()} index terms, {len(vocab)} vocab words)")

    os.makedirs(args.out, exist_ok=True)
    index.save(os.path.join(args.out, "index.pkl"))
    save_vocab(vocab, os.path.join(args.out, "vocab.pkl"))
    small_docs = [
        {"id": d["id"], "title": d["title"], "text": d["text"][: args.snippet_chars]}
        for d in docs
    ]
    with open(os.path.join(args.out, "docs.pkl"), "wb") as f:
        pickle.dump(small_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    total = 0.0
    for name in ("index.pkl", "vocab.pkl", "docs.pkl"):
        mb = os.path.getsize(os.path.join(args.out, name)) / (1024 * 1024)
        total += mb
        print(f"  {name:10} {mb:6.1f} MB")
    print(f"  total      {total:6.1f} MB")
    if total > 50:
        print("WARNING: over 50 MB. GitHub warns above 50 MB; use a smaller --max-docs.")


if __name__ == "__main__":
    main()
