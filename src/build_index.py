import os
import time
from src.load_data import load_documents
from src.preprocess import preprocess
from src.index import InvertedIndex

INDEX_PATH = "data/index.pkl"


def build_index(docs) -> InvertedIndex:
    idx = InvertedIndex()
    for d in docs:
        idx.add_document(d["id"], preprocess(d["title"] + " " + d["text"]))
    return idx


if __name__ == "__main__":
    docs = load_documents()

    t0 = time.perf_counter()
    idx = build_index(docs)
    build_time = time.perf_counter() - t0

    print(f"Docs indexed:     {idx.num_docs}")
    print(f"Unique terms:     {idx.vocab_size()}")
    print(f"Avg doc length:   {idx.avg_doc_length:.1f}")
    print(f"Build time:       {build_time:.2f}s")

    os.makedirs("data", exist_ok=True)
    idx.save(INDEX_PATH)
    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)

    t0 = time.perf_counter()
    InvertedIndex.load(INDEX_PATH)
    load_time = time.perf_counter() - t0

    print(f"Index file size:  {size_mb:.1f} MB")
    print(f"Load time:        {load_time:.2f}s")
    print(f"Build vs load:    {build_time / load_time:.0f}x faster to load")
