import time
from src.load_data import load_documents
from src.preprocess import preprocess

docs = load_documents()
start = time.perf_counter()
total_tokens = 0
for d in docs:
    total_tokens += len(preprocess(d["title"] + " " + d["text"]))
elapsed = time.perf_counter() - start

print(f"Docs: {len(docs)}")
print(f"Tokens: {total_tokens}")
print(f"Preprocessing time: {elapsed:.2f}s")
