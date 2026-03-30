"""
retriever.py — FAISS-based Retriever Module

Loads the FAISS index and text chunks at import time, and exposes a
retrieve(query, k=3) function that returns the top-k matching text chunks.
"""

import json
import os
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────── Configuration ────────────────────────────
INDEX_PATH: str = "index.faiss"
CHUNKS_PATH: str = "chunks.json"
MODEL_NAME: str = "all-MiniLM-L6-v2"

# ──────────────────────────── Load at import time ─────────────────────
# Load FAISS index
if not os.path.isfile(INDEX_PATH):
    print(f"[ERROR] FAISS index not found at '{INDEX_PATH}'. Run ingest.py first.")
    sys.exit(1)

_index: faiss.IndexFlatIP = faiss.read_index(INDEX_PATH)
print(f"[INFO] Loaded FAISS index with {_index.ntotal} vectors.")

# Load chunks
if not os.path.isfile(CHUNKS_PATH):
    print(f"[ERROR] Chunks file not found at '{CHUNKS_PATH}'. Run ingest.py first.")
    sys.exit(1)

try:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        _chunks: list[str] = json.load(f)
    print(f"[INFO] Loaded {len(_chunks)} text chunks.")
except Exception as e:
    print(f"[ERROR] Failed to load chunks: {e}")
    sys.exit(1)

# Load embedding model
_model: SentenceTransformer = SentenceTransformer(MODEL_NAME)
print(f"[INFO] Embedding model '{MODEL_NAME}' loaded.")


# ──────────────────────────── Public API ──────────────────────────────
def retrieve(query: str, k: int = 3) -> list[str]:
    """
    Embed a query and search the FAISS index for the top-k matching chunks.

    Args:
        query: The user's question or search string.
        k: Number of top results to return.

    Returns:
        A list of the top-k matching text chunks.
    """
    # Encode the query
    query_embedding = _model.encode([query], convert_to_numpy=True)
    # L2 normalise to match the index
    faiss.normalize_L2(query_embedding)

    # Search
    distances, indices = _index.search(query_embedding.astype(np.float32), k)

    results: list[str] = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(_chunks):
            results.append(_chunks[idx])
            print(f"  [Match {i+1}] score={distances[0][i]:.4f}  (chunk #{idx})")
    return results


# ──────────────────────────── Main (test) ─────────────────────────────
if __name__ == "__main__":
    test_query = "What wavelengths does thermal infrared remote sensing use?"
    print(f"\n[TEST] Query: \"{test_query}\"\n")
    results = retrieve(test_query, k=3)

    print(f"\n[RESULT] Retrieved {len(results)} chunks:\n")
    for i, chunk in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(chunk[:300])  # Print first 300 chars for readability
        print()
