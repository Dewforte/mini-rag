"""
ingest.py — PDF Text Extraction, Chunking, Embedding, and FAISS Index Building

This script:
1. Opens remote_sensing.pdf with PyMuPDF and extracts all text.
2. Splits text into overlapping chunks (500 chars, 100 overlap).
3. Embeds chunks using the all-MiniLM-L6-v2 sentence-transformer model.
4. Builds a FAISS IndexFlatIP index (with L2 normalization).
5. Saves index.faiss and chunks.json.
"""

import json
import os
import sys

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────── Configuration ────────────────────────────
PDF_PATH: str = "remote_sensing.pdf"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
MODEL_NAME: str = "all-MiniLM-L6-v2"
INDEX_PATH: str = "index.faiss"
CHUNKS_PATH: str = "chunks.json"


# ──────────────────────────── Functions ────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Failed to open PDF: {e}")
        sys.exit(1)

    text = ""
    num_pages = len(doc)
    for page_num in range(num_pages):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()
    print(f"[INFO] Extracted text from {num_pages} pages ({len(text)} characters).")
    return text


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The full text to split.
        chunk_size: Number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    print(f"[INFO] Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks


def embed_chunks(chunks: list[str], model_name: str) -> np.ndarray:
    """
    Embed text chunks using a sentence-transformers model and L2-normalise.

    Args:
        chunks: List of text strings.
        model_name: HuggingFace model identifier.

    Returns:
        NumPy array of L2-normalised embeddings (N x D).
    """
    print(f"[INFO] Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("[INFO] Encoding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalisation so that inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP from L2-normalised embeddings.

    Args:
        embeddings: NumPy array of shape (N, D).

    Returns:
        A populated FAISS IndexFlatIP index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    print(f"[INFO] FAISS index built with {index.ntotal} vectors (dim={dimension}).")
    return index


def save_index_and_chunks(
    index: faiss.IndexFlatIP,
    chunks: list[str],
    index_path: str,
    chunks_path: str,
) -> None:
    """
    Save the FAISS index and text chunks to disk.

    Args:
        index: The FAISS index object.
        chunks: The list of text chunks.
        index_path: File path for the FAISS index.
        chunks_path: File path for the chunks JSON.
    """
    try:
        faiss.write_index(index, index_path)
        print(f"[INFO] FAISS index saved to '{index_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed to save FAISS index: {e}")
        sys.exit(1)

    try:
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Chunks saved to '{chunks_path}'.")
    except Exception as e:
        print(f"[ERROR] Failed to save chunks: {e}")
        sys.exit(1)


# ──────────────────────────── Main ─────────────────────────────────────
if __name__ == "__main__":
    # Ensure the PDF exists
    if not os.path.isfile(PDF_PATH):
        print(f"[ERROR] PDF not found at '{PDF_PATH}'. Please download it first.")
        sys.exit(1)

    # Pipeline
    raw_text = extract_text_from_pdf(PDF_PATH)
    chunks = split_into_chunks(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = embed_chunks(chunks, MODEL_NAME)
    index = build_faiss_index(embeddings)
    save_index_and_chunks(index, chunks, INDEX_PATH, CHUNKS_PATH)

    print("\n[DONE] Ingestion complete. Files created:")
    print(f"  - {INDEX_PATH}")
    print(f"  - {CHUNKS_PATH}")
