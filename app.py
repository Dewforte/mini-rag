"""
app.py — Dynamic RAG Web Application with Dual Output Comparison
Framework : Streamlit
LLM backend: Google Gemini API  (https://ai.google.dev)
"""

import base64
import io
import os
import re

import faiss
import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# ──────────────────────────── Configuration ────────────────────────────
load_dotenv()

GEMINI_API_KEY : str = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     : str = "gemini-2.5-flash"
EMBED_MODEL    : str = "all-MiniLM-L6-v2"
CHUNK_SIZE     : int = 500
CHUNK_OVERLAP  : int = 100
TOP_K          : int = 5

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Demo",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────── Global CSS ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════
   ANIMATED BACKGROUND
═══════════════════════════════════════ */

html, body, .stApp {
    font-family: 'Rajdhani', sans-serif !important;
    background-color: #020c1b !important;
    color: #ccd6f6 !important;
    overflow-x: hidden;
}

/* Dot-grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(0,229,255,0.09) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* Animated floating orbs */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 600px 400px at 15% 40%, rgba(0,229,255,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 500px 350px at 85% 20%, rgba(139,0,255,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 400px 300px at 60% 80%, rgba(0,200,150,0.05) 0%, transparent 70%);
    animation: orb-drift 18s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes orb-drift {
    0%   { transform: translate(0px, 0px) scale(1); }
    33%  { transform: translate(30px, -20px) scale(1.04); }
    66%  { transform: translate(-20px, 30px) scale(0.97); }
    100% { transform: translate(10px, -10px) scale(1.02); }
}

/* Scan-line sweep */
@keyframes scanline {
    0%   { top: -4px; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100vh; opacity: 0; }
}

.scanline {
    position: fixed;
    left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(0,229,255,0.15) 20%,
        rgba(0,229,255,0.35) 50%,
        rgba(0,229,255,0.15) 80%,
        transparent 100%);
    animation: scanline 7s linear infinite;
    pointer-events: none;
    z-index: 9999;
}

/* Corner bracket decorations */
@keyframes bracket-pulse {
    0%, 100% { opacity: 0.3; }
    50%       { opacity: 0.8; }
}
.corner-tl, .corner-tr, .corner-bl, .corner-br {
    position: fixed;
    width: 28px; height: 28px;
    pointer-events: none;
    z-index: 100;
    animation: bracket-pulse 3s ease-in-out infinite;
}
.corner-tl { top: 12px;  left: 12px;  border-top: 2px solid #00e5ff; border-left: 2px solid #00e5ff; }
.corner-tr { top: 12px;  right: 12px; border-top: 2px solid #00e5ff; border-right: 2px solid #00e5ff; }
.corner-bl { bottom: 12px; left: 12px;  border-bottom: 2px solid #00e5ff; border-left: 2px solid #00e5ff; }
.corner-br { bottom: 12px; right: 12px; border-bottom: 2px solid #00e5ff; border-right: 2px solid #00e5ff; }

/* ═══════════════════════════════════════
   STREAMLIT CHROME REMOVAL
═══════════════════════════════════════ */
#MainMenu, footer                 { visibility: hidden !important; }
.stDeployButton                   { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }
[data-testid="stDecoration"]      { display: none !important; }
header[data-testid="stHeader"]    { background: transparent !important; }

/* ═══════════════════════════════════════
   LAYOUT
═══════════════════════════════════════ */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    position: relative;
    z-index: 1;
}
p, span, div, label { color: inherit; }

/* ═══════════════════════════════════════
   SIDEBAR COLUMN
═══════════════════════════════════════ */
[data-testid="column"]:first-child > div {
    background: rgba(2, 12, 30, 0.92) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    border-radius: 14px !important;
    padding: 1.4rem 1.1rem !important;
    min-height: 84vh;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 30px rgba(0,229,255,0.06), inset 0 1px 0 rgba(0,229,255,0.08);
    position: relative;
    overflow: hidden;
}

/* Sidebar top glow line */
[data-testid="column"]:first-child > div::before {
    content: '';
    position: absolute;
    top: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00e5ff, transparent);
    opacity: 0.5;
}

[data-testid="column"]:first-child label,
[data-testid="column"]:first-child p,
[data-testid="column"]:first-child span,
[data-testid="column"]:first-child small {
    color: #8892b0 !important;
}

/* ── File upload bounding box ── */
[data-testid="column"]:first-child [data-testid="stFileUploader"] {
    border: 1.5px solid rgba(0,229,255,0.4) !important;
    border-radius: 10px !important;
    padding: 10px 10px 6px !important;
    background: rgba(0,229,255,0.03) !important;
    box-shadow: 0 0 14px rgba(0,229,255,0.08) !important;
}

[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] {
    background: rgba(0,8,20,0.8) !important;
    border: 1.5px dashed rgba(0,229,255,0.3) !important;
    border-radius: 7px !important;
    padding: 8px 12px !important;
    min-height: unset !important;
}
[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] span,
[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] p {
    color: #00e5ff !important;
    font-size: 12px !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] svg {
    color: rgba(0,229,255,0.5) !important;
    width: 18px !important; height: 18px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small { display: none !important; }

[data-testid="column"]:first-child [data-testid="stFileUploader"] label p {
    color: #00e5ff !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Spinner */
[data-testid="stSpinner"] p {
    color: #00e5ff !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* ═══════════════════════════════════════
   MAIN COLUMN
═══════════════════════════════════════ */
[data-testid="column"]:last-child > div {
    background: rgba(2, 12, 30, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(8px);
}

/* ═══════════════════════════════════════
   TABS
═══════════════════════════════════════ */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(0,229,255,0.15) !important;
    gap: 4px;
    margin-bottom: 16px;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 9px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    color: #2a4a6a !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 18px !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #4a8aaa !important;
    border-bottom: 2px solid rgba(0,229,255,0.3) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════
   QUERY INPUT
═══════════════════════════════════════ */
input[type="text"], textarea {
    background: rgba(0,10,25,0.9) !important;
    color: #ccd6f6 !important;
    border: 1.5px solid rgba(0,229,255,0.3) !important;
    border-radius: 8px !important;
    caret-color: #00e5ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
input::placeholder, textarea::placeholder {
    color: rgba(136,146,176,0.6) !important;
}
input:focus, textarea:focus {
    border-color: #00e5ff !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.1), 0 0 18px rgba(0,229,255,0.12) !important;
    background: rgba(0,14,32,0.95) !important;
}

/* ═══════════════════════════════════════
   ASK BUTTON
═══════════════════════════════════════ */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #003d5c 0%, #006b8a 100%) !important;
    color: #00e5ff !important;
    border: 1.5px solid rgba(0,229,255,0.5) !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    height: 38px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 14px rgba(0,229,255,0.2) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #005070 0%, #008aad 100%) !important;
    box-shadow: 0 0 28px rgba(0,229,255,0.45), 0 0 60px rgba(0,229,255,0.15) !important;
    border-color: #00e5ff !important;
    transform: translateY(-1px) !important;
}

/* ═══════════════════════════════════════
   SIDEBAR CLEAR BUTTON
═══════════════════════════════════════ */
[data-testid="column"]:first-child [data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    color: #4a6a8a !important;
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
}
[data-testid="column"]:first-child [data-testid="stButton"] button:hover {
    background: rgba(0,229,255,0.06) !important;
    border-color: rgba(0,229,255,0.5) !important;
    color: #00e5ff !important;
    box-shadow: 0 0 10px rgba(0,229,255,0.1) !important;
}

/* ═══════════════════════════════════════
   EXPANDER
═══════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,229,255,0.15) !important;
    border-radius: 8px !important;
    background: rgba(0,8,20,0.6) !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #00e5ff !important;
    font-weight: 600 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.5px !important;
    padding: 8px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    color: #7affff !important;
}

/* ═══════════════════════════════════════
   DIVIDER
═══════════════════════════════════════ */
[data-testid="stDivider"] {
    border-color: rgba(0,229,255,0.15) !important;
    margin: 10px 0 !important;
}

/* ═══════════════════════════════════════
   ALERTS
═══════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: 'Rajdhani', sans-serif !important;
    background: rgba(0,8,20,0.8) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
}

/* ═══════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════ */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00e5ff, #8b00ff) !important;
    border-radius: 4px !important;
}

/* ═══════════════════════════════════════
   RESPONSE CARD ANIMATIONS
═══════════════════════════════════════ */
@keyframes card-glow-cyan {
    0%, 100% { box-shadow: 0 0 16px rgba(0,229,255,0.12), 0 0 0 1px rgba(0,229,255,0.15); }
    50%       { box-shadow: 0 0 28px rgba(0,229,255,0.22), 0 0 0 1px rgba(0,229,255,0.3); }
}
@keyframes card-glow-violet {
    0%, 100% { box-shadow: 0 0 16px rgba(139,0,255,0.12), 0 0 0 1px rgba(139,0,255,0.15); }
    50%       { box-shadow: 0 0 28px rgba(139,0,255,0.22), 0 0 0 1px rgba(139,0,255,0.3); }
}
.card-rag { animation: card-glow-cyan   3s ease-in-out infinite; }
.card-llm { animation: card-glow-violet 3s ease-in-out infinite; }

</style>

<!-- Animated overlay elements -->
<div class="scanline"></div>
<div class="corner-tl"></div>
<div class="corner-tr"></div>
<div class="corner-bl"></div>
<div class="corner-br"></div>
""", unsafe_allow_html=True)


# ──────────────────────────── Model loading (cached) ───────────────────
@st.cache_resource(show_spinner="Initialising neural engine…")
def load_models():
    embed  = SentenceTransformer(EMBED_MODEL)
    llm    = genai.Client(api_key=GEMINI_API_KEY)
    return embed, llm

embed_model, client = load_models()


# ──────────────────────────── Session state ────────────────────────────
_defaults = {
    "faiss_index"      : None,
    "chunks"           : [],
    "chunk_counts"     : {},
    "uploaded_files"   : [],
    "raw_texts"        : {},       # filename -> raw extracted text
    "rag_ans"          : None,
    "raw_ans"          : None,
    "retrieved_chunks" : [],
    "eval_results"     : None,     # dict of config -> metrics
    "eval_queries"     : [],       # auto-generated test queries
    "eval_chart_png"   : None,     # bytes of rendered chart
    "eval_detail"      : [],       # per-query judgment table
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════
#  BACKEND — INGESTION & RETRIEVAL
# ══════════════════════════════════════════════════════════════════════

def _ingest_file(uploaded_file) -> tuple[list[str], int, str]:
    """Extract text, chunk by characters, return (chunks, count, raw_text)."""
    data = uploaded_file.read()
    doc  = fitz.open(stream=data, filetype="pdf")
    raw_text = "".join(page.get_text() for page in doc)
    doc.close()
    if not raw_text.strip():
        raise ValueError("PDF appears empty or unreadable.")
    chunks, n = _chunk_chars(raw_text)
    return chunks, n, raw_text


def _chunk_chars(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Fixed-size character-based chunking (main app pathway)."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks, len(chunks)


def _chunk_by_tokens(text: str, tokenizer, chunk_tokens: int = 200, overlap_tokens: int = 40) -> list[str]:
    """Token-based chunking using the embedding model's own tokenizer.

    Note: all-MiniLM-L6-v2 truncates at 256 tokens — so configs ≥256 will
    have their embeddings silently truncated by the model, which is itself
    an important trade-off to observe in evaluation.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, start = [], 0
    while start < len(token_ids):
        end = min(start + chunk_tokens, len(token_ids))
        chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += chunk_tokens - overlap_tokens
    return chunks


def _rebuild_index(all_chunks: list[str]) -> faiss.IndexFlatIP:
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def _build_temp_index(chunks: list[str]):
    """Build an ephemeral FAISS index for evaluation (not stored in session_state)."""
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index, chunks


def answer_rag(question: str) -> tuple[str, list[str]]:
    if st.session_state.faiss_index is None or not st.session_state.chunks:
        return "Upload and process a PDF document first.", []
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    _, idxs = st.session_state.faiss_index.search(q_emb.astype(np.float32), TOP_K)
    retrieved = [
        st.session_state.chunks[i]
        for i in idxs[0]
        if 0 <= i < len(st.session_state.chunks)
    ]
    context = "\n\n---\n\n".join(f"Passage {i+1}:\n{c}" for i, c in enumerate(retrieved))
    system = (
        "You are a precise technical assistant. "
        "Answer ONLY using the provided passages. "
        "If the passages do not contain the answer, reply EXACTLY: "
        "'The document does not cover this topic.' "
        "Do not use outside knowledge."
    )
    try:
        r = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Context:\n{context}\n\nQuestion: {question}",
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                max_output_tokens=2000,
            ),
        )
        return r.text.strip(), retrieved
    except Exception as e:
        return f"[ERROR] RAG call failed: {e}", retrieved


def get_llm_answer_bare(question: str) -> str:
    try:
        r = client.models.generate_content(
            model=MODEL_NAME,
            contents=question,
            config=types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=2000,
            ),
        )
        return r.text.strip()
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"


# ══════════════════════════════════════════════════════════════════════
#  BACKEND — EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _generate_test_queries(raw_text: str, n: int = 8) -> list[str]:
    """Ask Gemini to synthesise factual test queries from document content."""
    sample = raw_text[:4000]
    prompt = (
        f"From the document excerpt below, write exactly {n} specific factual questions "
        f"whose answers are clearly stated in the text. "
        f"Output only a numbered list (1. … {n}. …). No preamble, no answers.\n\n"
        f"Document:\n{sample}"
    )
    r = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=600),
    )
    queries = []
    for line in r.text.strip().split("\n"):
        line = line.strip()
        q = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
        if len(q) > 12:
            queries.append(q)
    return queries[:n]


def _llm_judge(query: str, retrieved_chunks: list[str]) -> bool:
    """LLM-as-judge: does the retrieved context contain the answer? Returns True/False."""
    context = "\n\n".join(
        f"[Chunk {i+1}]: {c[:400]}" for i, c in enumerate(retrieved_chunks)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Do the retrieved chunks contain sufficient information to answer the query?\n"
        "Reply with exactly one word: YES or NO."
    )
    try:
        r = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
        )
        return r.text.strip().upper().startswith("Y")
    except Exception:
        return False


def _run_evaluation(raw_text: str, k: int = 5) -> tuple[dict, list[str], list[dict]]:
    """Full evaluation: 3 token-based chunk configs × N queries.

    Returns:
        results   – {config_name: {hit_rate, n_chunks, hits, total}}
        queries   – list of generated questions
        detail    – per-query table rows
    """
    tokenizer = embed_model.tokenizer

    eval_configs = [
        {"label": "200 tokens",  "tokens": 200,  "overlap": 40},
        {"label": "500 tokens",  "tokens": 500,  "overlap": 100},
        {"label": "1000 tokens", "tokens": 1000, "overlap": 200},
    ]

    # Generate queries once; reuse across all configs
    queries = _generate_test_queries(raw_text, n=8)

    results = {}
    detail  = []

    for cfg in eval_configs:
        chunks = _chunk_by_tokens(raw_text, tokenizer, cfg["tokens"], cfg["overlap"])
        index, _ = _build_temp_index(chunks)
        hits = 0
        cfg_detail = []

        for q in queries:
            q_emb = embed_model.encode([q], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            _, idxs = index.search(q_emb.astype(np.float32), k)
            retrieved = [chunks[i] for i in idxs[0] if 0 <= i < len(chunks)]
            verdict = _llm_judge(q, retrieved)
            if verdict:
                hits += 1
            cfg_detail.append({
                "config": cfg["label"],
                "query": q[:80] + ("…" if len(q) > 80 else ""),
                "hit": "✓" if verdict else "✗",
            })

        hit_rate = hits / len(queries) if queries else 0.0
        results[cfg["label"]] = {
            "hit_rate": hit_rate,
            "n_chunks": len(chunks),
            "hits": hits,
            "total": len(queries),
        }
        detail.extend(cfg_detail)

    return results, queries, detail


def _render_eval_chart(results: dict) -> bytes:
    """Render a dark-themed matplotlib bar chart; return PNG bytes."""
    BG   = "#020c1b"
    GRID = "#0a1e38"
    COLS = ["#00e5ff", "#8b5cf6", "#00c896"]
    TEXT = "#8892b0"

    labels    = list(results.keys())
    hit_rates = [results[c]["hit_rate"] * 100 for c in labels]
    n_chunks  = [results[c]["n_chunks"]        for c in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)

    # ── Left: Hit Rate bar chart ──────────────────────────────────────
    ax1.set_facecolor(BG)
    bars = ax1.bar(labels, hit_rates, color=COLS, width=0.45,
                   edgecolor="white", linewidth=0.3, zorder=3)

    # Target line
    ax1.axhline(y=85, color="#ff6b6b", linestyle="--", linewidth=1.4,
                alpha=0.75, zorder=4, label="Target HR@5 ≥ 85%")

    for bar, val in zip(bars, hit_rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.0f}%",
            ha="center", va="bottom",
            color="white", fontsize=12, fontweight="bold",
        )

    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Hit Rate @5 (%)", color=TEXT, fontsize=10)
    ax1.set_title("Retrieval Quality by Chunk Size", color="#ccd6f6",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.tick_params(colors=TEXT, labelsize=9)
    ax1.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ax1.spines.values():
        sp.set_edgecolor(GRID)
    ax1.legend(facecolor=BG, labelcolor=TEXT, fontsize=8,
               framealpha=0.8, edgecolor=GRID)

    # ── Right: Chunk count bar chart ──────────────────────────────────
    ax2.set_facecolor(BG)
    bars2 = ax2.bar(labels, n_chunks, color=COLS, width=0.45,
                    edgecolor="white", linewidth=0.3, zorder=3, alpha=0.85)

    for bar, val in zip(bars2, n_chunks):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(n_chunks) * 0.02,
            str(val),
            ha="center", va="bottom",
            color="white", fontsize=11, fontweight="bold",
        )

    ax2.set_ylabel("Number of Chunks", color=TEXT, fontsize=10)
    ax2.set_title("Index Size by Chunk Config", color="#ccd6f6",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.tick_params(colors=TEXT, labelsize=9)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ax2.spines.values():
        sp.set_edgecolor(GRID)

    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ──────────────────────────── Logo helper ──────────────────────────────

def _logo_html(height: int = 40) -> str:
    for ext in ("png", "jpg", "jpeg", "webp", "svg"):
        for name in (f"logo.{ext}", f"Logo.{ext}"):
            if os.path.isfile(name):
                if ext == "svg":
                    try:
                        return open(name).read().replace("<svg", f'<svg height="{height}"', 1)
                    except Exception:
                        pass
                else:
                    try:
                        with open(name, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
                        return (
                            f'<img src="data:image/{mime};base64,{b64}" height="{height}" '
                            f'style="border-radius:8px;object-fit:contain;display:block;">'
                        )
                    except Exception:
                        pass
    return ""


# ══════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════
col_sidebar, col_main = st.columns([1, 3], gap="large")


# ════════════════════════════════════════════════
#  LEFT — SIDEBAR
# ════════════════════════════════════════════════
with col_sidebar:

    # ── Logo block ──
    logo = _logo_html(height=54)
    if logo:
        st.markdown(f"""
        <div style="
            display:flex; align-items:center; gap:12px;
            padding:12px 14px;
            background:rgba(0,229,255,0.04);
            border:1.5px solid rgba(0,229,255,0.25);
            border-radius:12px;
            margin-bottom:16px;
            box-shadow: 0 0 20px rgba(0,229,255,0.1);
        ">
            <div style="
                padding:4px;
                border:1.5px solid rgba(0,229,255,0.35);
                border-radius:9px;
                background:rgba(0,229,255,0.06);
                box-shadow:0 0 12px rgba(0,229,255,0.2);
            ">{logo}</div>
            <div>
                <div style="
                    font-family:'Orbitron',monospace;
                    font-size:11px; font-weight:700;
                    color:#00e5ff; letter-spacing:1.5px;
                    text-transform:uppercase;
                ">RAG System</div>
                <div style="
                    font-size:10px; color:#4a6a8a;
                    letter-spacing:0.5px; margin-top:2px;
                ">Knowledge Engine v2.0</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            padding:14px 16px;
            background:rgba(0,229,255,0.04);
            border:1.5px solid rgba(0,229,255,0.25);
            border-radius:12px;
            margin-bottom:16px;
            box-shadow:0 0 20px rgba(0,229,255,0.1);
        ">
            <div style="
                font-family:'Orbitron',monospace;
                font-size:13px; font-weight:700;
                color:#00e5ff; letter-spacing:2px;
            ">◎ RAG SYSTEM</div>
            <div style="font-size:10px;color:#4a6a8a;margin-top:3px;letter-spacing:0.5px;">
                Knowledge Engine v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <p style="
        font-family:'Orbitron',monospace;
        color:#00e5ff; font-size:10px; font-weight:700;
        letter-spacing:2.5px; text-transform:uppercase;
        margin:0 0 2px 0;
    ">Knowledge Sources</p>
    <p style="color:#7a9aba;font-size:11px;margin:0 0 10px 0;
              font-family:'Rajdhani',sans-serif;letter-spacing:0.3px;">
        PDFs ingested for retrieval
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    # ── File uploader ──
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    new_files = [
        f for f in (uploaded_files or [])
        if f.name not in st.session_state.chunk_counts
    ]

    if new_files:
        with st.spinner("Indexing documents…"):
            errors = []
            for f in new_files:
                try:
                    chunks, n, raw_text = _ingest_file(f)
                    st.session_state.chunks.extend(chunks)
                    st.session_state.chunk_counts[f.name] = n
                    st.session_state.uploaded_files.append(f.name)
                    st.session_state.raw_texts[f.name] = raw_text
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
            if st.session_state.chunks:
                st.session_state.faiss_index = _rebuild_index(st.session_state.chunks)
                # Invalidate cached eval when corpus changes
                st.session_state.eval_results  = None
                st.session_state.eval_queries   = []
                st.session_state.eval_chart_png = None
                st.session_state.eval_detail    = []
            for err in errors:
                st.error(err, icon="⚠️")

    # ── Loaded docs list ──
    st.markdown("""
    <p style="
        font-family:'Orbitron',monospace;
        color:#00e5ff; font-size:9px; font-weight:700;
        letter-spacing:2.5px; text-transform:uppercase;
        margin:14px 0 8px 0;
    ">Indexed Files</p>
    """, unsafe_allow_html=True)

    if st.session_state.uploaded_files:
        for fname in st.session_state.uploaded_files:
            n  = st.session_state.chunk_counts.get(fname, 0)
            dn = fname if len(fname) <= 26 else fname[:23] + "…"
            st.markdown(f"""
            <div style="
                background:rgba(0,8,20,0.7);
                border:1px solid rgba(0,229,255,0.15);
                border-left:3px solid #00e5ff;
                border-radius:8px;
                padding:8px 12px;
                margin-bottom:6px;
                display:flex; align-items:center; gap:10px;
            ">
                <div style="
                    background:#7f1d1d;
                    color:#fca5a5;
                    font-size:8px; font-weight:700;
                    padding:2px 5px; border-radius:4px;
                    font-family:'Orbitron',monospace;
                    letter-spacing:0.5px; flex-shrink:0;
                ">PDF</div>
                <div style="flex:1;min-width:0;">
                    <div style="
                        color:#ccd6f6; font-size:12px;
                        font-family:'Rajdhani',sans-serif;
                        font-weight:600;
                        white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                    ">{dn}</div>
                    <div style="
                        color:#00e5ff; font-size:10px;
                        font-family:'Orbitron',monospace;
                        opacity:0.7; letter-spacing:0.3px;
                    ">{n} chunks</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            border:1px dashed rgba(0,229,255,0.1);
            border-radius:8px; padding:14px;
            text-align:center;
        ">
            <div style="color:#3a6a9a;font-size:20px;margin-bottom:4px;">⬡</div>
            <div style="color:#3a6a9a;font-size:11px;
                        font-family:'Rajdhani',sans-serif;letter-spacing:0.3px;">
                No documents loaded
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    if st.button("⌫  Clear all documents", key="clear_btn", use_container_width=True):
        for k in _defaults:
            st.session_state[k] = _defaults[k]
        st.rerun()

    # ── System status ──
    status_color  = "#00e5ff" if st.session_state.faiss_index else "#1e3a5f"
    status_label  = "ONLINE" if st.session_state.faiss_index else "STANDBY"
    doc_count     = len(st.session_state.uploaded_files)
    chunk_total   = len(st.session_state.chunks)
    st.markdown(f"""
    <div style="margin-top:20px; border-top:1px solid rgba(0,229,255,0.1); padding-top:14px;">
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:10px;">
            <div style="
                width:7px;height:7px;border-radius:50%;
                background:{status_color};
                box-shadow:0 0 8px {status_color};
            "></div>
            <span style="
                font-family:'Orbitron',monospace;
                font-size:9px;font-weight:700;
                color:{status_color};letter-spacing:2px;
            ">{status_label}</span>
        </div>
        <div style="font-size:11px;color:#5a7aaa;font-family:'Rajdhani',sans-serif;line-height:1.9;">
            <div>Documents : <span style="color:#9ab0c8;">{doc_count}</span></div>
            <div>Chunks &nbsp;&nbsp;&nbsp;: <span style="color:#9ab0c8;">{chunk_total}</span></div>
            <div>LLM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#9ab0c8;">Gemini 2.5 Flash</span></div>
            <div>Retrieval : <span style="color:#9ab0c8;">FAISS top-{TOP_K}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  RIGHT — MAIN (tabbed)
# ════════════════════════════════════════════════
with col_main:

    # ── Top bar ──
    logo_main = _logo_html(height=36)
    logo_slot = f"""
    <div style="
        padding:5px;
        border:1.5px solid rgba(0,229,255,0.4);
        border-radius:10px;
        background:rgba(0,229,255,0.05);
        box-shadow:0 0 16px rgba(0,229,255,0.18);
        flex-shrink:0;
    ">{logo_main}</div>
    <div style="width:1px;height:36px;background:rgba(0,229,255,0.15);flex-shrink:0;"></div>
    """ if logo_main else ""

    st.html(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;padding-bottom:16px;border-bottom:1px solid rgba(0,229,255,0.12);">
        {logo_slot}
        <div>
            <div style="font-family:'Orbitron',monospace;font-size:17px;font-weight:900;color:#ccd6f6;letter-spacing:1px;line-height:1.1;">AI&LLM RAG Demo</div>
            <div style="font-size:11px;color:#4a6a8a;font-family:'Rajdhani',sans-serif;letter-spacing:0.4px;margin-top:2px;">Document-grounded AI · Evaluation Lab · Research</div>
        </div>
        <div style="margin-left:auto;">
            <span style="background:rgba(0,229,255,0.08);color:#00e5ff;font-size:10px;font-weight:700;font-family:'Orbitron',monospace;padding:4px 12px;border-radius:20px;border:1px solid rgba(0,229,255,0.3);letter-spacing:1px;box-shadow:0 0 10px rgba(0,229,255,0.1);">RAG &#8644; LLM</span>
        </div>
    </div>
    """)

    # ══ TABS ═══════════════════════════════════════════════════════════
    tab_query, tab_eval, tab_report = st.tabs([
        "◎  Query Interface",
        "⚗  Evaluation Lab",
        "📄  Research Report",
    ])

    # ─────────────────────────────────────────────
    #  TAB 1 — QUERY INTERFACE
    # ─────────────────────────────────────────────
    with tab_query:

        q_col, btn_col = st.columns([5, 1], gap="small")
        with q_col:
            query = st.text_input(
                label="",
                placeholder="Enter query directive…",
                label_visibility="collapsed",
                key="query",
            )
        with btn_col:
            ask_clicked = st.button("ASK →", type="primary", key="ask_btn", use_container_width=True)

        if ask_clicked:
            if not query.strip():
                st.warning("Enter a question before submitting.", icon="⚠️")
            elif st.session_state.faiss_index is None:
                st.warning("Upload at least one PDF before querying.", icon="📡")
            else:
                col_l, col_r = st.columns(2, gap="medium")
                with col_l:
                    with st.spinner("Retrieving + generating RAG answer…"):
                        rag_ans, retrieved = answer_rag(query)
                    st.session_state.rag_ans          = rag_ans
                    st.session_state.retrieved_chunks = retrieved
                with col_r:
                    with st.spinner("Generating bare LLM answer…"):
                        raw_ans = get_llm_answer_bare(query)
                    st.session_state.raw_ans = raw_ans

        has_response = st.session_state.rag_ans is not None

        if not has_response:
            st.markdown("""
            <div style="
                text-align:center; padding:70px 20px;
                border:1px dashed rgba(0,229,255,0.1);
                border-radius:14px;
                background:rgba(0,229,255,0.02);
                margin-top:12px;
            ">
                <div style="
                    font-size:40px; margin-bottom:14px;
                    color:rgba(0,229,255,0.25);
                    font-family:'Orbitron',monospace;
                ">◎</div>
                <div style="
                    font-family:'Orbitron',monospace;
                    font-size:13px; font-weight:600;
                    color:#3a6a9a; margin-bottom:8px;
                    letter-spacing:1px;
                ">AWAITING QUERY INPUT</div>
                <div style="font-size:13px;color:#3a6a9a;
                            font-family:'Rajdhani',sans-serif;line-height:1.6;">
                    Upload a document and submit a question.<br>
                    RAG and LLM responses render side by side.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            col_rag, col_llm = st.columns(2, gap="medium")

            with col_rag:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="
                        background:rgba(0,229,255,0.08);
                        color:#00e5ff;
                        font-family:'Orbitron',monospace;
                        font-size:9px; font-weight:700;
                        padding:3px 12px; border-radius:20px;
                        border:1px solid rgba(0,229,255,0.35);
                        letter-spacing:1.5px;
                        box-shadow:0 0 10px rgba(0,229,255,0.15);
                    ">RAG ANSWER</span>
                </div>
                <p style="
                    font-size:12px; color:#4a6a8a; font-weight:500;
                    font-family:'Rajdhani',sans-serif;
                    letter-spacing:0.3px;
                    margin-bottom:10px; margin-top:0;
                ">Grounded strictly in your uploaded documents</p>
                """, unsafe_allow_html=True)

                rag_text = st.session_state.rag_ans or ""
                rag_safe = rag_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                st.html(
                    '<div class="card-rag" style="'
                    'background:rgba(0,20,40,0.85);'
                    'border:1.5px solid rgba(0,229,255,0.3);'
                    'border-top:3px solid #00e5ff;'
                    'border-radius:10px;'
                    'padding:18px 20px;'
                    'font-family:Rajdhani,sans-serif;'
                    'font-size:15px;font-weight:400;'
                    'color:#c8e8f8;'
                    'line-height:1.85;min-height:80px;'
                    f'">{rag_safe}</div>'
                )

                if st.session_state.retrieved_chunks:
                    with st.expander("▸ View retrieved context passages"):
                        for i, chunk in enumerate(st.session_state.retrieved_chunks):
                            safe = chunk.replace("<", "&lt;").replace(">", "&gt;")
                            st.markdown(f"""
                            <div style="
                                background:rgba(0,229,255,0.03);
                                border:1px solid rgba(0,229,255,0.12);
                                border-left:3px solid rgba(0,229,255,0.4);
                                border-radius:6px;
                                padding:12px 14px; margin-bottom:8px;
                                font-size:12px; color:#4a6a8a;
                                font-family:'Rajdhani',sans-serif;
                                line-height:1.65;
                            ">
                                <div style="
                                    font-family:'Orbitron',monospace;
                                    font-size:8px; color:rgba(0,229,255,0.5);
                                    letter-spacing:1.5px; margin-bottom:8px;
                                ">PASSAGE {i+1}</div>
                                {safe}
                            </div>
                            """, unsafe_allow_html=True)

            with col_llm:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <span style="
                        background:rgba(139,0,255,0.08);
                        color:#bf7fff;
                        font-family:'Orbitron',monospace;
                        font-size:9px; font-weight:700;
                        padding:3px 12px; border-radius:20px;
                        border:1px solid rgba(139,0,255,0.35);
                        letter-spacing:1.5px;
                        box-shadow:0 0 10px rgba(139,0,255,0.15);
                    ">LLM ANSWER</span>
                </div>
                <p style="
                    font-size:12px; color:#4a6a8a; font-weight:500;
                    font-family:'Rajdhani',sans-serif;
                    letter-spacing:0.3px;
                    margin-bottom:10px; margin-top:0;
                ">From model training data only — no document context</p>
                """, unsafe_allow_html=True)

                raw_text_ans = st.session_state.raw_ans or ""
                llm_safe = raw_text_ans.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                st.html(
                    '<div class="card-llm" style="'
                    'background:rgba(20,0,40,0.85);'
                    'border:1.5px solid rgba(139,0,255,0.3);'
                    'border-top:3px solid #8b00ff;'
                    'border-radius:10px;'
                    'padding:18px 20px;'
                    'font-family:Rajdhani,sans-serif;'
                    'font-size:15px;font-weight:400;'
                    'color:#ddc8f0;'
                    'line-height:1.85;min-height:80px;'
                    f'">{llm_safe}</div>'
                )

                st.markdown("""
                <div style="
                    background:rgba(239,68,68,0.06);
                    border:1px solid rgba(239,68,68,0.2);
                    border-left:3px solid rgba(239,68,68,0.5);
                    border-radius:8px;
                    padding:10px 14px; margin-top:10px;
                    display:flex; gap:10px; align-items:flex-start;
                ">
                    <div style="
                        width:6px;height:6px;border-radius:50%;
                        background:#ef4444;
                        box-shadow:0 0 6px #ef4444;
                        flex-shrink:0;margin-top:5px;
                    "></div>
                    <span style="
                        font-size:12px; color:#7f4444;
                        font-family:'Rajdhani',sans-serif;
                        line-height:1.6; letter-spacing:0.2px;
                    ">
                        Training data only. Facts, names, and dates may be
                        hallucinated or outdated. Cross-reference with the
                        RAG answer for accuracy.
                    </span>
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    #  TAB 2 — EVALUATION LAB
    # ─────────────────────────────────────────────
    with tab_eval:

        st.markdown("""
        <div style="margin-bottom:18px;">
            <div style="font-family:'Orbitron',monospace;font-size:13px;font-weight:700;
                        color:#00e5ff;letter-spacing:1.5px;margin-bottom:6px;">
                RAG EVALUATION LAB
            </div>
            <div style="font-size:13px;color:#4a6a8a;font-family:'Rajdhani',sans-serif;line-height:1.7;">
                Measures <b style="color:#ccd6f6;">Hit Rate @5</b> across three token-based chunk-size
                configurations using <b style="color:#ccd6f6;">LLM-as-judge</b> (Gemini grades each
                retrieval as relevant or not). Methodology from Session 8, Slide 26–28.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Config summary cards
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3, gap="small")
        cfg_data = [
            ("200 tokens", "40 tok overlap", "#00e5ff", "Small — precise, many chunks"),
            ("500 tokens", "100 tok overlap", "#8b5cf6", "Medium — balanced (current default)"),
            ("1000 tokens", "200 tok overlap", "#00c896", "Large — broad context, fewer chunks"),
        ]
        for col, (label, sub, color, desc) in zip([cfg_col1, cfg_col2, cfg_col3], cfg_data):
            with col:
                st.markdown(f"""
                <div style="
                    background:rgba(0,8,20,0.8);
                    border:1.5px solid {color}33;
                    border-top:3px solid {color};
                    border-radius:10px;
                    padding:14px 16px;
                    margin-bottom:12px;
                ">
                    <div style="font-family:'Orbitron',monospace;font-size:11px;
                                font-weight:700;color:{color};letter-spacing:1px;">{label}</div>
                    <div style="font-size:10px;color:#4a6a8a;font-family:'Orbitron',monospace;
                                margin:3px 0 6px 0;letter-spacing:0.5px;">{sub}</div>
                    <div style="font-size:12px;color:#8892b0;font-family:'Rajdhani',sans-serif;">
                        {desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Warning about model truncation
        st.markdown("""
        <div style="
            background:rgba(255,193,7,0.06);
            border:1px solid rgba(255,193,7,0.25);
            border-left:3px solid #ffc107;
            border-radius:8px;
            padding:10px 14px; margin-bottom:16px;
            font-size:12px; color:#a08000;
            font-family:'Rajdhani',sans-serif; line-height:1.6;
        ">
            <b style="color:#ffc107;">Note:</b> all-MiniLM-L6-v2 truncates input to 256 tokens.
            The 1000-token config will have its embeddings silently truncated — a deliberate
            trade-off demonstration showing why chunk size must respect the embedding model's limit.
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.faiss_index is None:
            st.info("Upload a PDF in the sidebar to enable evaluation.", icon="📡")
        else:
            run_eval = st.button(
                "⚗  Run Evaluation  (generates queries + judges retrieval via Gemini)",
                type="primary",
                key="run_eval_btn",
                use_container_width=True,
            )

            if run_eval:
                # Concatenate all raw texts
                combined_text = "\n\n".join(st.session_state.raw_texts.values())
                if not combined_text.strip():
                    st.error("No raw text found. Re-upload your PDF.")
                else:
                    prog = st.progress(0, text="Generating test queries via Gemini…")
                    try:
                        # Phase 1 — query generation
                        queries_generated = _generate_test_queries(combined_text, n=8)
                        prog.progress(15, text=f"Generated {len(queries_generated)} test queries. Building indexes…")

                        tokenizer = embed_model.tokenizer
                        eval_configs = [
                            {"label": "200 tokens",  "tokens": 200,  "overlap": 40},
                            {"label": "500 tokens",  "tokens": 500,  "overlap": 100},
                            {"label": "1000 tokens", "tokens": 1000, "overlap": 200},
                        ]
                        results = {}
                        detail  = []
                        n_cfgs  = len(eval_configs)
                        n_q     = len(queries_generated)
                        total_ops = n_cfgs * n_q

                        op_done = 0
                        for ci, cfg in enumerate(eval_configs):
                            chunks_cfg = _chunk_by_tokens(
                                combined_text, tokenizer, cfg["tokens"], cfg["overlap"]
                            )
                            index_cfg, _ = _build_temp_index(chunks_cfg)
                            hits = 0

                            for qi, q in enumerate(queries_generated):
                                op_done += 1
                                pct = 15 + int(85 * op_done / total_ops)
                                prog.progress(
                                    pct,
                                    text=f"Config {ci+1}/{n_cfgs} · Query {qi+1}/{n_q}: {q[:50]}…"
                                )
                                q_emb = embed_model.encode([q], convert_to_numpy=True)
                                faiss.normalize_L2(q_emb)
                                _, idxs = index_cfg.search(q_emb.astype(np.float32), 5)
                                retrieved = [
                                    chunks_cfg[i] for i in idxs[0]
                                    if 0 <= i < len(chunks_cfg)
                                ]
                                verdict = _llm_judge(q, retrieved)
                                if verdict:
                                    hits += 1
                                detail.append({
                                    "Config": cfg["label"],
                                    "Query": q[:75] + ("…" if len(q) > 75 else ""),
                                    "Result": "✓ HIT" if verdict else "✗ MISS",
                                })

                            results[cfg["label"]] = {
                                "hit_rate": hits / n_q if n_q else 0.0,
                                "n_chunks": len(chunks_cfg),
                                "hits": hits,
                                "total": n_q,
                            }

                        prog.progress(100, text="Evaluation complete.")

                        st.session_state.eval_results   = results
                        st.session_state.eval_queries    = queries_generated
                        st.session_state.eval_chart_png  = _render_eval_chart(results)
                        st.session_state.eval_detail     = detail

                    except Exception as e:
                        st.error(f"Evaluation failed: {e}", icon="⚠️")
                    finally:
                        prog.empty()

            # ── Display cached results ──
            if st.session_state.eval_results:
                results = st.session_state.eval_results

                # Metric summary row
                st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3, gap="small")
                colors_metric = ["#00e5ff", "#8b5cf6", "#00c896"]
                for col, (label, metrics), color in zip(
                    [m1, m2, m3], results.items(), colors_metric
                ):
                    with col:
                        hr_pct = metrics["hit_rate"] * 100
                        target_ok = "✓" if hr_pct >= 85 else "✗"
                        target_col = "#00e5ff" if hr_pct >= 85 else "#ef4444"
                        st.markdown(f"""
                        <div style="
                            background:rgba(0,8,20,0.85);
                            border:1.5px solid {color}44;
                            border-radius:10px; padding:16px 18px;
                            text-align:center; margin-bottom:14px;
                        ">
                            <div style="font-family:'Orbitron',monospace;font-size:9px;
                                        color:{color};letter-spacing:1.5px;margin-bottom:6px;">
                                {label}
                            </div>
                            <div style="font-family:'Orbitron',monospace;font-size:26px;
                                        font-weight:900;color:{color};line-height:1;">
                                {hr_pct:.0f}%
                            </div>
                            <div style="font-size:11px;color:#4a6a8a;
                                        font-family:'Rajdhani',sans-serif;margin-top:4px;">
                                Hit Rate @5
                            </div>
                            <div style="font-size:10px;color:{target_col};
                                        font-family:'Rajdhani',sans-serif;margin-top:4px;">
                                {target_ok} Target ≥ 85% &nbsp;·&nbsp;
                                {metrics['hits']}/{metrics['total']} queries
                            </div>
                            <div style="font-size:10px;color:#5a7aaa;
                                        font-family:'Orbitron',monospace;margin-top:6px;
                                        letter-spacing:0.5px;">
                                {metrics['n_chunks']} chunks
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Chart
                if st.session_state.eval_chart_png:
                    st.image(
                        st.session_state.eval_chart_png,
                        use_container_width=True,
                        caption="Figure 1 — Hit Rate @5 and index size by chunk configuration",
                    )

                # Analysis
                best_cfg   = max(results, key=lambda c: results[c]["hit_rate"])
                best_rate  = results[best_cfg]["hit_rate"] * 100
                worst_cfg  = min(results, key=lambda c: results[c]["hit_rate"])
                worst_rate = results[worst_cfg]["hit_rate"] * 100
                delta      = best_rate - worst_rate

                st.markdown(f"""
                <div style="
                    background:rgba(0,8,20,0.85);
                    border:1.5px solid rgba(0,229,255,0.2);
                    border-left:4px solid #00e5ff;
                    border-radius:10px; padding:18px 20px; margin-top:4px;
                ">
                    <div style="font-family:'Orbitron',monospace;font-size:10px;
                                color:#00e5ff;letter-spacing:2px;margin-bottom:10px;">
                        ANALYSIS
                    </div>
                    <div style="font-size:14px;color:#a8d8e8;
                                font-family:'Rajdhani',sans-serif;line-height:1.8;">
                        The <b style="color:#00e5ff;">{best_cfg}</b> configuration achieved the
                        highest Hit Rate @5 at <b style="color:#00e5ff;">{best_rate:.0f}%</b>,
                        outperforming <b style="color:#ef4444;">{worst_cfg}</b>
                        ({worst_rate:.0f}%) by <b style="color:#ffc107;">{delta:.0f} percentage points</b>.
                        <br><br>
                        Smaller chunks produce more granular index entries, improving precision at the cost
                        of potentially splitting coherent passages across boundaries. Larger chunks preserve
                        contextual continuity but risk exceeding the embedding model's 256-token limit
                        (all-MiniLM-L6-v2), causing silent truncation that degrades semantic fidelity.
                        The optimal configuration balances chunk coherence with embedding-window capacity.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Per-query detail table
                with st.expander("▸ Per-query judgment detail"):
                    for cfg_name, color in zip(
                        ["200 tokens", "500 tokens", "1000 tokens"],
                        ["#00e5ff", "#8b5cf6", "#00c896"],
                    ):
                        st.markdown(f"""
                        <div style="font-family:'Orbitron',monospace;font-size:9px;
                                    color:{color};letter-spacing:1.5px;
                                    margin:12px 0 6px 0;">{cfg_name}</div>
                        """, unsafe_allow_html=True)
                        rows = [r for r in st.session_state.eval_detail if r["Config"] == cfg_name]
                        for row in rows:
                            hit_col = "#00e5ff" if "HIT" in row["Result"] else "#ef4444"
                            st.markdown(f"""
                            <div style="
                                display:flex; gap:12px; align-items:center;
                                padding:7px 12px; margin-bottom:3px;
                                background:rgba(0,8,20,0.6);
                                border:1px solid rgba(255,255,255,0.05);
                                border-radius:6px;
                                font-family:'Rajdhani',sans-serif; font-size:12px;
                            ">
                                <span style="color:{hit_col};font-weight:700;
                                            font-family:'Orbitron',monospace;
                                            font-size:9px;flex-shrink:0;
                                            min-width:50px;">{row['Result']}</span>
                                <span style="color:#8892b0;">{row['Query']}</span>
                            </div>
                            """, unsafe_allow_html=True)

                # Auto-generated queries list
                if st.session_state.eval_queries:
                    with st.expander("▸ Auto-generated test queries"):
                        for i, q in enumerate(st.session_state.eval_queries, 1):
                            st.markdown(f"""
                            <div style="
                                padding:8px 14px; margin-bottom:4px;
                                background:rgba(0,8,20,0.6);
                                border-left:3px solid rgba(0,229,255,0.3);
                                border-radius:0 6px 6px 0;
                                font-size:13px; color:#8892b0;
                                font-family:'Rajdhani',sans-serif;
                            ">
                                <span style="color:rgba(0,229,255,0.4);
                                            font-family:'Orbitron',monospace;
                                            font-size:9px;margin-right:10px;">{i:02d}</span>
                                {q}
                            </div>
                            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    #  TAB 3 — RESEARCH REPORT
    # ─────────────────────────────────────────────
    with tab_report:

        # Build best-practices list as Python (avoids f-string nesting issues)
        _bp_items = [
            ("01", "Chunk at logical boundaries",
             "Engineering standards use numbered sections; split at headings to preserve regulatory context and avoid crossing definition boundaries."),
            ("02", "Preserve metadata per chunk",
             "Store source filename, page number, section title, and document version alongside each chunk vector for citation auditability and filtered retrieval."),
            ("03", "Set TOP_K to 5\u20137",
             "Retrieval accuracy peaks in this range (Session\u00a08, Slide\u00a021). Below 5: insufficient context. Above 7: context dilution degrades LLM generation quality."),
            ("04", "Evaluate before optimising",
             "Build a labelled test set of 50+ query-answer pairs before tuning chunk size or embedding model. You cannot improve what you cannot measure."),
            ("05", "Use strict system prompts",
             "Ground LLM generation strictly in retrieved context (temperature \u2264\u00a00.3, system_instruction restricting to passages). Eliminates hallucination risk demonstrated in the bare LLM panel."),
        ]
        _bp_html = "".join(
            '<div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:10px;'
            'padding:12px 16px;background:rgba(0,10,28,0.7);border-radius:8px;'
            'border:1px solid rgba(0,229,255,0.12);">'
            f'<div style="background:#00e5ff;color:#020c1b;font-weight:900;'
            f'font-family:Orbitron,monospace;font-size:10px;padding:3px 9px;'
            f'border-radius:4px;flex-shrink:0;margin-top:3px;">{num}</div>'
            f'<div style="font-size:13px;color:#b8d4ee;line-height:1.75;">'
            f'<b style="color:#e8f4ff;">{title}</b> \u2014 {body}</div></div>'
            for num, title, body in _bp_items
        )

        _report_html = (
'<div style="background:rgba(0,8,24,0.9);border:1.5px solid rgba(0,229,255,0.25);'
'border-top:3px solid #00e5ff;border-radius:12px;padding:30px 34px;'
'font-family:Rajdhani,sans-serif;line-height:1.85;">'

# ── Header ──
'<div style="text-align:center;margin-bottom:28px;padding-bottom:22px;'
'border-bottom:1px solid rgba(0,229,255,0.18);">'
'<div style="font-family:Orbitron,monospace;font-size:10px;font-weight:700;'
'color:#4a8aaa;letter-spacing:3px;margin-bottom:6px;text-transform:uppercase;">Research Report</div>'
'<div style="font-size:18px;font-weight:700;color:#00e5ff;margin-bottom:8px;line-height:1.3;">'
'Retrieval-Augmented Generation: Chunking, Embedding, and Evaluation Strategies</div>'
'<div style="font-size:13px;color:#7a9aba;margin-bottom:4px;">'
'Group 18 &nbsp;\u00b7&nbsp; AI and Advanced Large Models &nbsp;\u00b7&nbsp; Beihang University</div>'
'<div style="font-size:12px;color:#5a7a9a;">'
'Abiola Olayinka Ademola (LS2525239) &nbsp;\u00b7&nbsp; '
'Maryam Omolade Ajadi (LS2525249) &nbsp;\u00b7&nbsp; '
'Nasifah Alalade Ajoke (LS2525250)</div></div>'

# ── Section 1 ──
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#00e5ff;'
'letter-spacing:2.5px;margin:0 0 10px 0;padding-bottom:4px;'
'border-bottom:1px solid rgba(0,229,255,0.1);">1. INTRODUCTION</div>'
'<div style="font-size:14px;color:#c0ddf2;margin-bottom:22px;line-height:1.85;">'
'Foundation models such as Gemini and GPT-4 encode world knowledge in their weights, frozen at '
'training time. This <em style="color:#a0c8e8;">parametric memory</em> exhibits three critical failure modes: it cannot be '
'updated without retraining, its capacity is bounded by model size, and it produces '
'<em style="color:#a0c8e8;">hallucinations</em> \u2014 confident yet factually incorrect outputs \u2014 when queried beyond its '
'knowledge boundary. Lewis et al. (2020) formalised <strong style="color:#e8f4ff;">Retrieval-Augmented '
'Generation (RAG)</strong> as the engineering solution: connect the LLM to non-parametric external '
'memory at inference time, grounding generation in retrieved, verifiable evidence. This enables '
'document-specific QA, auditability, and knowledge freshness without retraining.</div>'

# ── Section 2 ──
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#00e5ff;'
'letter-spacing:2.5px;margin:0 0 10px 0;padding-bottom:4px;'
'border-bottom:1px solid rgba(0,229,255,0.1);">2. CHUNKING STRATEGY COMPARISON</div>'
'<div style="font-size:14px;color:#c0ddf2;margin-bottom:14px;line-height:1.85;">'
'Documents must be segmented prior to embedding because (a) embedding models have fixed '
'context windows \u2014 all-MiniLM-L6-v2 truncates at 256 tokens \u2014 and (b) full documents dilute '
'retrieval precision. Three primary strategies exist:</div>'
'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px;">'
'<div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.25);'
'border-top:2px solid #00e5ff;border-radius:8px;padding:16px;">'
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#00e5ff;'
'letter-spacing:1.5px;margin-bottom:10px;">FIXED-SIZE</div>'
'<div style="font-size:13px;color:#b0cce6;line-height:1.75;">'
'Equal token blocks with configurable overlap (10\u201320%). Simple and fast. '
'Risk: splits sentences mid-thought at boundaries.<br>'
'<span style="color:#6a90b0;font-size:12px;">Best for: homogeneous prose, quick prototypes.</span></div></div>'
'<div style="background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.25);'
'border-top:2px solid #8b5cf6;border-radius:8px;padding:16px;">'
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#a07fff;'
'letter-spacing:1.5px;margin-bottom:10px;">SEMANTIC</div>'
'<div style="font-size:13px;color:#b0cce6;line-height:1.75;">'
'Splits at paragraph or sentence boundaries. Preserves meaning. '
'Variable chunk sizes complicate batch embedding.<br>'
'<span style="color:#6a90b0;font-size:12px;">Best for: narrative documents, academic papers.</span></div></div>'
'<div style="background:rgba(0,200,150,0.05);border:1px solid rgba(0,200,150,0.25);'
'border-top:2px solid #00c896;border-radius:8px;padding:16px;">'
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#00c896;'
'letter-spacing:1.5px;margin-bottom:10px;">RECURSIVE</div>'
'<div style="font-size:13px;color:#b0cce6;line-height:1.75;">'
'Hierarchical decomposition: Doc \u2192 Section \u2192 Paragraph. '
'Optimal for structured technical documents.<br>'
'<span style="color:#6a90b0;font-size:12px;">Best for: engineering manuals, legal documents.</span></div></div>'
'</div>'
'<div style="font-size:13px;color:#8aacca;margin-bottom:24px;padding:10px 14px;'
'background:rgba(0,229,255,0.04);border-left:3px solid rgba(0,229,255,0.3);border-radius:0 6px 6px 0;">'
'This implementation uses fixed-size chunking as a reproducible baseline (500 chars, '
'100 char overlap \u2248 20% overlap, within the 10\u201320% guideline). The Evaluation Lab '
'compares three token-based sizes to empirically identify the optimal configuration.</div>'

# ── Section 3 ──
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#00e5ff;'
'letter-spacing:2.5px;margin:0 0 10px 0;padding-bottom:4px;'
'border-bottom:1px solid rgba(0,229,255,0.1);">3. EMBEDDING MODEL TRADE-OFFS</div>'
'<div style="font-size:14px;color:#c0ddf2;margin-bottom:14px;line-height:1.85;">'
'Embedding models convert text into dense vectors where semantic similarity is measurable '
'via cosine distance. Model selection involves four competing dimensions:</div>'
'<table style="width:100%;border-collapse:collapse;font-size:13px;'
'font-family:Rajdhani,sans-serif;margin-bottom:14px;">'
'<thead><tr style="border-bottom:1px solid rgba(0,229,255,0.25);background:rgba(0,229,255,0.06);">'
'<th style="text-align:left;padding:10px 14px;color:#00e5ff;font-family:Orbitron,monospace;font-size:9px;letter-spacing:1px;">Model</th>'
'<th style="text-align:center;padding:10px;color:#00e5ff;font-family:Orbitron,monospace;font-size:9px;">Dims</th>'
'<th style="text-align:center;padding:10px;color:#00e5ff;font-family:Orbitron,monospace;font-size:9px;">MTEB</th>'
'<th style="text-align:center;padding:10px;color:#00e5ff;font-family:Orbitron,monospace;font-size:9px;">Latency</th>'
'<th style="text-align:center;padding:10px;color:#00e5ff;font-family:Orbitron,monospace;font-size:9px;">Cost</th>'
'</tr></thead><tbody>'
'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);background:rgba(0,229,255,0.04);">'
'<td style="padding:10px 14px;color:#e8f4ff;font-weight:600;">all-MiniLM-L6-v2 \u2605</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">384</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">56.3</td>'
'<td style="text-align:center;padding:10px;color:#00c896;">~5 ms</td>'
'<td style="text-align:center;padding:10px;color:#00c896;">Free</td></tr>'
'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
'<td style="padding:10px 14px;color:#ccd6f6;">BGE-large-en-v1.5</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">1024</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">64.6</td>'
'<td style="text-align:center;padding:10px;color:#ffc107;">~40 ms</td>'
'<td style="text-align:center;padding:10px;color:#00c896;">Free</td></tr>'
'<tr style="border-bottom:1px solid rgba(255,255,255,0.06);">'
'<td style="padding:10px 14px;color:#ccd6f6;">text-embedding-3-small</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">1536</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">62.3</td>'
'<td style="text-align:center;padding:10px;color:#ffc107;">~20 ms</td>'
'<td style="text-align:center;padding:10px;color:#ef6060;">$0.02/1M</td></tr>'
'<tr><td style="padding:10px 14px;color:#ccd6f6;">Gemini text-embedding-004</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">768</td>'
'<td style="text-align:center;padding:10px;color:#a0bcd8;">66.1</td>'
'<td style="text-align:center;padding:10px;color:#ffc107;">~30 ms</td>'
'<td style="text-align:center;padding:10px;color:#ef6060;">API rate</td></tr>'
'</tbody></table>'
'<div style="font-size:13px;color:#8aacca;margin-bottom:24px;padding:10px 14px;'
'background:rgba(0,229,255,0.04);border-left:3px solid rgba(0,229,255,0.3);border-radius:0 6px 6px 0;">'
'\u2605 Selected for this project: all-MiniLM-L6-v2 (Reimers &amp; Gurevych, 2019) provides a strong '
'free baseline with 384-dimensional vectors and ~5\u202fms local latency. Production deployments '
'demanding higher recall on specialised corpora should consider BGE-large or fine-tuning on '
'domain data for 5\u201315% accuracy gains (Muennighoff et al., 2022).</div>'

# ── Section 4 ──
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#00e5ff;'
'letter-spacing:2.5px;margin:0 0 10px 0;padding-bottom:4px;'
'border-bottom:1px solid rgba(0,229,255,0.1);">4. BEST PRACTICES FOR ENGINEERING DOCUMENT RAG</div>'
'<div style="font-size:14px;color:#c0ddf2;margin-bottom:14px;line-height:1.85;">'
'Based on Session 8 guidelines and empirical evaluation results:</div>'
f'<div style="margin-bottom:24px;">{_bp_html}</div>'

# ── Section 5 ──
'<div style="font-family:Orbitron,monospace;font-size:9px;color:#00e5ff;'
'letter-spacing:2.5px;margin:0 0 14px 0;padding-bottom:4px;'
'border-bottom:1px solid rgba(0,229,255,0.1);">5. REFERENCES</div>'
'<div style="font-size:13px;color:#7a9aba;line-height:2.1;font-family:Rajdhani,sans-serif;">'
'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
'[1] Lewis, P., Perez, E., et al. (2020). <em style="color:#a0bcd8;">Retrieval-Augmented Generation for '
'Knowledge-Intensive NLP Tasks.</em> NeurIPS 2020. arXiv:2005.11401</div>'
'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
'[2] Johnson, J., Douze, M., &amp; J\u00e9gou, H. (2019). <em style="color:#a0bcd8;">Billion-scale similarity '
'search with GPUs.</em> IEEE Transactions on Big Data. arXiv:1702.08734</div>'
'<div style="padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
'[3] Reimers, N., &amp; Gurevych, I. (2019). <em style="color:#a0bcd8;">Sentence-BERT: Sentence Embeddings '
'using Siamese BERT-Networks.</em> EMNLP 2019. arXiv:1908.10084</div>'
'<div style="padding:8px 0;">'
'[4] Muennighoff, N., et al. (2022). <em style="color:#a0bcd8;">MTEB: Massive Text Embedding Benchmark.</em> '
'EACL 2023. arXiv:2210.07316</div>'
'</div>'
'</div>'
        )
        st.html(_report_html)
