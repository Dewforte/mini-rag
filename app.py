"""
app.py — Dynamic RAG Web Application with Dual Output Comparison
Framework : Streamlit
LLM backend: GitHub Models  (https://models.inference.ai.azure.com)
"""

import base64
import os

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ──────────────────────────── Configuration ────────────────────────────
load_dotenv()

GITHUB_TOKEN  : str = os.getenv("GITHUB_TOKEN", "")
API_BASE_URL  : str = "https://models.inference.ai.azure.com"
MODEL_NAME    : str = "gpt-4o"
EMBED_MODEL   : str = "all-MiniLM-L6-v2"
CHUNK_SIZE    : int = 500
CHUNK_OVERLAP : int = 100
TOP_K         : int = 3

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
    llm    = OpenAI(api_key=GITHUB_TOKEN, base_url=API_BASE_URL)
    return embed, llm

embed_model, client = load_models()


# ──────────────────────────── Session state ────────────────────────────
_defaults = {
    "faiss_index"      : None,
    "chunks"           : [],
    "chunk_counts"     : {},
    "uploaded_files"   : [],
    "rag_ans"          : None,
    "raw_ans"          : None,
    "retrieved_chunks" : [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────── Backend functions ────────────────────────

def _ingest_file(uploaded_file) -> tuple[list[str], int]:
    data = uploaded_file.read()
    doc  = fitz.open(stream=data, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    doc.close()
    if not text.strip():
        raise ValueError("PDF appears empty or unreadable.")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks, len(chunks)


def _rebuild_index(all_chunks: list[str]) -> faiss.IndexFlatIP:
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def answer_rag(question: str) -> tuple[str, list[str]]:
    if st.session_state.faiss_index is None or not st.session_state.chunks:
        return "⚠️ Upload and process a PDF document first.", []
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
        "You are a helpful assistant. "
        "Answer ONLY using the provided passages. "
        "If the passages do not contain the answer, reply EXACTLY: "
        "'The document does not cover this topic.' "
        "Do not use outside knowledge."
    )
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2, max_tokens=600,
        )
        return r.choices[0].message.content.strip(), retrieved
    except Exception as e:
        return f"[ERROR] RAG call failed: {e}", retrieved


def get_llm_answer_bare(question: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": question}],
            temperature=0.7, max_tokens=600,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"


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
                ">Knowledge Engine v1.0</div>
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
                Knowledge Engine v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Section header ──
    st.markdown("""
    <p style="
        font-family:'Orbitron',monospace;
        color:#00e5ff; font-size:10px; font-weight:700;
        letter-spacing:2.5px; text-transform:uppercase;
        margin:0 0 2px 0;
    ">Knowledge Sources</p>
    <p style="color:#4a6a8a;font-size:11px;margin:0 0 10px 0;
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
                    chunks, n = _ingest_file(f)
                    st.session_state.chunks.extend(chunks)
                    st.session_state.chunk_counts[f.name] = n
                    st.session_state.uploaded_files.append(f.name)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
            if st.session_state.chunks:
                st.session_state.faiss_index = _rebuild_index(st.session_state.chunks)
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
            <div style="color:#1e3a5f;font-size:20px;margin-bottom:4px;">⬡</div>
            <div style="color:#1e3a5f;font-size:11px;
                        font-family:'Rajdhani',sans-serif;letter-spacing:0.3px;">
                No documents loaded
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    if st.button("⌫  Clear all documents", key="clear_btn", use_container_width=True):
        st.session_state.faiss_index      = None
        st.session_state.chunks           = []
        st.session_state.chunk_counts     = {}
        st.session_state.uploaded_files   = []
        st.session_state.rag_ans          = None
        st.session_state.raw_ans          = None
        st.session_state.retrieved_chunks = []
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
        <div style="font-size:11px;color:#2a4a6a;font-family:'Rajdhani',sans-serif;line-height:1.9;">
            <div>Documents : <span style="color:#64748b;">{doc_count}</span></div>
            <div>Chunks &nbsp;&nbsp;&nbsp;: <span style="color:#64748b;">{chunk_total}</span></div>
            <div>Model &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#64748b;">GPT-4o</span></div>
            <div>Retrieval : <span style="color:#64748b;">FAISS top-{TOP_K}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  RIGHT — MAIN
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
            <div style="font-size:11px;color:#4a6a8a;font-family:'Rajdhani',sans-serif;letter-spacing:0.4px;margin-top:2px;">Document-grounded AI vs unconstrained LLM</div>
        </div>
        <div style="margin-left:auto;">
            <span style="background:rgba(0,229,255,0.08);color:#00e5ff;font-size:10px;font-weight:700;font-family:'Orbitron',monospace;padding:4px 12px;border-radius:20px;border:1px solid rgba(0,229,255,0.3);letter-spacing:1px;box-shadow:0 0 10px rgba(0,229,255,0.1);">RAG &#8644; LLM</span>
        </div>
    </div>
    """)

    # ── Query row ──
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

    # ── Handle click ──
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

    # ── Output area ──
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
                color:#1e3a5f; margin-bottom:8px;
                letter-spacing:1px;
            ">AWAITING QUERY INPUT</div>
            <div style="font-size:13px;color:#1e3a5f;
                        font-family:'Rajdhani',sans-serif;line-height:1.6;">
                Upload a document and submit a question.<br>
                RAG and LLM responses will render side by side.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        col_rag, col_llm = st.columns(2, gap="medium")

        # ── RAG panel ──
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
            st.markdown(f"""
            <div class="card-rag" style="
                background:rgba(0,20,40,0.85);
                border:1.5px solid rgba(0,229,255,0.3);
                border-top:3px solid #00e5ff;
                border-radius:10px;
                padding:18px 20px;
                font-family:'Rajdhani',sans-serif;
                font-size:15px; font-weight:400;
                color:#a8d8e8;
                line-height:1.8; min-height:80px;
            ">{rag_text}</div>
            """, unsafe_allow_html=True)

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

        # ── LLM panel ──
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

            raw_text = st.session_state.raw_ans or ""
            st.markdown(f"""
            <div class="card-llm" style="
                background:rgba(20,0,40,0.85);
                border:1.5px solid rgba(139,0,255,0.3);
                border-top:3px solid #8b00ff;
                border-radius:10px;
                padding:18px 20px;
                font-family:'Rajdhani',sans-serif;
                font-size:15px; font-weight:400;
                color:#d4b8e8;
                line-height:1.8; min-height:80px;
            ">{raw_text}</div>
            """, unsafe_allow_html=True)

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
