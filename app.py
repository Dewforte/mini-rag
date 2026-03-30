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
MODEL_NAME    : str = "gpt-4o-mini"
EMBED_MODEL   : str = "all-MiniLM-L6-v2"
CHUNK_SIZE    : int = 500
CHUNK_OVERLAP : int = 100
TOP_K         : int = 3

# ── Page config — must be the very first Streamlit call ────────────────
st.set_page_config(
    page_title="RAG Demo",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────── Global CSS ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #d9e1eb !important;
}

/* Remove default Streamlit padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden !important; }
.stDeployButton   { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* Fix text visibility */
p, span, div, label { color: inherit; }

/* ── Dark sidebar column ── */
[data-testid="column"]:first-child > div {
    background: #0f1f33 !important;
    border-radius: 12px !important;
    padding: 1.2rem 1rem !important;
    min-height: 82vh;
}

/* Sidebar text overrides */
[data-testid="column"]:first-child label,
[data-testid="column"]:first-child p,
[data-testid="column"]:first-child span,
[data-testid="column"]:first-child small {
    color: #94a3b8 !important;
}

/* ── File upload area — teal bounding box ── */
[data-testid="column"]:first-child [data-testid="stFileUploader"] {
    border: 2px solid #0e7490 !important;
    border-radius: 10px !important;
    padding: 10px 10px 6px !important;
    background: rgba(14,116,144,0.06) !important;
}

/* File uploader dropzone — dark variant inside sidebar */
[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] {
    background: #0a1628 !important;
    border: 1.5px dashed #0e7490 !important;
    border-radius: 7px !important;
    padding: 8px 12px !important;
    min-height: unset !important;
}
[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] span,
[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] p {
    color: #67e8f9 !important;
    font-size: 12px !important;
}
[data-testid="column"]:first-child [data-testid="stFileUploaderDropzone"] svg {
    color: #0e7490 !important;
    width: 18px !important;
    height: 18px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small { display: none !important; }

/* File uploader label inside the teal box */
[data-testid="column"]:first-child [data-testid="stFileUploader"] label p {
    color: #7dd3fc !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}

/* Spinner text */
[data-testid="stSpinner"] p { color: #94a3b8 !important; }

/* ── Query input (main column) ── */
input[type="text"], textarea {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 8px !important;
    caret-color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
input::placeholder, textarea::placeholder { color: #94a3b8 !important; }
input:focus, textarea:focus {
    border-color: #4f46e5 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.14) !important;
    background: #ffffff !important;
}

/* ── Primary "Ask" button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    height: 38px !important;
    width: 100% !important;
    letter-spacing: 0.2px !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: #4338ca !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.4) !important;
}

/* ── Sidebar clear button — muted outline ── */
[data-testid="column"]:first-child [data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
}
[data-testid="column"]:first-child [data-testid="stButton"] button:hover {
    background: #1e3a5f !important;
    color: #e2e8f0 !important;
    border-color: #4a6fa5 !important;
}

/* ── Expander (retrieved context) ── */
[data-testid="stExpander"] {
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
    background: #f9fafb !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important;
    color: #1f2937 !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    color: #059669 !important;
}

/* ── Divider ── */
[data-testid="stDivider"] { border-color: #1e3a5f !important; margin: 10px 0 !important; }

/* ── Warning / info banners ── */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important; }

/* ── Main column white card ── */
[data-testid="column"]:last-child > div {
    background: #f1f5f9 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Model loading (cached) ───────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_models():
    embed  = SentenceTransformer(EMBED_MODEL)
    llm    = OpenAI(api_key=GITHUB_TOKEN, base_url=API_BASE_URL)
    return embed, llm

embed_model, client = load_models()


# ──────────────────────────── Session state ────────────────────────────
_defaults = {
    "faiss_index"      : None,
    "chunks"           : [],        # combined chunks across all files
    "chunk_counts"     : {},        # {filename: n_chunks}
    "uploaded_files"   : [],        # list of indexed filenames (ordered)
    "rag_ans"          : None,
    "raw_ans"          : None,
    "retrieved_chunks" : [],        # raw chunk strings for context expander
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────── Backend functions ────────────────────────
# Core chunking/embedding/retrieval/LLM logic — unchanged.

def _ingest_file(uploaded_file) -> tuple[list[str], int]:
    """Extract text from a PDF and chunk it. Returns (chunks, n_chunks)."""
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
    """Embed all chunks and build a fresh FAISS index."""
    embeddings = embed_model.encode(all_chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return index


def answer_rag(question: str) -> tuple[str, list[str]]:
    """Document-grounded answer + list of retrieved raw chunk strings."""
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
    context = "\n\n---\n\n".join(
        f"Passage {i+1}:\n{c}" for i, c in enumerate(retrieved)
    )
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
            temperature=0.2,
            max_tokens=600,
        )
        return r.choices[0].message.content.strip(), retrieved
    except Exception as e:
        return f"[ERROR] RAG call failed: {e}", retrieved


def get_llm_answer_bare(question: str) -> str:
    """Bare LLM call — no system prompt, no document context."""
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=600,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"


# ──────────────────────────── Logo helper ──────────────────────────────

def _logo_html(height: int = 32) -> str:
    """Return an <img> tag for the logo file in the project root, or ''."""
    for ext in ("png", "jpg", "jpeg", "webp", "svg"):
        for name in (f"logo.{ext}", f"Logo.{ext}"):
            if os.path.isfile(name):
                if ext == "svg":
                    try:
                        svg = open(name).read()
                        # Inject height/width so SVG scales correctly
                        return svg.replace("<svg", f'<svg height="{height}"', 1)
                    except Exception:
                        pass
                else:
                    try:
                        with open(name, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
                        return (
                            f'<img src="data:image/{mime};base64,{b64}" '
                            f'height="{height}" '
                            f'style="border-radius:6px;object-fit:contain;display:block;">'
                        )
                    except Exception:
                        pass
    return ""


# ══════════════════════════════════════════════════════════════════════
#  TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════
col_sidebar, col_main = st.columns([1, 3], gap="large")


# ════════════════════════════════════════════════
#  LEFT — SIDEBAR
# ════════════════════════════════════════════════
with col_sidebar:

    # 1. Header
    st.markdown(
        "<p style='color:#f0f9ff;font-size:15px;font-weight:700;"
        "letter-spacing:-0.2px;margin:0 0 2px 0;'>Knowledge Sources</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#7dd3fc;font-size:11px;margin:0 0 10px 0;'>"
        "PDFs used for retrieval</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # 2. File uploader (multiple)
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    # Identify new files (not yet indexed)
    current_names = [f.name for f in (uploaded_files or [])]
    new_files     = [
        f for f in (uploaded_files or [])
        if f.name not in st.session_state.chunk_counts
    ]

    if new_files:
        with st.spinner("Indexing…"):
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
            if errors:
                for err in errors:
                    st.error(err, icon="⚠️")

    # 3. Loaded documents list
    st.markdown(
        "<p style='color:#38bdf8;font-size:10px;font-weight:700;"
        "letter-spacing:2px;text-transform:uppercase;"
        "margin:14px 0 8px 0;'>Loaded documents</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.uploaded_files:
        for fname in st.session_state.uploaded_files:
            n_chunks = st.session_state.chunk_counts.get(fname, 0)
            # Truncate long names for display
            display_name = fname if len(fname) <= 28 else fname[:25] + "…"
            st.markdown(f"""
            <div style="
                background:#0f172a;
                border:0.5px solid #334155;
                border-radius:8px;
                padding:8px 12px;
                margin-bottom:6px;
                display:flex;
                align-items:center;
                gap:10px;
            ">
                <div style="
                    background:#dc2626;
                    color:white;
                    font-size:9px;
                    font-weight:700;
                    padding:3px 5px;
                    border-radius:4px;
                    flex-shrink:0;
                ">PDF</div>
                <div style="flex:1;min-width:0;">
                    <div style="
                        color:#cbd5e1;
                        font-size:12px;
                        white-space:nowrap;
                        overflow:hidden;
                        text-overflow:ellipsis;
                    ">{display_name}</div>
                    <div style="color:#64748b;font-size:10px;">{n_chunks} chunks indexed</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(
            "<p style='color:#475569;font-size:12px;margin-top:4px;"
            "font-style:italic;'>No documents loaded yet.</p>",
            unsafe_allow_html=True,
        )

    # 4. Clear button
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    if st.button("Clear all documents", key="clear_btn", use_container_width=True):
        st.session_state.faiss_index     = None
        st.session_state.chunks          = []
        st.session_state.chunk_counts    = {}
        st.session_state.uploaded_files  = []
        st.session_state.rag_ans         = None
        st.session_state.raw_ans         = None
        st.session_state.retrieved_chunks = []
        st.rerun()


# ════════════════════════════════════════════════
#  RIGHT — MAIN
# ════════════════════════════════════════════════
with col_main:

    # 1. Title bar with logo
    logo = _logo_html(height=28)
    logo_part = (
        f'<div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">'
        f'{logo}'
        f'</div>'
        f'<div style="width:1px;height:28px;background:#e5e7eb;flex-shrink:0;"></div>'
        if logo else ""
    )
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;
                padding-bottom:14px;border-bottom:2px solid #c7d2e0;">
        {logo_part}
        <span style="font-size:19px;font-weight:700;color:#0f172a;letter-spacing:-0.4px;">
            Remote Sensing RAG Demo
        </span>
        <span style="
            background:#ddd6fe;color:#4c1d95;
            font-size:11px;font-weight:700;
            padding:3px 11px;border-radius:10px;
            letter-spacing:0.3px;
        ">RAG vs LLM comparison</span>
    </div>
    """, unsafe_allow_html=True)

    # 2. Query input row
    q_col, btn_col = st.columns([5, 1], gap="small")
    with q_col:
        query = st.text_input(
            label="",
            placeholder="Ask a question about your documents…",
            label_visibility="collapsed",
            key="query",
        )
    with btn_col:
        ask_clicked = st.button("Ask →", type="primary", key="ask_btn", use_container_width=True)

    # Handle Ask click
    if ask_clicked:
        if not query.strip():
            st.warning("Please enter a question before asking.", icon="💬")
        elif st.session_state.faiss_index is None:
            st.warning("Upload at least one PDF before asking a question.", icon="📄")
        else:
            col_l, col_r = st.columns(2, gap="medium")
            with col_l:
                with st.spinner("Generating RAG answer…"):
                    rag_ans, retrieved = answer_rag(query)
                st.session_state.rag_ans          = rag_ans
                st.session_state.retrieved_chunks = retrieved
            with col_r:
                with st.spinner("Generating LLM answer…"):
                    raw_ans = get_llm_answer_bare(query)
                st.session_state.raw_ans = raw_ans

    # 3. Dual response panels
    has_response = st.session_state.rag_ans is not None

    if not has_response:
        # Empty state
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:36px;margin-bottom:12px;color:#94a3b8;">◎</div>
            <div style="font-size:15px;font-weight:600;color:#374151;margin-bottom:6px;">
                Ask a question to see the comparison
            </div>
            <div style="font-size:13px;color:#64748b;line-height:1.6;">
                The RAG answer and raw LLM answer will appear side by side
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_rag, col_llm = st.columns(2, gap="medium")

        # ── LEFT: RAG Answer ──
        with col_rag:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="
                    background:#d1fae5;color:#065f46;
                    font-size:10px;font-weight:700;
                    padding:3px 10px;border-radius:10px;
                    letter-spacing:.08em;text-transform:uppercase;
                    border:1px solid #6ee7b7;
                ">RAG Answer</span>
            </div>
            <p style="font-size:12px;color:#374151;font-weight:500;margin-bottom:12px;margin-top:0;">
                Grounded strictly in your uploaded documents
            </p>
            """, unsafe_allow_html=True)

            rag_text = st.session_state.rag_ans or ""
            st.markdown(f"""
            <div style="
                background:#f0fdf4;
                border:2.5px solid #059669;
                border-radius:10px;
                padding:18px 20px;
                font-size:14px;
                color:#064e3b;
                line-height:1.8;
                min-height:80px;
                box-shadow:0 2px 12px rgba(5,150,105,0.1);
            ">{rag_text}</div>
            """, unsafe_allow_html=True)

            # Context expander
            if st.session_state.retrieved_chunks:
                with st.expander("View retrieved context", expanded=False):
                    for i, chunk in enumerate(st.session_state.retrieved_chunks):
                        safe_chunk = chunk.replace("<", "&lt;").replace(">", "&gt;")
                        st.markdown(f"""
                        <div style="
                            background:#fffbeb;
                            border:0.5px solid #fde68a;
                            border-left:3px solid #f59e0b;
                            border-radius:6px;
                            padding:12px 14px;
                            margin-bottom:10px;
                            font-size:12px;
                            color:#78350f;
                            line-height:1.6;
                        "><strong style="color:#92400e;font-size:10px;
                          text-transform:uppercase;letter-spacing:.5px;">
                          Passage {i+1}</strong><br><br>{safe_chunk}</div>
                        """, unsafe_allow_html=True)

        # ── RIGHT: Bare LLM Answer ──
        with col_llm:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                <span style="
                    background:#ede9fe;color:#4c1d95;
                    font-size:10px;font-weight:700;
                    padding:3px 10px;border-radius:10px;
                    letter-spacing:.08em;text-transform:uppercase;
                    border:1px solid #c4b5fd;
                ">LLM Answer</span>
            </div>
            <p style="font-size:12px;color:#374151;font-weight:500;margin-bottom:12px;margin-top:0;">
                From model training only — no document context
            </p>
            """, unsafe_allow_html=True)

            raw_text = st.session_state.raw_ans or ""
            st.markdown(f"""
            <div style="
                background:#f5f3ff;
                border:2.5px solid #7c3aed;
                border-radius:10px;
                padding:18px 20px;
                font-size:14px;
                color:#2e1065;
                line-height:1.8;
                min-height:80px;
                box-shadow:0 2px 12px rgba(124,58,237,0.1);
            ">{raw_text}</div>
            """, unsafe_allow_html=True)

            # Hallucination warning
            st.markdown("""
            <div style="
                background:#fef2f2;
                border:0.5px solid #fecaca;
                border-radius:8px;
                padding:10px 14px;
                margin-top:10px;
                display:flex;
                gap:10px;
                align-items:flex-start;
            ">
                <div style="
                    width:8px;height:8px;
                    background:#ef4444;border-radius:50%;
                    flex-shrink:0;margin-top:4px;
                "></div>
                <span style="font-size:12px;color:#991b1b;line-height:1.5;">
                    This answer uses training data only. Specific facts (values,
                    names, dates) may be wrong or outdated. Compare with the RAG
                    answer for accuracy.
                </span>
            </div>
            """, unsafe_allow_html=True)
