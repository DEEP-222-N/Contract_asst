"""Streamlit demo for the Contract Intelligence Assistant.

Run: streamlit run src/app.py
"""
from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

# Streamlit launches this as a top-level script, not as a package module, so
# relative imports fail. Add the project root to sys.path and use absolute
# imports instead.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env so GROQ_API_KEY is available when LLM toggles are on.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

import streamlit as st  # noqa: E402

from src.dataset import load_queries  # noqa: E402
from src.pipeline import build_from_config  # noqa: E402


st.set_page_config(
    page_title="Contract Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .hero-title {
        font-size: 2.1rem; font-weight: 700; margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #6b7280; font-size: 0.98rem; margin-bottom: 1.4rem;
    }
    .metric-card {
        background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 0.8rem 1rem; text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #111827; }
    .metric-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.5px; margin-top: 2px; }
    .result-rank {
        display: inline-block; width: 1.9rem; height: 1.9rem; line-height: 1.9rem;
        text-align: center; border-radius: 50%; background: #111827; color: #fff;
        font-weight: 700; font-size: 0.9rem; margin-right: 0.6rem;
    }
    .doc-pill {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        background: #eef2ff; color: #3730a3; font-size: 0.78rem; font-weight: 600;
        margin-right: 0.5rem;
    }
    .score-pill {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        background: #ecfdf5; color: #047857; font-size: 0.78rem; font-weight: 600;
        margin-right: 0.5rem; font-family: ui-monospace, monospace;
    }
    .offset-pill {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
        background: #f1f5f9; color: #475569; font-size: 0.75rem;
        font-family: ui-monospace, monospace;
    }
    .snippet {
        background: #ffffff; border-left: 3px solid #6366f1;
        padding: 0.85rem 1rem; margin-top: 0.6rem; line-height: 1.55;
        color: #1f2937; font-size: 0.93rem; border-radius: 4px;
        white-space: pre-wrap; word-wrap: break-word;
    }
    .snippet mark {
        background: #fef08a; color: #713f12; padding: 0 2px; border-radius: 2px;
    }
    .result-card {
        border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.2rem;
        margin-bottom: 0.9rem; background: #ffffff;
    }
    .result-card.top {
        border: 1.5px solid #6366f1; background: linear-gradient(0deg, #ffffff, #f5f3ff);
    }
    .example-btn button {
        text-align: left !important; font-size: 0.82rem !important;
        padding: 0.35rem 0.6rem !important; white-space: normal !important;
        height: auto !important;
    }
    @media (prefers-color-scheme: dark) {
        .metric-card { background: #1f2937; border-color: #374151; }
        .metric-value { color: #f9fafb; }
        .metric-label { color: #9ca3af; }
        .doc-pill { background: #312e81; color: #c7d2fe; }
        .score-pill { background: #064e3b; color: #a7f3d0; }
        .offset-pill { background: #334155; color: #cbd5e1; }
        .snippet { background: #111827; color: #e5e7eb; border-left-color: #818cf8; }
        .snippet mark { background: #78350f; color: #fef3c7; }
        .result-card { background: #111827; border-color: #374151; }
        .result-card.top { background: linear-gradient(0deg, #111827, #1e1b4b); border-color: #818cf8; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Contract Intelligence Assistant</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Clause search over 29 commercial contracts '
    '— hybrid retrieval (BM25 + semantic) with Reciprocal Rank Fusion, '
    'optional LLM rerank.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
cfg_choice = st.sidebar.selectbox(
    "Pipeline config",
    options=sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("*.yaml")),
    index=0,
    help="baseline.yaml = unmodified. final.yaml = after improvement loop (+39% R@4).",
)
k_show = st.sidebar.slider("Top-k results", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Try an example**")
EXAMPLES = [
    "Does this agreement allow sublicensing to a third-party platform?",
    "What is the exclusivity period and territory?",
    "Who owns the intellectual property in derivative works?",
    "What are the termination rights for material breach?",
    "What governing law applies to this contract?",
]

# Widget state must be initialised before the widget is created.
if "query_box" not in st.session_state:
    st.session_state.query_box = ""


def _use_example(text: str) -> None:
    # on_click runs BEFORE the text_area is re-rendered this run, so writing
    # to session_state.query_box here propagates into the widget cleanly.
    st.session_state.query_box = text


def _clear_query() -> None:
    st.session_state.query_box = ""


for i, ex in enumerate(EXAMPLES):
    st.sidebar.button(
        ex,
        key=f"ex_{i}",
        on_click=_use_example,
        args=(ex,),
        use_container_width=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**LLM assist** _(live toggles)_")
has_key = bool(os.environ.get("GROQ_API_KEY"))
llm_rewrite = st.sidebar.checkbox(
    "Query rewrite",
    value=False,
    disabled=not has_key,
    help="Groq rewrites the query before retrieval. Needs GROQ_API_KEY.",
)
llm_rerank = st.sidebar.checkbox(
    "LLM rerank top candidates",
    value=False,
    disabled=not has_key,
    help="Groq re-scores the top 20 chunks and keeps the best 5. Needs GROQ_API_KEY.",
)
if not has_key:
    st.sidebar.caption("GROQ_API_KEY not found in .env — LLM toggles disabled.")

st.sidebar.markdown("---")


# ---------------------------------------------------------------------------
# Pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pipeline and building indices...")
def _pipe(cfg: str):
    queries = load_queries()
    return build_from_config(ROOT / cfg, queries=queries)


pipe = _pipe(cfg_choice)

# Apply the sidebar's live toggles onto the (cached) pipeline config so the
# next pipe.search() call uses them. The underlying BM25/FAISS indices are
# unaffected — rewrite and rerank are search-time only.
pipe.cfg.setdefault("query_rewrite", {})["enabled"] = llm_rewrite
pipe.cfg.setdefault("reranker", {})["enabled"] = llm_rerank

# Pipeline info
n_docs = len({c.doc_id for c in pipe._chunks})
n_chunks = len(pipe._chunks)
mode = pipe.cfg["retrieval"]["mode"]
llm_on = llm_rewrite or llm_rerank

with st.sidebar.expander("Pipeline details", expanded=False):
    st.markdown(f"**Chunker:** {pipe.cfg['chunker']['strategy']} "
                f"(size={pipe.cfg['chunker']['size']}, overlap={pipe.cfg['chunker']['overlap']})")
    st.markdown(f"**Embedder:** `{pipe.cfg['embedder']['model_name']}`")
    st.markdown(f"**Retrieval:** {mode} (RRF k={pipe.cfg['retrieval'].get('rrf_k', 60)})")
    st.markdown(f"**Query rewrite:** {'on' if pipe.cfg.get('query_rewrite', {}).get('enabled') else 'off'}")
    st.markdown(f"**LLM rerank:** {'on' if pipe.cfg.get('reranker', {}).get('enabled') else 'off'}")
    st.markdown(f"**Abstention:** {pipe.cfg.get('abstention', {}).get('mode', 'none')}")

# ---------------------------------------------------------------------------
# Stats row
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{n_docs}</div>'
                f'<div class="metric-label">Contracts</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{n_chunks:,}</div>'
                f'<div class="metric-label">Chunks indexed</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{mode}</div>'
                f'<div class="metric-label">Retrieval</div></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">'
                f'{"on" if llm_on else "off"}</div>'
                f'<div class="metric-label">LLM assist</div></div>',
                unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------------------------------
# Query box
# ---------------------------------------------------------------------------
q = st.text_area(
    "Ask a question about the indexed contracts",
    height=90,
    placeholder="e.g. Does this agreement allow sublicensing to a third-party platform?",
    key="query_box",
)

col_a, col_b, _ = st.columns([1, 1, 4])
with col_a:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col_b:
    st.button("Clear", on_click=_clear_query, use_container_width=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from",
    "this", "that", "these", "those", "it", "its", "as", "any", "all",
    "does", "do", "did", "has", "have", "had", "will", "would", "should",
    "can", "could", "may", "might", "what", "which", "who", "whom", "whose",
    "when", "where", "why", "how", "there", "here", "between",
}


def _highlight(text: str, query: str) -> str:
    """HTML-escape text and <mark> the query's content words."""
    tokens = [t.lower() for t in re.findall(r"\w{3,}", query) if t.lower() not in _STOP]
    tokens = list(dict.fromkeys(tokens))[:10]
    escaped = html.escape(text)
    if not tokens:
        return escaped
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in tokens) + r")\b",
        flags=re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", escaped)


# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------
if search_clicked and q.strip():
    with st.spinner("Searching..."):
        res = pipe.search(q.strip())

    if res.rewritten_query and res.rewritten_query.strip() != q.strip():
        st.info(f"**LLM-rewritten query:** {res.rewritten_query}")

    if res.abstained:
        st.warning(
            "**No clause found.** The indexed contracts do not contain an answer "
            "to this question.",
            icon=None,
        )
    else:
        st.markdown(f"**{min(len(res.hits), k_show)} clauses retrieved** "
                    f"(of {len(res.hits)} ranked)")

        for i, h in enumerate(res.hits[:k_show], start=1):
            card_class = "result-card top" if i == 1 else "result-card"
            snippet_html = _highlight(h.chunk.text.strip(), q)
            doc_short = h.chunk.doc_id.rsplit("/", 1)[-1]

            st.markdown(
                f'<div class="{card_class}">'
                f'  <div style="margin-bottom:0.3rem">'
                f'    <span class="result-rank">{i}</span>'
                f'    <span class="doc-pill">{html.escape(doc_short)}</span>'
                f'    <span class="score-pill">score {h.score:.3f}</span>'
                f'    <span class="offset-pill">offsets {h.chunk.start}&ndash;{h.chunk.end}</span>'
                f'  </div>'
                f'  <div class="snippet">{snippet_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

elif search_clicked and not q.strip():
    st.warning("Enter a question above or pick an example from the sidebar.")
else:
    with st.expander("About this system", expanded=False):
        st.markdown(
            """
            **What it does.** Given a natural-language question, it retrieves the
            most relevant passages from a corpus of 29 commercial contracts
            (CUAD dataset), with document source and exact character offsets.
            If nothing in the corpus matches, it abstains instead of hallucinating.

            **How it works.**
            1. Query is optionally stripped of the CUAD "Consider the X Agreement" preamble.
            2. BM25 (keyword) and FAISS (semantic, `bge-small-en-v1.5`) each return top-20.
            3. **Reciprocal Rank Fusion** merges the two rankings.
            4. Optional LLM rerank / query rewrite — currently off in `final.yaml`
               (evaluation showed no improvement).

            **Measurement.** Primary metric is character-level recall@4 on the
            LegalBench-RAG CUAD-mini benchmark (train/test split by SHA-256).
            Final config reaches **R@4 = 0.096** on train (+39% relative over baseline),
            with guardrails on precision@4 and correct-abstention rate.

            **Why LLM is off by default.** The improvement loop evaluated LLM
            query rewriting in iteration 5 — it did not improve recall. The code
            path is wired and cached; flip `query_rewrite.enabled: true` in the
            config to re-enable.
            """
        )
