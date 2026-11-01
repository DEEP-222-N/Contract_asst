"""Streamlit demo for the Contract Intelligence Assistant.

Run: streamlit run src/app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from .dataset import load_queries
from .pipeline import build_from_config

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Contract Intelligence", layout="wide")
st.title("Contract Intelligence Assistant")
st.caption("Clause search over CUAD contracts — retrieval-first, LLM-assisted.")

cfg_choice = st.sidebar.selectbox(
    "Pipeline config",
    options=sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("*.yaml")),
    index=0,
)
k_show = st.sidebar.slider("Top-k to show", 1, 10, 5)


@st.cache_resource(show_spinner="Loading pipeline…")
def _pipe(cfg: str):
    queries = load_queries()
    return build_from_config(ROOT / cfg, queries=queries)


pipe = _pipe(cfg_choice)
q = st.text_area("Ask a question about the indexed contracts", height=100,
                 placeholder="e.g. Does this agreement allow sublicensing to a third-party platform?")

if st.button("Search", type="primary") and q.strip():
    with st.spinner("Searching…"):
        res = pipe.search(q.strip())
    if res.abstained:
        st.warning("No clause in the indexed contracts answers this question.")
    else:
        for i, h in enumerate(res.hits[:k_show], start=1):
            with st.container(border=True):
                st.markdown(
                    f"**[{i}] {h.chunk.doc_id}** — score {h.score:.3f}  "
                    f"(offsets {h.chunk.start}–{h.chunk.end})"
                )
                st.text(h.chunk.text.strip())
