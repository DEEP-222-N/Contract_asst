"""Optional: rerank top candidates with a Groq LLM.

The LLM only *scores* candidate chunks — it never generates clause text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..retrieval.hybrid import Hit
from .groq_client import GroqLLM

SYSTEM = (
    "You are a legal retrieval judge. Given a user question and candidate contract "
    "passages, score each candidate 0-10 for how directly it ANSWERS the question. "
    "Return STRICT JSON like {\"scores\": [8, 2, 6, ...]} with one score per candidate, "
    "in the same order. No prose."
)

_NUM_LIST = re.compile(r"\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\]")


@dataclass
class RerankResult:
    hits: list[Hit]
    raw: str


def rerank(
    query: str,
    hits: list[Hit],
    top_k_out: int,
    llm: GroqLLM | None = None,
) -> RerankResult:
    if not hits:
        return RerankResult(hits=[], raw="")
    llm = llm or GroqLLM()

    numbered = "\n\n".join(
        f"[{i}] (doc={h.chunk.doc_id}, offsets={h.chunk.start}-{h.chunk.end})\n{h.chunk.text[:800]}"
        for i, h in enumerate(hits)
    )
    user = f"Question:\n{query}\n\nCandidates:\n{numbered}"
    raw = llm.chat(SYSTEM, user, max_tokens=512)
    scores = _parse_scores(raw, len(hits))

    reranked = sorted(
        zip(hits, scores),
        key=lambda t: t[1],
        reverse=True,
    )
    out: list[Hit] = []
    for h, s in reranked[:top_k_out]:
        # overwrite fused score with normalized LLM score so downstream
        # abstention sees a comparable value in [0, 1].
        new = Hit(
            chunk=h.chunk,
            score=float(s) / 10.0,
            bm25_rank=h.bm25_rank,
            faiss_rank=h.faiss_rank,
            bm25_score=h.bm25_score,
            faiss_score=h.faiss_score,
        )
        out.append(new)
    return RerankResult(hits=out, raw=raw)


def _parse_scores(raw: str, n: int) -> list[float]:
    """Try strict JSON, then regex fallback. Pad/truncate to n."""
    try:
        obj = json.loads(raw)
        scores = obj.get("scores") if isinstance(obj, dict) else obj
        if isinstance(scores, list):
            vals = [float(x) for x in scores]
        else:
            raise ValueError
    except Exception:
        m = _NUM_LIST.search(raw)
        if not m:
            return [0.0] * n
        vals = [float(x) for x in m.group(0).strip("[]").split(",")]
    if len(vals) < n:
        vals = vals + [0.0] * (n - len(vals))
    return vals[:n]
