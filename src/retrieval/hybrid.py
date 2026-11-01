"""Reciprocal Rank Fusion of BM25 and FAISS results."""
from __future__ import annotations

from dataclasses import dataclass

from ..chunker import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused score (RRF or passthrough)
    bm25_rank: int | None = None
    faiss_rank: int | None = None
    bm25_score: float | None = None
    faiss_score: float | None = None


def rrf(
    bm25_hits: list,
    faiss_hits: list,
    k: int = 60,
    top_k: int = 10,
) -> list[Hit]:
    """Reciprocal Rank Fusion.

    score(c) = sum over retrievers of 1 / (k + rank(c)).
    Chunks appearing in only one retriever still get a (lower) score.
    """
    by_chunk: dict[str, Hit] = {}

    for rank, h in enumerate(bm25_hits):
        chunk = h.chunk
        existing = by_chunk.get(chunk.chunk_id)
        if existing is None:
            by_chunk[chunk.chunk_id] = Hit(
                chunk=chunk,
                score=1.0 / (k + rank + 1),
                bm25_rank=rank + 1,
                bm25_score=h.score,
            )
        else:
            existing.score += 1.0 / (k + rank + 1)
            existing.bm25_rank = rank + 1
            existing.bm25_score = h.score

    for rank, h in enumerate(faiss_hits):
        chunk = h.chunk
        existing = by_chunk.get(chunk.chunk_id)
        if existing is None:
            by_chunk[chunk.chunk_id] = Hit(
                chunk=chunk,
                score=1.0 / (k + rank + 1),
                faiss_rank=rank + 1,
                faiss_score=h.score,
            )
        else:
            existing.score += 1.0 / (k + rank + 1)
            existing.faiss_rank = rank + 1
            existing.faiss_score = h.score

    out = sorted(by_chunk.values(), key=lambda h: h.score, reverse=True)
    return out[:top_k]
