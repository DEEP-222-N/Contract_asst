"""FAISS dense retrieval over chunks."""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from ..chunker import Chunk
from ..embeddings import Embedder


@dataclass
class FaissHit:
    chunk: Chunk
    score: float


class FaissIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        vecs = embedder.encode([c.text for c in chunks], tag="chunks")
        # cosine similarity via normalized inner product
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, top_k: int = 20) -> list[FaissHit]:
        q = self.embedder.encode([query], tag="query", use_cache=False)
        D, I = self.index.search(q, top_k)
        hits: list[FaissHit] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            hits.append(FaissHit(chunk=self.chunks[idx], score=float(score)))
        return hits
