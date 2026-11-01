"""BM25 keyword retrieval over chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ..chunker import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class BM25Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, top_k: int = 20) -> list[BM25Hit]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        # argsort descending, take top_k
        order = scores.argsort()[::-1][:top_k]
        return [BM25Hit(chunk=self.chunks[i], score=float(scores[i])) for i in order]
