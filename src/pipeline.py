"""End-to-end retrieval pipeline, driven by a YAML config."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .abstention import should_abstain
from .chunker import Chunk, chunk_document
from .dataset import Query, load_document, load_queries, referenced_documents
from .embeddings import Embedder
from .retrieval.bm25 import BM25Index
from .retrieval.faiss_store import FaissIndex
from .retrieval.hybrid import Hit, rrf

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]            # empty iff abstained
    abstained: bool
    rewritten_query: str | None = None


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._chunks: list[Chunk] = []
        self._bm25: BM25Index | None = None
        self._faiss: FaissIndex | None = None
        self._embedder: Embedder | None = None
        self._llm = None
        self._rewriter_llm = None

    # ------------------------------------------------------------------
    # Build the indices from the corpus referenced by the given queries.
    # ------------------------------------------------------------------
    def build(self, queries: list[Query]) -> None:
        docs = referenced_documents(queries)
        log.info("Building indices over %d documents", len(docs))

        self._chunks = []
        for d in docs:
            text = load_document(d)
            self._chunks.extend(chunk_document(d, text, self.cfg["chunker"]))
        log.info("Produced %d chunks", len(self._chunks))

        mode = self.cfg["retrieval"]["mode"]
        if mode in ("bm25", "hybrid"):
            self._bm25 = BM25Index(self._chunks)
        if mode in ("faiss", "hybrid"):
            self._embedder = Embedder(
                model_name=self.cfg["embedder"]["model_name"],
                batch_size=int(self.cfg["embedder"].get("batch_size", 32)),
                normalize=bool(self.cfg["embedder"].get("normalize", True)),
            )
            self._faiss = FaissIndex(self._chunks, self._embedder)

    # ------------------------------------------------------------------
    # Serve one query.
    # ------------------------------------------------------------------
    def search(self, query: str) -> RetrievalResult:
        q = query

        # Optional pre-processing: strip the CUAD-style preamble
        # "Consider the X Agreement between Y and Z; <actual question>".
        qp = self.cfg.get("query_preprocess", {}) or {}
        if qp.get("strip_consider_prefix"):
            if ";" in q and q.lstrip().lower().startswith("consider "):
                q = q.split(";", 1)[1].strip()

        rewritten = None
        if self.cfg.get("query_rewrite", {}).get("enabled"):
            from .llm.groq_client import GroqLLM
            from .llm.rewriter import rewrite
            if self._rewriter_llm is None:
                self._rewriter_llm = GroqLLM(model=self.cfg["query_rewrite"].get("model"))
            rewritten = rewrite(q, llm=self._rewriter_llm)
            q = rewritten or q

        mode = self.cfg["retrieval"]["mode"]
        tk = int(self.cfg["retrieval"]["top_k_per_retriever"])
        tk_final = int(self.cfg["retrieval"]["top_k_final"])

        if mode == "bm25":
            raw = self._bm25.search(q, top_k=tk)
            hits = [Hit(chunk=h.chunk, score=h.score, bm25_rank=i + 1, bm25_score=h.score)
                    for i, h in enumerate(raw[:tk_final])]
        elif mode == "faiss":
            raw = self._faiss.search(q, top_k=tk)
            hits = [Hit(chunk=h.chunk, score=h.score, faiss_rank=i + 1, faiss_score=h.score)
                    for i, h in enumerate(raw[:tk_final])]
        elif mode == "hybrid":
            bm25_hits = self._bm25.search(q, top_k=tk)
            faiss_hits = self._faiss.search(q, top_k=tk)
            hits = rrf(
                bm25_hits,
                faiss_hits,
                k=int(self.cfg["retrieval"].get("rrf_k", 60)),
                top_k=tk_final,
            )
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        # Optional LLM rerank.
        if self.cfg.get("reranker", {}).get("enabled") and hits:
            from .llm.groq_client import GroqLLM
            from .llm.reranker import rerank
            if self._llm is None:
                self._llm = GroqLLM(model=self.cfg["reranker"].get("model"))
            top_in = int(self.cfg["reranker"].get("top_k_in", 20))
            top_out = int(self.cfg["reranker"].get("top_k_out", 5))
            rr = rerank(q, hits[:top_in], top_k_out=top_out, llm=self._llm)
            hits = rr.hits

        abstained = should_abstain(hits, self.cfg.get("abstention", {}))
        if abstained:
            return RetrievalResult(query=query, hits=[], abstained=True, rewritten_query=rewritten)
        return RetrievalResult(query=query, hits=hits, abstained=False, rewritten_query=rewritten)


# ---------------------------------------------------------------------------
# Convenience constructor.
# ---------------------------------------------------------------------------
def build_from_config(cfg_path: str | Path, queries: list[Query] | None = None) -> Pipeline:
    from .config import load_config
    cfg = load_config(cfg_path)
    pipe = Pipeline(cfg)
    if queries is None:
        queries = load_queries()
    pipe.build(queries)
    return pipe
