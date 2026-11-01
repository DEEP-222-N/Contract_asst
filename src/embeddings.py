"""Dense embeddings via sentence-transformers (cached on disk)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "embeddings"


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 32, normalize: bool = True):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, texts: list[str], tag: str) -> Path:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(tag.encode())
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
        return CACHE_DIR / f"{h.hexdigest()[:16]}.npy"

    def encode(self, texts: list[str], tag: str = "", use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if use_cache:
            cp = self._cache_path(texts, tag)
            if cp.exists():
                return np.load(cp)
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 128,
        ).astype(np.float32)
        if use_cache:
            np.save(cp, vecs)
        return vecs
