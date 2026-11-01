"""Document chunking strategies that keep character offsets honest."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document.

    `text == document[start:end]` (half-open interval). The scorer relies on
    this invariant to compute character-level precision/recall.
    """
    doc_id: str          # the document's file_path, e.g. "cuad/foo.txt"
    chunk_id: str        # f"{doc_id}:{start}-{end}" — unique
    text: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Fixed-size chunking with character-level overlap.
# Simple, deterministic, and leaves offsets exact.
# ---------------------------------------------------------------------------
def chunk_fixed_tokens(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    chunks: list[Chunk] = []
    step = size - overlap
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}:{i}-{j}",
            text=text[i:j],
            start=i,
            end=j,
        ))
        if j == n:
            break
        i += step
    return chunks


# ---------------------------------------------------------------------------
# Sentence-ish chunking: group sentences up to `size` characters.
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_sentence(doc_id: str, text: str, size: int, overlap: int = 0) -> list[Chunk]:
    """Pack sentences into windows of up to `size` chars.

    Keeps exact offsets by splitting on whitespace between sentences and tracking
    cursor position in the original text.
    """
    chunks: list[Chunk] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        end = min(cursor + size, n)
        # Try to end on a sentence boundary within [cursor, end] if we're not at EOF.
        if end < n:
            # look backwards for a terminator
            tail = text[cursor:end]
            m = list(_SENT_SPLIT.finditer(tail))
            if m:
                end = cursor + m[-1].end()
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}:{cursor}-{end}",
            text=text[cursor:end],
            start=cursor,
            end=end,
        ))
        if end >= n:
            break
        cursor = max(cursor + 1, end - overlap)
    return chunks


# ---------------------------------------------------------------------------
# Dispatcher.
# ---------------------------------------------------------------------------
def chunk_document(doc_id: str, text: str, cfg: dict) -> list[Chunk]:
    strategy = cfg.get("strategy", "fixed_tokens")
    size = int(cfg.get("size", 500))
    overlap = int(cfg.get("overlap", 100))
    if strategy == "fixed_tokens":
        return chunk_fixed_tokens(doc_id, text, size, overlap)
    if strategy == "sentence":
        return chunk_sentence(doc_id, text, size, overlap)
    raise ValueError(f"unknown chunker strategy: {strategy}")
