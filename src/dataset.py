"""Load the CUAD subset of LegalBench-RAG-mini."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_DIR = DATA_DIR / "corpus" / "cuad"
BENCHMARK_PATH = DATA_DIR / "benchmarks" / "cuad.json"
UNANSWERABLE_PATH = DATA_DIR / "unanswerable.jsonl"


@dataclass(frozen=True)
class Snippet:
    """One ground-truth answer span in a specific document."""
    file_path: str                  # relative path inside corpus (e.g. "cuad/foo.txt")
    span: tuple[int, int]           # (start, end), half-open, over utf-8 text
    answer: str                     # verbatim text of document[start:end]


@dataclass(frozen=True)
class Query:
    """One benchmark query, which may have 1+ answer snippets."""
    qid: str                        # stable id derived from the query text
    query: str
    snippets: tuple[Snippet, ...]
    split: str                      # "train" | "test"
    answerable: bool = True


# ---------------------------------------------------------------------------
# Mini selection (exact function from the assignment brief).
# ---------------------------------------------------------------------------
def mini(tests: list[dict], n: int = 194) -> list[dict]:
    if len(tests) <= n:
        return tests

    def key(t: dict) -> float:
        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


# ---------------------------------------------------------------------------
# Train / test split (exact function from the assignment brief; DO NOT CHANGE).
# ---------------------------------------------------------------------------
def assign_split(query: str) -> str:
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"


def qid_for(query: str) -> str:
    """Stable 12-char id derived from the query text."""
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
def _normalize_file_path(fp: str) -> str:
    """The benchmark's file_path field is like 'cuad/XYZ.txt'. Keep it as-is
    so we can resolve against the corpus dir in load_document()."""
    return fp.replace("\\", "/")


def load_queries(
    benchmark_path: Path = BENCHMARK_PATH,
    use_mini: bool = True,
) -> list[Query]:
    """Load benchmark queries from cuad.json, with the mini filter + split applied."""
    raw = benchmark_path.read_text(encoding="utf-8")
    tests = json.loads(raw)["tests"]
    if use_mini:
        tests = mini(tests)

    out: list[Query] = []
    for t in tests:
        q = t["query"]
        snippets = tuple(
            Snippet(
                file_path=_normalize_file_path(s["file_path"]),
                span=(int(s["span"][0]), int(s["span"][1])),
                answer=s["answer"],
            )
            for s in t["snippets"]
        )
        out.append(
            Query(
                qid=qid_for(q),
                query=q,
                snippets=snippets,
                split=assign_split(q),
                answerable=True,
            )
        )
    return out


def load_unanswerable(path: Path = UNANSWERABLE_PATH) -> list[Query]:
    """Load the hand-written unanswerable queries. One JSON object per line."""
    if not path.exists():
        return []
    out: list[Query] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        obj = json.loads(line)
        q = obj["query"]
        out.append(
            Query(
                qid=qid_for(q),
                query=q,
                snippets=(),
                split=assign_split(q),
                answerable=False,
            )
        )
    return out


def referenced_documents(queries: Iterable[Query]) -> list[str]:
    """Unique document file paths referenced by any snippet in the given queries."""
    seen: set[str] = set()
    for q in queries:
        for s in q.snippets:
            seen.add(s.file_path)
    return sorted(seen)


def load_document(file_path: str) -> str:
    """Read a corpus document in the exact mode the benchmark offsets assume
    (Python default text mode, UTF-8)."""
    # file_path is like "cuad/XYZ.txt"; our corpus is at data/corpus/cuad/XYZ.txt
    rel = file_path.split("/", 1)[1] if file_path.startswith("cuad/") else file_path
    return (CORPUS_DIR / rel).read_text(encoding="utf-8")


def verify_offsets(queries: Iterable[Query]) -> dict:
    """Check that document[start:end] == snippet.answer for every snippet."""
    ok = 0
    bad: list[dict] = []
    cache: dict[str, str] = {}
    for q in queries:
        for s in q.snippets:
            doc = cache.get(s.file_path)
            if doc is None:
                try:
                    doc = load_document(s.file_path)
                except FileNotFoundError:
                    bad.append({"qid": q.qid, "file": s.file_path, "reason": "missing_file"})
                    continue
                cache[s.file_path] = doc
            got = doc[s.span[0]: s.span[1]]
            if got == s.answer:
                ok += 1
            else:
                bad.append({
                    "qid": q.qid,
                    "file": s.file_path,
                    "span": s.span,
                    "expected_prefix": s.answer[:80],
                    "got_prefix": got[:80],
                })
    return {"ok": ok, "bad": bad, "total": ok + len(bad)}
