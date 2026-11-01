"""Character-level precision/recall at k, plus abstention metrics.

Follows LegalBench-RAG: for each query, compare the union of character
intervals in the top-k retrieved chunks against the union of ground-truth
answer intervals, scoped to the same document.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .dataset import Query
from .pipeline import Pipeline
from .retrieval.hybrid import Hit


# ---------------------------------------------------------------------------
# Interval helpers.
# ---------------------------------------------------------------------------
Interval = tuple[str, int, int]  # (doc_id, start, end)


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    s = sorted(intervals)
    out = [s[0]]
    for a, b in s[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _length(intervals: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in intervals)


def _overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Sum of intersection lengths between two sorted, merged interval lists."""
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def char_precision_recall(
    retrieved: list[Hit],
    gt_snippets: list[tuple[str, int, int]],
    k: int,
) -> tuple[float, float]:
    """Character-level precision & recall over the top-k retrieved chunks.

    Precision = (chars retrieved that overlap any GT span) / (total chars retrieved)
    Recall    = (GT chars covered by retrieved spans)      / (total GT chars)
    """
    top = retrieved[:k]
    # bucket retrieved intervals by doc
    retr_by_doc: dict[str, list[tuple[int, int]]] = {}
    for h in top:
        retr_by_doc.setdefault(h.chunk.doc_id, []).append((h.chunk.start, h.chunk.end))
    retr_by_doc = {d: _merge(iv) for d, iv in retr_by_doc.items()}

    gt_by_doc: dict[str, list[tuple[int, int]]] = {}
    for d, s, e in gt_snippets:
        gt_by_doc.setdefault(d, []).append((s, e))
    gt_by_doc = {d: _merge(iv) for d, iv in gt_by_doc.items()}

    retr_total = sum(_length(iv) for iv in retr_by_doc.values())
    gt_total = sum(_length(iv) for iv in gt_by_doc.values())

    overlap_total = 0
    for d, iv in retr_by_doc.items():
        if d in gt_by_doc:
            overlap_total += _overlap(iv, gt_by_doc[d])

    p = overlap_total / retr_total if retr_total else 0.0
    r = overlap_total / gt_total if gt_total else 0.0
    return p, r


# ---------------------------------------------------------------------------
# Suite-level metrics.
# ---------------------------------------------------------------------------
@dataclass
class QueryMetrics:
    qid: str
    query: str
    answerable: bool
    abstained: bool
    precision_at: dict[int, float] = field(default_factory=dict)
    recall_at: dict[int, float] = field(default_factory=dict)


@dataclass
class SuiteMetrics:
    n_queries: int
    n_answerable: int
    n_unanswerable: int
    mean_precision_at: dict[int, float]
    mean_recall_at: dict[int, float]
    correct_abstention_rate: float   # on unanswerable queries
    wrong_abstention_rate: float     # on answerable queries
    per_query: list[QueryMetrics]


KS = (1, 2, 4, 8)


def evaluate(
    pipeline: Pipeline,
    queries: Iterable[Query],
    ks: tuple[int, ...] = KS,
) -> SuiteMetrics:
    queries = list(queries)
    per_q: list[QueryMetrics] = []
    sums_p: dict[int, float] = {k: 0.0 for k in ks}
    sums_r: dict[int, float] = {k: 0.0 for k in ks}
    n_ans = 0
    n_un = 0
    correct_abstain = 0
    wrong_abstain = 0

    for q in queries:
        res = pipeline.search(q.query)
        m = QueryMetrics(
            qid=q.qid,
            query=q.query,
            answerable=q.answerable,
            abstained=res.abstained,
        )
        if q.answerable:
            n_ans += 1
            gt = [(s.file_path, s.span[0], s.span[1]) for s in q.snippets]
            for k in ks:
                p, r = char_precision_recall(res.hits, gt, k=k)
                m.precision_at[k] = p
                m.recall_at[k] = r
                sums_p[k] += p
                sums_r[k] += r
            if res.abstained:
                wrong_abstain += 1
        else:
            n_un += 1
            if res.abstained:
                correct_abstain += 1
        per_q.append(m)

    return SuiteMetrics(
        n_queries=len(queries),
        n_answerable=n_ans,
        n_unanswerable=n_un,
        mean_precision_at={k: (sums_p[k] / n_ans if n_ans else 0.0) for k in ks},
        mean_recall_at={k: (sums_r[k] / n_ans if n_ans else 0.0) for k in ks},
        correct_abstention_rate=(correct_abstain / n_un) if n_un else 0.0,
        wrong_abstention_rate=(wrong_abstain / n_ans) if n_ans else 0.0,
        per_query=per_q,
    )


def format_summary(m: SuiteMetrics) -> str:
    lines = [
        f"queries: {m.n_queries}  (answerable {m.n_answerable}, unanswerable {m.n_unanswerable})",
        "k    precision   recall",
    ]
    for k in sorted(m.mean_precision_at):
        lines.append(f"{k:>2}   {m.mean_precision_at[k]:.3f}       {m.mean_recall_at[k]:.3f}")
    lines.append(f"correct abstention (unanswerable): {m.correct_abstention_rate:.3f}")
    lines.append(f"wrong abstention   (answerable):   {m.wrong_abstention_rate:.3f}")
    return "\n".join(lines)
