"""Dump the 10 worst train-split queries for the final config.

Writes experiments/results/error_analysis.jsonl — one JSON object per failure
with the query, GT snippet file + answer, top-5 retrieved chunks, and a
coarse diagnosis bucket.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.dataset import load_queries  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "final.yaml"
OUT = ROOT / "experiments" / "results" / "error_analysis.jsonl"


def diagnose_bucket(hits, gt_docs: set[str], recall_at_4: float) -> str:
    top_docs = {h.chunk.doc_id for h in hits[:4]}
    if not (gt_docs & top_docs):
        return "wrong_document"
    if recall_at_4 == 0.0:
        return "right_doc_wrong_section"
    if recall_at_4 < 0.3:
        return "partial_coverage"
    return "ok"


def main() -> None:
    from src.evaluation import char_precision_recall

    cfg = load_config(CFG)
    all_qs = load_queries()
    pipe = Pipeline(cfg)
    pipe.build(all_qs)

    train = [q for q in all_qs if q.split == "train"]

    scored = []
    for q in train:
        res = pipe.search(q.query)
        gt = [(s.file_path, s.span[0], s.span[1]) for s in q.snippets]
        _, r4 = char_precision_recall(res.hits, gt, k=4)
        scored.append((r4, q, res))

    scored.sort(key=lambda t: t[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r4, q, res in scored[:12]:
            gt_docs = {s.file_path for s in q.snippets}
            bucket = diagnose_bucket(res.hits, gt_docs, r4)
            f.write(json.dumps({
                "qid": q.qid,
                "query": q.query,
                "gt_doc": q.snippets[0].file_path,
                "gt_span": [q.snippets[0].span[0], q.snippets[0].span[1]],
                "gt_answer_prefix": q.snippets[0].answer[:160],
                "recall_at_4": r4,
                "diagnosis": bucket,
                "top_5": [
                    {
                        "doc": h.chunk.doc_id,
                        "offsets": [h.chunk.start, h.chunk.end],
                        "score": round(h.score, 4),
                        "text_prefix": h.chunk.text[:160],
                    }
                    for h in res.hits[:5]
                ],
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
