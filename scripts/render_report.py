"""Render the iteration log and the latest train/test eval JSONs into a
compact markdown block for REPORT.md.

Usage:
    python scripts/render_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments" / "iteration_log.jsonl"
RESULTS = ROOT / "experiments" / "results"


def _fmt_metrics(d: dict) -> str:
    return (
        f"P@1={d.get('precision_at_1', 0):.3f} "
        f"P@4={d.get('precision_at_4', 0):.3f} "
        f"R@4={d.get('recall_at_4', 0):.3f} "
        f"R@8={d.get('recall_at_8', 0):.3f} "
        f"corrAbs={d.get('correct_abstention_rate', 0):.3f} "
        f"wrAbs={d.get('wrong_abstention_rate', 0):.3f}"
    )


def main() -> None:
    print("## Iteration log\n")
    print("| # | proposer | change | decision | metrics_after |")
    print("| - | - | - | - | - |")
    for line in LOG.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        after = e.get("metrics_after", {})
        print(
            f"| {e.get('iteration', '?')} | {e.get('proposer', '?')} | "
            f"{e.get('change', '')} | {e.get('decision', '')} | "
            f"{_fmt_metrics(after)} |"
        )

    print("\n## Results tables\n")
    for name in ("baseline_train", "final_train", "final_test"):
        p = RESULTS / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"### {name}  —  n={d['n_queries']} (ans={d['n_answerable']}, un={d['n_unanswerable']})")
        print(_fmt_metrics({
            **{f"precision_at_{k}": v for k, v in d["mean_precision_at"].items()},
            **{f"recall_at_{k}": v for k, v in d["mean_recall_at"].items()},
            "correct_abstention_rate": d["correct_abstention_rate"],
            "wrong_abstention_rate": d["wrong_abstention_rate"],
        }))
        print()


if __name__ == "__main__":
    main()
