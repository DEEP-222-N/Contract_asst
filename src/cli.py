"""CLI for the Contract Intelligence Assistant.

Usage:
    python -m src.cli query "Does this agreement allow sublicensing?"
    python -m src.cli eval --split train
    python -m src.cli eval --split test  --config configs/final.yaml
    python -m src.cli verify-offsets
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from .dataset import (
    load_queries,
    load_unanswerable,
    verify_offsets,
)
from .evaluation import evaluate, format_summary
from .pipeline import build_from_config

app = typer.Typer(add_completion=False, no_args_is_help=True)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = ROOT / "configs" / "baseline.yaml"


def _setup_log(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def query(
    text: str,
    config: Path = typer.Option(DEFAULT_CFG, help="Pipeline config YAML."),
    k: int = typer.Option(5, help="How many results to show."),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run a single query against the pipeline and print the top-k hits."""
    _setup_log(verbose)
    queries = load_queries()
    pipe = build_from_config(config, queries=queries)
    res = pipe.search(text)
    if res.abstained:
        typer.echo("No relevant clause found in the indexed contracts.")
        return
    for i, h in enumerate(res.hits[:k], start=1):
        typer.echo(f"\n[{i}] score={h.score:.3f}  doc={h.chunk.doc_id}  offsets={h.chunk.start}-{h.chunk.end}")
        typer.echo(h.chunk.text.strip()[:600])


@app.command()
def eval(
    split: str = typer.Option("train", help="train | test | all"),
    config: Path = typer.Option(DEFAULT_CFG),
    include_unanswerable: bool = typer.Option(True),
    out: Path = typer.Option(ROOT / "experiments" / "results" / "latest.json"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Evaluate the pipeline on a split and write a metrics JSON."""
    _setup_log(verbose)
    all_queries = load_queries()
    unanswerable = load_unanswerable() if include_unanswerable else []

    pool = all_queries + unanswerable
    if split != "all":
        pool = [q for q in pool if q.split == split]

    pipe = build_from_config(config, queries=all_queries)   # index the full referenced corpus
    metrics = evaluate(pipe, pool)

    typer.echo(f"\n--- {split} ---")
    typer.echo(format_summary(metrics))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": str(config),
        "split": split,
        "n_queries": metrics.n_queries,
        "n_answerable": metrics.n_answerable,
        "n_unanswerable": metrics.n_unanswerable,
        "mean_precision_at": metrics.mean_precision_at,
        "mean_recall_at": metrics.mean_recall_at,
        "correct_abstention_rate": metrics.correct_abstention_rate,
        "wrong_abstention_rate": metrics.wrong_abstention_rate,
        "per_query": [
            {
                "qid": m.qid,
                "query": m.query,
                "answerable": m.answerable,
                "abstained": m.abstained,
                "precision_at": m.precision_at,
                "recall_at": m.recall_at,
            }
            for m in metrics.per_query
        ],
    }, indent=2), encoding="utf-8")
    typer.echo(f"\nWrote {out}")


@app.command("verify-offsets")
def verify_offsets_cmd() -> None:
    """Check that each snippet's (start, end) matches the ground-truth answer text."""
    queries = load_queries()
    rep = verify_offsets(queries)
    typer.echo(f"ok={rep['ok']}/{rep['total']}  bad={len(rep['bad'])}")
    if rep["bad"]:
        for b in rep["bad"][:10]:
            typer.echo(json.dumps(b, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    app()
