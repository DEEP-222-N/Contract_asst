"""Self-improvement loop.

Each iteration:
  1. Evaluates the current config on the TRAIN split only.
  2. Diagnoses failure modes (wrong doc / wrong section / partial coverage / abstention).
  3. Proposes one config change (rule-based now; LLM-driven if --use-llm).
  4. Evaluates the proposed config on train.
  5. Accepts the change iff the primary metric improves AND no guardrail degrades
     by more than `guardrail_tolerance`.
  6. Appends the whole run to experiments/iteration_log.jsonl.

Usage:
    python -m src.improvement_loop run --iterations 5
    python -m src.improvement_loop run --iterations 5 --use-llm
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
import yaml

from .config import load_config
from .dataset import load_queries, load_unanswerable
from .evaluation import SuiteMetrics, evaluate, format_summary
from .pipeline import Pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs"
LOG_PATH = ROOT / "experiments" / "iteration_log.jsonl"

PRIMARY_METRIC = "recall_at_4"
GUARDRAIL_METRICS = ("precision_at_4", "correct_abstention_rate")
GUARDRAIL_TOLERANCE = 0.02   # may drop by at most 2 absolute points


# ---------------------------------------------------------------------------
# Metric extraction / summary.
# ---------------------------------------------------------------------------
def flat_metrics(m: SuiteMetrics) -> dict:
    out: dict = {}
    for k, v in m.mean_precision_at.items():
        out[f"precision_at_{k}"] = v
    for k, v in m.mean_recall_at.items():
        out[f"recall_at_{k}"] = v
    out["correct_abstention_rate"] = m.correct_abstention_rate
    out["wrong_abstention_rate"] = m.wrong_abstention_rate
    return out


# ---------------------------------------------------------------------------
# Failure diagnosis.
# ---------------------------------------------------------------------------
@dataclass
class FailureSummary:
    n_wrong_doc: int
    n_right_doc_wrong_section: int
    n_partial_cover: int
    n_wrong_abstention: int

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose(pipe: Pipeline, metrics: SuiteMetrics, queries) -> FailureSummary:
    """Classify failed answerable queries into coarse buckets."""
    from .dataset import Query
    queries_by_qid = {q.qid: q for q in queries}
    wrong_doc = 0
    wrong_section = 0
    partial = 0
    wrong_abs = 0
    for m in metrics.per_query:
        if not m.answerable:
            continue
        if m.abstained:
            wrong_abs += 1
            continue
        q: Query = queries_by_qid[m.qid]
        gt_docs = {s.file_path for s in q.snippets}
        res = pipe.search(q.query)
        top_docs = {h.chunk.doc_id for h in res.hits[:4]}
        hit_doc = bool(gt_docs & top_docs)
        p4 = m.precision_at.get(4, 0.0)
        r4 = m.recall_at.get(4, 0.0)
        if not hit_doc:
            wrong_doc += 1
        elif r4 == 0.0:
            wrong_section += 1
        elif r4 < 0.5:
            partial += 1
    return FailureSummary(
        n_wrong_doc=wrong_doc,
        n_right_doc_wrong_section=wrong_section,
        n_partial_cover=partial,
        n_wrong_abstention=wrong_abs,
    )


# ---------------------------------------------------------------------------
# Rule-based proposer.
#
# We keep an ORDERED candidate queue instead of modulo rules so each iteration
# tries a *different* change. The queue is tuned for CUAD + hybrid retrieval:
# the most impactful levers come first.
# ---------------------------------------------------------------------------
def _candidates(cfg: dict, fs: FailureSummary) -> list[tuple[dict, str, str]]:
    """List (patch, change_desc, rationale) ordered by expected impact."""
    import copy

    def patched(spec: dict, change: str, reason: str) -> tuple[dict, str, str]:
        new = copy.deepcopy(cfg)
        for dotted, val in spec.items():
            keys = dotted.split(".")
            node = new
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = val
        return new, change, reason

    out: list[tuple[dict, str, str]] = []

    # 1. Strip the CUAD query preamble. Highest expected impact on this corpus:
    #    the preamble pulls ranking toward the document header, not the clause.
    if not cfg.get("query_preprocess", {}).get("strip_consider_prefix"):
        out.append(patched(
            {"query_preprocess.strip_consider_prefix": True},
            "query_preprocess.strip_consider_prefix -> true",
            "Diagnosis shows top-1 chunk is often the document preamble. Strip the "
            "'Consider the X Agreement between Y and Z;' prefix so BM25/FAISS score "
            "the actual question.",
        ))

    # 2. Enlarge chunks so the answer isn't split across windows.
    cur_size = int(cfg["chunker"]["size"])
    if cur_size < 1200:
        new_size = int(cur_size * 1.6)
        new_overlap = min(int(cfg["chunker"]["overlap"] * 1.6), new_size // 2 - 1)
        out.append(patched(
            {"chunker.size": new_size, "chunker.overlap": new_overlap},
            f"chunker.size -> {new_size}, overlap -> {new_overlap}",
            "Right doc but wrong section is a leading failure. Larger windows mean "
            "the answer span is less often split across chunks.",
        ))

    # 3. Return more neighbouring chunks so a partially-covered span gets its
    #    tail or head picked up at a slightly deeper rank.
    cur_tk = int(cfg["retrieval"]["top_k_final"])
    if cur_tk < 16:
        new_tk = cur_tk + 4
        new_tr = max(cfg["retrieval"]["top_k_per_retriever"], new_tk * 2)
        out.append(patched(
            {"retrieval.top_k_final": new_tk, "retrieval.top_k_per_retriever": new_tr},
            f"retrieval.top_k_final -> {new_tk}",
            "Partial coverage is a leading failure. Pulling more neighbouring chunks "
            "lets the scorer union more of the gold span.",
        ))

    # 4. Switch to sentence-aware chunking so section headings line up with
    #    sentence boundaries (fewer mid-clause splits).
    if cfg["chunker"].get("strategy") != "sentence":
        out.append(patched(
            {"chunker.strategy": "sentence"},
            "chunker.strategy -> sentence",
            "Fixed-size windows split through sentences and definitions. Sentence "
            "packing keeps clauses intact.",
        ))

    # 5. Steepen RRF so top-of-list evidence dominates.
    cur_rrf = int(cfg["retrieval"].get("rrf_k", 60))
    if cur_rrf > 20:
        new_rrf = max(20, int(cur_rrf * 0.5))
        out.append(patched(
            {"retrieval.rrf_k": new_rrf},
            f"retrieval.rrf_k -> {new_rrf}",
            "Lower RRF k steepens the fusion curve: top ranks in each retriever "
            "contribute more, bottom ranks contribute less.",
        ))

    # 6. Enable LLM query rewriting (requires GROQ_API_KEY).
    if not cfg.get("query_rewrite", {}).get("enabled"):
        out.append(patched(
            {"query_rewrite.enabled": True},
            "query_rewrite.enabled -> true",
            "Legal queries are long and compositional. Have the LLM extract the "
            "retrieval-relevant core before scoring.",
        ))

    # 7. Enable LLM reranker on top 20 -> top 5.
    if not cfg.get("reranker", {}).get("enabled"):
        out.append(patched(
            {"reranker.enabled": True},
            "reranker.enabled -> true",
            "RRF treats chunks as independent passages. An LLM reranker can read "
            "clause context and score candidates on how directly they answer the question.",
        ))

    # 8. Tune abstention threshold up (if wrong abstentions on answerables are rare,
    #    we can afford to abstain more confidently on noise).
    if fs.n_wrong_abstention <= 2:
        cur = float(cfg["abstention"].get("min_top_score", 0.0))
        new_th = round(cur + 0.004, 4)
        new_mode = "score_and_gap"
        out.append(patched(
            {"abstention.mode": new_mode, "abstention.min_top_score": new_th},
            f"abstention: mode -> score_and_gap, min_top_score -> {new_th}",
            "Current threshold is permissive and lets through low-score noise. "
            "Nudge it up to improve correct-abstention on unanswerables.",
        ))

    return out


def propose_rule_based(cfg: dict, fs: FailureSummary, step: int) -> tuple[dict, str, str]:
    """Pick the step-th still-applicable candidate. 1-indexed step."""
    cs = _candidates(cfg, fs)
    if not cs:
        # Nothing left to try — repeat a safe no-op so the loop logs the decision.
        import copy
        return copy.deepcopy(cfg), "noop", "No more candidates in the proposer queue."
    idx = (step - 1) % len(cs)
    return cs[idx]


# ---------------------------------------------------------------------------
# Optional LLM proposer.
# ---------------------------------------------------------------------------
def propose_llm(cfg: dict, fs: FailureSummary) -> tuple[dict, str, str]:
    import copy

    from .llm.groq_client import GroqLLM
    llm = GroqLLM()
    sys_msg = (
        "You are tuning a retrieval pipeline. Given a YAML config and a failure "
        "summary, propose ONE small change (chunker size, retrieval top_k, rrf_k, "
        "or abstention threshold). Return STRICT JSON: "
        '{"patch": {<dotted.key>: <value>}, "reason": "..."}. No prose.'
    )
    user = (
        f"CONFIG:\n{yaml.safe_dump(cfg, sort_keys=False)}\n\n"
        f"FAILURES:\n{json.dumps(fs.to_dict())}"
    )
    raw = llm.chat(sys_msg, user, max_tokens=256)
    try:
        obj = json.loads(raw)
        patch = obj["patch"]
        reason = obj.get("reason", "LLM-proposed change")
    except Exception:
        return propose_rule_based(cfg, fs, step=0)
    new = copy.deepcopy(cfg)
    changes: list[str] = []
    for dotted, val in patch.items():
        keys = dotted.split(".")
        node = new
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
        changes.append(f"{dotted} -> {val}")
    return new, "; ".join(changes), reason


# ---------------------------------------------------------------------------
# One loop iteration.
# ---------------------------------------------------------------------------
def run_iteration(step: int, cfg: dict, use_llm: bool, train_queries, metrics_before: dict):
    pipe_before = Pipeline(cfg)
    pipe_before.build(train_queries)
    fs = diagnose(pipe_before, _rebuild_suite(pipe_before, train_queries), train_queries)

    if use_llm:
        new_cfg, change, reason = propose_llm(cfg, fs)
        proposer = "groq_llm_analysis"
    else:
        new_cfg, change, reason = propose_rule_based(cfg, fs, step)
        proposer = "manual_rule"

    pipe_after = Pipeline(new_cfg)
    pipe_after.build(train_queries)
    m_after = _rebuild_suite(pipe_after, train_queries)
    metrics_after = flat_metrics(m_after)

    decision, verdict = _decide(metrics_before, metrics_after)
    return {
        "iteration": step,
        "proposer": proposer,
        "change": change,
        "rationale": reason,
        "failures_before": fs.to_dict(),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "decision": decision,
        "verdict": verdict,
        "config_after": new_cfg,
    }


def _rebuild_suite(pipe: Pipeline, queries) -> SuiteMetrics:
    return evaluate(pipe, queries)


def _decide(before: dict, after: dict) -> tuple[str, str]:
    primary_delta = after[PRIMARY_METRIC] - before[PRIMARY_METRIC]
    if primary_delta <= 0:
        return "rejected", f"primary {PRIMARY_METRIC} did not improve (Δ={primary_delta:+.4f})"
    for g in GUARDRAIL_METRICS:
        if (before[g] - after[g]) > GUARDRAIL_TOLERANCE:
            return "rejected", f"guardrail {g} dropped by {before[g]-after[g]:+.4f} > {GUARDRAIL_TOLERANCE}"
    return "accepted", f"primary {PRIMARY_METRIC} +{primary_delta:.4f}; guardrails ok"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
@app.command()
def run(
    iterations: int = 5,
    base_config: Path = CFG_DIR / "baseline.yaml",
    use_llm: bool = typer.Option(False, "--use-llm"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_queries = load_queries()
    train_queries = [q for q in all_queries if q.split == "train"]
    unanswerable_train = [q for q in load_unanswerable() if q.split == "train"]
    train_pool = train_queries + unanswerable_train

    cfg = load_config(base_config)
    pipe = Pipeline(cfg)
    pipe.build(all_queries)
    m0 = evaluate(pipe, train_pool)
    typer.echo("\nBASELINE on train:\n" + format_summary(m0))
    metrics_before = flat_metrics(m0)
    _log({"iteration": 0, "proposer": "baseline", "metrics_after": metrics_before,
          "decision": "accepted", "config_after": cfg})

    current_cfg = cfg
    accepted_cfg = cfg
    for step in range(1, iterations + 1):
        t0 = time.time()
        entry = run_iteration(step, current_cfg, use_llm, train_pool, metrics_before)
        entry["elapsed_s"] = time.time() - t0
        _log(entry)
        typer.echo(
            f"\n[iter {step}] {entry['decision']}: {entry['change']}  "
            f"({entry['verdict']})"
        )
        if entry["decision"] == "accepted":
            current_cfg = entry["config_after"]
            metrics_before = entry["metrics_after"]
            accepted_cfg = current_cfg

    final_path = CFG_DIR / "final.yaml"
    final_path.write_text(yaml.safe_dump(accepted_cfg, sort_keys=False), encoding="utf-8")
    typer.echo(f"\nFinal config written to {final_path}")


def _log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    app()
