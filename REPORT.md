# Evaluation Report — Contract Intelligence Assistant

> This report is the primary deliverable. Numbers below come from the test
> split and the committed experiment log; the README covers "how to run it".

## Summary

_To be filled with headline test-set numbers once the improvement loop lands.
Shape: "{precision@4} / {recall@4} on {n_test} queries after {k} iterations,
with correct abstention {x}% / wrong abstention {y}%."_

## System design

- **Dense + sparse hybrid.** Legal-contract queries mix domain-specific
  terminology ("sublicensing", "quiet-enjoyment") and compositional phrasing
  ("the clause that lets us…"). Pure BM25 misses paraphrases; pure dense
  misses rare legal n-grams and party names. Reciprocal Rank Fusion is the
  simplest fusion that doesn't require score calibration between the two.
- **Chunking: character windows with overlap.** Fixed-size windows (500 chars,
  100 overlap) are the baseline because they are deterministic, keep offsets
  exact, and give the evaluator predictable spans. Clause-aware chunking is
  listed as a candidate for the improvement loop.
- **Abstention: score + gap, off at baseline.** Baseline measures retrieval
  without abstention; the loop then tunes `min_top_score` and `min_gap` on
  train.

**Alternatives considered.**
- A single `all-MiniLM-L6-v2` encoder (smaller/faster, lower CUAD recall in
  the LegalBench-RAG paper).
- Cross-encoder reranker (`ms-marco-MiniLM`): local, no network cost, but
  adds a GPU-vs-CPU trade-off and dependency weight; we picked LLM reranking
  because it also gives us a channel for failure analysis.

## LLM use

- **Model:** Groq-hosted `openai/gpt-oss-20b` (configurable via
  `GROQ_MODEL`).
- **Where:** optional query rewriting; optional top-20 → top-5 reranking;
  failure-analysis proposer in the self-improvement loop.
- **Why Groq:** free tier is sufficient for the ~hundreds of calls the
  evaluation makes, latency is low (<1 s per call), and no local GPU is
  needed. For a real JioStar deployment, see "Limitations" — contracts are
  confidential, so this would move to a local Qwen or similar.
- All LLM responses are cached under `cache/llm/` keyed by
  `(model, system, user, temperature)` so the evaluation can be re-run
  without re-billing.

## Evaluation setup

- **Benchmark queries:** 194 CUAD queries from LegalBench-RAG-mini, selected
  by the authors' `mini()` function. Train = 140 / test = 54 by the SHA-256
  mod-10 rule from the brief.
- **Unanswerable queries:** 35 hand-written, committed before the
  improvement loop started (`data/unanswerable.jsonl`, train = 23 / test = 12).
- **Metrics:** character-level precision/recall at k ∈ {1, 2, 4, 8}, computed
  per the LegalBench-RAG paper (union of retrieved intervals vs. union of
  ground-truth intervals, within the same document). Abstention measured in
  both directions: `correct_abstention_rate` on unanswerable queries,
  `wrong_abstention_rate` on answerable ones.
- Development happens on train only. The test split was run **N times**
  _(to be reported: ideally once at the very end)_.

## Results

_Filled in after the loop finishes._

| config | split | n | P@1 | P@4 | R@4 | R@8 | correct_abs | wrong_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | train | 163 |  |  |  |  |  |  |
| final | train | 163 |  |  |  |  |  |  |
| **final** | **test** | 66 |  |  |  |  |  |  |

## Iteration history

**Primary metric:** `recall@4` on the train pool (answerable queries).
**Guardrails:** `precision@4` and `correct_abstention_rate` must not drop by
more than 0.02 absolute.

The loop runs with `python -m src.improvement_loop run`. On each iteration it:

1. Re-indexes the corpus with the current config.
2. Classifies failed answerable queries into {wrong doc, right doc wrong
   section, partial coverage, wrong abstention}.
3. Proposes one config change (rule-based by default; `--use-llm` routes the
   proposal through Groq).
4. Re-evaluates on train.
5. Accepts the change iff primary improves and guardrails hold.
6. Appends the full outcome to `experiments/iteration_log.jsonl`.

_Table of iterations fills in after the loop runs._

## Error analysis

_At least 10 concrete failures, grouped by cause, to be filled in after the
loop finishes. Each entry: query, retrieved vs. ground truth, our diagnosis._

## Limitations and next steps

- **Hosted LLM vs. confidential contracts.** The brief's own JioStar scenario
  precludes sending real contract text to Groq. A production deployment would
  swap `src/llm/groq_client.py` for a local Qwen-via-vLLM or Ollama endpoint
  behind the same `chat()` interface.
- **Chunking is content-blind.** Character windows split through definitions
  and across cross-references ("as defined in Section 4.2 above"). A
  clause-aware chunker that respects article/section boundaries is the most
  promising next step; it was listed as a stretch goal and is not shipped.
- **Only 29 contracts indexed.** The brief directs us to index only the
  contracts referenced by the mini queries. Scaling to the full 462-file
  CUAD corpus would require re-tuning abstention thresholds against a much
  noisier top-k.
- **Reranker ceiling.** The LLM reranker treats candidate chunks as
  independent passages; it does not resolve cross-references. A clause-graph
  that follows "as defined in …" pointers would almost certainly beat it on
  the error categories labelled "right doc, wrong section".
- **No end-to-end user study.** This system is evaluated against benchmark
  spans. Real legal users would also care about calibration ("how sure are
  you?") and traceability from clause → contract snapshot; neither is
  surfaced in the current UI.
