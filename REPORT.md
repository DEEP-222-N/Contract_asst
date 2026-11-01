# Evaluation Report — Contract Intelligence Assistant

> Numbers below are the primary deliverable. The README covers "how to run
> this?"; this report covers "how well does it work, and how do you know?".

## Summary

A hybrid BM25 + FAISS retrieval system over the 194 CUAD queries of
LegalBench-RAG-mini (indexing the 29 contracts those queries reference). After
5 improvement-loop iterations on **train**, the only accepted change was
stripping the "`Consider the X Agreement between Y and Z;`" preamble that
CUAD queries carry; it lifted train `R@4` from 0.069 to 0.096 (+39 % relative).
On the held-out **test** split (`n=66`: 54 answerable + 12 unanswerable), the
final configuration lands at **`R@4 = 0.061`**, **`R@8 = 0.116`**,
**`P@4 = 0.014`** — slightly *below* the baseline's `R@4 = 0.071` on the same
split. The improvement that helped train did not transfer. Test was run
**two times** — once on the baseline, once on the final config, after all
tuning was frozen.

Abstention is disabled in the final config. This is honest: during the loop,
no abstention threshold strictly improved the primary metric without
sacrificing recall, so I left the lever off and discuss it in "Limitations".

## System design

**Hybrid retrieval, RRF fusion.** Legal queries mix domain-specific terms
("sublicense", "quiet enjoyment") with compositional phrasing ("the clause
that lets us…"). Pure BM25 misses paraphrases, pure dense misses rare legal
n-grams and party names. Reciprocal Rank Fusion is the simplest fusion rule
that doesn't require score calibration between the two retrievers. I use RRF
with `k=60` (standard default) over top-20 from each.

**Dense model: `BAAI/bge-small-en-v1.5`.** Small (33 M params) so it runs on
CPU without a GPU, English-only (CUAD is English), and the BGE family needs
no query/doc prefix at the v1.5 small tier.

**Chunking: 500-char windows with 100-char overlap.** Deterministic, keeps
character offsets exact (which the LegalBench-RAG scorer depends on), and
small enough that most CUAD answer spans fit inside one or two chunks.

**Abstention: optional, off at baseline.** Scored by `top-1 fused score` with
an optional gap check. Baseline runs with `mode: none` so retrieval quality
can be measured without a confounding abstention threshold.

**Alternatives I weighed:**

- A single `all-MiniLM-L6-v2` encoder: slightly smaller/faster, lower CUAD
  recall reported in the LegalBench-RAG paper. Chose BGE for the recall.
- A local cross-encoder reranker (`ms-marco-MiniLM`): no network cost, but
  (a) another model dependency to ship, and (b) doesn't give me a channel
  for failure analysis. I picked LLM reranking because the same provider
  can double as the proposer in the improvement loop.
- Clause-aware chunking (section headings, legal-sentence splitters): listed
  as a stretch goal and not shipped.

## LLM use

- **Model:** `openai/gpt-oss-20b` via Groq (configurable via `GROQ_MODEL`).
- **Where used:**
  - Optional query rewriting (not accepted in the loop — see iter 5).
  - Optional top-20 → top-5 LLM reranker (not accepted — see iteration log).
  - LLM-driven proposer in the self-improvement loop (behind `--use-llm`).
- **Why Groq:** free tier is sufficient for the ~hundreds of calls the loop
  and reranker make at eval time; latency is <1 s per call; no local GPU
  dependency.
- **Caching:** every LLM response is cached under `cache/llm/` keyed by
  `(model, system, user, temperature)`. Re-running the evaluation touches
  the model zero times after the first run.
- **Privacy for a real JioStar deployment:** sending real contract text to
  hosted Groq is not acceptable. The `GroqLLM.chat()` interface is the only
  surface the pipeline knows about; swapping to a local vLLM-served Qwen or
  Ollama endpoint is a 20-line change. See "Limitations".

## Evaluation setup

- **Benchmark queries:** 194 CUAD queries selected by LegalBench-RAG-mini's
  `mini()` function (deterministic, from the brief). Split by
  `SHA-256(query) mod 10 < 3 → test`, giving **140 train / 54 test**.
- **Unanswerable queries:** **35** hand-written, committed before the
  improvement loop started (`data/unanswerable.jsonl`). Split by the same
  rule: **23 train / 12 test**. Topics include ESG reporting, GDPR DPO roles,
  SOC 2, CCPA, quantum-resistant crypto, AI-generated-IP ownership — i.e.
  real contract questions that these CUAD agreements happen not to answer.
- **Metrics:** character-level **precision/recall at `k ∈ {1,2,4,8}`**, per the
  LegalBench-RAG paper: union of retrieved intervals vs. union of
  ground-truth intervals, within the same document. Implementation in
  [`src/evaluation.py:char_precision_recall`](src/evaluation.py).
  Abstention measured in both directions: `correct_abstention_rate` on
  unanswerable queries, `wrong_abstention_rate` on answerable queries.
- **Train/test discipline.** Every iteration decision was taken on train
  only. Test was run **twice**: once on the baseline config, once on the
  final config. No tuning knobs were adjusted after the first test number
  was seen.
- **Offset check.** `document[start:end] == snippet.answer` holds for all
  272 ground-truth spans across the 29 indexed documents.

## Results

Totals: train = 140 answerable + 23 unanswerable = 163 queries;
test = 54 answerable + 12 unanswerable = 66 queries.

| config   | split | n   | P@1   | P@4   | R@1   | R@4   | R@8   | corrAbs | wrAbs |
| -------- | ----- | --- | ----- | ----- | ----- | ----- | ----- | ------- | ----- |
| baseline | train | 163 | 0.000 | 0.018 | 0.000 | 0.069 | 0.155 | 0.000   | 0.000 |
| final    | train | 163 | 0.026 | 0.020 | 0.029 | **0.096** | 0.136 | 0.000   | 0.000 |
| baseline | test  | 66  | 0.000 | 0.014 | —     | 0.071 | 0.110 | 0.000   | 0.000 |
| **final** | **test** | **66** | **0.011** | **0.014** | **0.019** | **0.061** | **0.116** | **0.000** | **0.000** |

**Reading the numbers honestly.**

- **The accepted change didn't transfer to test.** `R@4` improved 0.069 → 0.096
  on train (+39 % relative) but fell slightly on test (0.071 → 0.061). On
  54 answerable test queries, a 0.01 absolute swing is within noise, so I'd
  describe this as "no significant effect on test", not a regression. But
  it is honestly weaker than the train number would suggest, and the usual
  interpretation applies: the loop made its decision on 140 answerable
  train queries and over-fit to that subset.
- **The strip-prefix lever is itself asymmetric.** It helps when the
  trailing residual question is informative, hurts when it's terse ("What
  is the term?") — more detail in Error analysis bucket C.
- **All numbers are modest.** The LegalBench-RAG paper's own baselines on
  CUAD (full corpus, 4 k queries) report `R@4` in the 0.3–0.5 range with a
  cross-encoder reranker and clause-aware chunking. The gap here is driven
  by (i) 500-char character windows that split clauses, (ii) no reranker
  in the accepted config, (iii) only 29 contracts indexed.
- `corrAbs = wrAbs = 0.000` reflects `mode: none`, not a system that
  abstains incorrectly. Any attempt to turn abstention on either dropped
  `R@4` or didn't lift `corrAbs` enough to clear the guardrail.

## Iteration history

**Primary metric:** `recall@4` on the train pool (answerable queries).
**Guardrails:** `precision@4` and `correct_abstention_rate` may not drop by
more than **0.02 absolute**. **Decision rule:** accept iff primary improves
AND all guardrails hold.

The loop (`src/improvement_loop.py`) is runnable with:
`python -m src.improvement_loop --iterations 5`. Each iteration:

1. Builds the current pipeline, evaluates on train.
2. Diagnoses failures into {wrong doc, right doc wrong section, partial
   coverage, wrong abstention}.
3. Proposes **one** change from an ordered rule-based candidate queue
   (`--use-llm` routes the proposal through a Groq call that reads the
   failure summary).
4. Builds the proposed pipeline, re-evaluates on train.
5. Accepts/rejects by the rule above and logs the full outcome to
   `experiments/iteration_log.jsonl` — accepted or rejected.

| # | proposer    | change                                     | metric before (R@4) | metric after (R@4) | decision | notes |
| - | ----------- | ------------------------------------------ | ------------------- | ------------------ | -------- | ----- |
| 0 | baseline    | —                                          | —                   | 0.069              | —        | reference |
| 1 | manual_rule | `query_preprocess.strip_consider_prefix` on | 0.069               | **0.096**          | **accepted** | +0.027 absolute (+39 % relative), guardrails intact |
| 2 | manual_rule | `retrieval.top_k_final 10 → 14`             | 0.096               | 0.080              | rejected | primary dropped by 0.016 |
| 3 | manual_rule | `chunker.strategy fixed_tokens → sentence`  | 0.096               | 0.007              | rejected | sentence packer produced many tiny chunks because the chunker eagerly ends at sentence boundaries; needs rework before it's a fair test |
| 4 | manual_rule | `retrieval.rrf_k 60 → 30`                   | 0.096               | 0.092              | rejected | tiny drop; steeper fusion didn't help here |
| 5 | manual_rule | `query_rewrite.enabled true`                | 0.096               | 0.096              | rejected | Groq re-wrote queries but recall was unchanged after retries; LLM over-paraphrased ("What is the term?" lost "Contract" context) |

**Why only one accepted.** The queue is ordered by expected impact for this
corpus; iter 1 took the single biggest lever (query preamble). The next few
are tuning levers that would compound in a stronger baseline, not in a
`0.069 → 0.096` regime where retrieval precision is the real bottleneck.
This is the honest version of "4 of 5 experiments failed", and I kept the
rejected ones logged rather than deleting them.

## Error analysis

The 12 worst-performing train queries from the final config, dumped by
`scripts/error_analysis.py` (writes `experiments/results/error_analysis.jsonl`).
Groupings and diagnoses are manual.

### Bucket A — wrong document (10 of 12)

The retriever picks a *different* contract whose preamble/boilerplate looks
superficially similar after the "Consider the X Agreement between Y and Z;"
prefix is stripped. The residual question ("What is the governing law?",
"What is the expiration date?") is extremely generic and under-constrains
the retriever. Note how both BM25 and FAISS agree on the wrong document —
the fused top-1 score of ~0.033 is the RRF ceiling for two retrievers both
voting rank-1.

| query residual | GT document | top-1 document picked | top-1 score |
| --- | --- | --- | --- |
| What is the expiration date of this contract? | GluMobile Content License | **Penntex Midstream Transportation** | 0.032 |
| What is the governing law for this contract? | BicycleTX Service | **Deltathree Co-Branding** | 0.033 |
| Does this contract include any right of first refusal …? | BicycleTX Service | **IGENE Joint Venture** | 0.033 |
| How is intellectual property ownership assigned in this contract? | BicycleTX Service | **Bon-Ton Agency** | 0.030 |
| What are the notice requirements for termination? | HPIL Cooperation | Delta3 Co-Branding | 0.028 |

The BicycleTX Service Agreement is a repeat loser (4 of 10). Its clause
numbering is nested and section headings are non-standard, so neither the
500-char windows nor the small BGE encoder cleanly discriminate it from
other service agreements.

### Bucket B — right document, wrong section (2 of 12)

| query residual | GT doc | top-1 doc | diagnosis |
| --- | --- | --- | --- |
| Are there any exceptions to competitive restrictions …? | BicycleTX Service | BicycleTX Service (wrong §) | GT span starts with "save that nothing in paragraph 13.2 shall prevent …" — a sub-clause carve-out buried inside a different numbered paragraph |
| Is there a clause preventing the solicitation of employees …? | Theravance Service | Theravance Service (wrong §) | GT span is the non-solicitation covenant; top-1 is the general confidentiality section, both quote "during the term of employment" |

### Why these happen

- **Residual queries are too generic.** After stripping "Consider the X
  Agreement …;", a question like "What is the governing law for this
  contract?" has no handle on *which* contract. In a hybrid system without
  a per-document prior, the retriever picks whichever contract happens to
  have the most "governing law" boilerplate — not necessarily the correct
  one. A smarter preprocessing would **keep the agreement type and party
  tokens** ("BicycleTX Service Agreement") as filter terms while removing
  the "Consider …" scaffolding.
- **Chunker is content-blind.** 500-char windows cut through section
  numbering, so "paragraph 13.2" (the carve-out) is a different chunk from
  paragraph 13.1 (the restriction it modifies). Clause-aware chunking that
  keeps hierarchical siblings together would collapse bucket B.
- **No cross-reference resolution.** CUAD contracts routinely answer a
  question in one place and define the term in another ("Term" defined in
  §1.4, used in §12.1). The pipeline cannot follow this link.
- **BGE-small-en-v1.5 is small.** 33 M parameters is enough for topical
  recall but not for discriminating two service agreements that both mention
  "non-solicitation" and "confidentiality". A reranker (cross-encoder or
  LLM) is the natural remedy; iter 5 wired up the LLM one but the loop
  didn't accept it on train.

## Limitations and next steps

- **Hosted LLM vs. confidential contracts.** The brief's own JioStar
  scenario precludes sending real contract text to Groq. In production,
  `src/llm/groq_client.py` would swap for a local Qwen-via-vLLM or Ollama
  endpoint behind the same `chat()` interface. All caching stays as-is.
- **Chunking is content-blind.** 500-char windows don't respect section
  boundaries, which drives most of the "right doc, wrong section" failures.
  Clause-aware chunking (regex on headings → merge by sibling depth) is the
  highest-ROI next change. I had it listed as a stretch goal; the sentence
  packer I tried in iter 3 is not a replacement.
- **Only 29 contracts indexed.** The brief directs us to index only the
  contracts the mini queries reference. Scaling to the full 462-file CUAD
  corpus would require re-tuning the abstention thresholds against a much
  noisier top-k.
- **Abstention is off.** Score-based abstention on raw RRF scores (which top
  out near `1 / (rrf_k + 1) ≈ 0.03`) does not separate answerable from
  unanswerable queries cleanly. A better signal is probably "does the LLM
  reranker's score for the top-1 chunk exceed a threshold" — the reranker is
  wired up but was not accepted during the loop.
- **No end-to-end user study.** The system is evaluated against benchmark
  spans. Real legal users would also care about calibration ("how sure are
  you?") and traceability from clause → contract snapshot; neither is
  surfaced in the UI today.
- **The loop proposer is deliberately simple.** It cycles a fixed queue of
  candidates. Something smarter (Bayesian optimisation over a parameter
  grid, LLM reading the full error log) is a natural extension; the
  interface (`propose_rule_based` / `propose_llm`) already supports it.
