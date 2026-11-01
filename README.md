# Contract Intelligence Assistant

Clause search over legal contracts, built on the [LegalBench-RAG](https://arxiv.org/abs/2408.10343)
CUAD benchmark. For a natural-language question, returns a ranked list of
contract passages that answer it (with document name and exact character
offsets), or abstains when no clause in the corpus is a good enough match.

> **Full evaluation and iteration history: [REPORT.md](REPORT.md).**
> The README is the "how do I run this?" document.

## Overview

- Two interfaces: a Streamlit **web app** and a **CLI**.
- Retrieval-first system: dense (FAISS over `BAAI/bge-small-en-v1.5` embeddings)
  + sparse (BM25) fused with Reciprocal Rank Fusion.
- An optional Groq-hosted LLM is used only for *query rewriting* and *reranking*
  — never to generate clause text. All returned passages are verbatim from the
  source contracts.
- A single-command **self-improvement loop** diagnoses failures, proposes
  config changes and accepts/rejects them against a declared primary metric
  with guardrails.

## Setup

```bash
# Python 3.11+
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate

pip install -r requirements.txt

# Download the LegalBench-RAG CUAD files (benchmark + corpus the 194 mini
# queries reference). ~32 MB; the Python API is used because the hf CLI
# chokes on glob patterns on Windows.
python scripts/download_dataset.py

# Groq (only needed for LLM-assisted configs — baseline runs without it).
cp .env.example .env
# edit .env and paste GROQ_API_KEY=...
```

## How to run

```bash
# One-off search from the CLI
python -m src.cli query "Does this agreement allow sublicensing to a third-party platform?"

# Streamlit UI
streamlit run src/app.py

# Full evaluation (train / test / all)
python -m src.cli eval --split train
python -m src.cli eval --split test  --config configs/final.yaml

# Self-improvement loop (5 iterations by default; add --use-llm for the
# Groq-driven proposer)
python -m src.improvement_loop --iterations 5

# Tests
python -m pytest
```

## Architecture

```
query → [query rewrite?]  → BM25  ╮
                           FAISS ─┤→ RRF fusion → [LLM rerank?] → abstention → results
```

| Module | Role |
| --- | --- |
| `src/dataset.py` | Loads the 194 mini queries, SHA-256 train/test split, offset verification |
| `src/chunker.py` | Fixed-size and sentence chunkers; chunk offsets stay exact |
| `src/embeddings.py` | sentence-transformers encoder with on-disk cache |
| `src/retrieval/bm25.py` | BM25 keyword index |
| `src/retrieval/faiss_store.py` | FAISS flat IP index over normalized embeddings |
| `src/retrieval/hybrid.py` | Reciprocal Rank Fusion |
| `src/abstention.py` | Score + score-gap abstention rules |
| `src/llm/*` | Groq client (cached), query rewriter, reranker |
| `src/pipeline.py` | Config-driven assembly of the above |
| `src/evaluation.py` | Char-level P@k, R@k, abstention metrics (LegalBench-RAG definitions) |
| `src/improvement_loop.py` | Diagnose → propose → evaluate → accept/reject |
| `src/cli.py` | typer CLI |
| `src/app.py` | Streamlit UI |

All pipeline knobs live in YAML under `configs/`. Baseline and the loop's final
accepted config live there as `baseline.yaml` and `final.yaml`.

## Reproducing results

See [REPORT.md](REPORT.md) for the numbers.

```bash
python -m src.cli eval --split train --config configs/baseline.yaml   # baseline train
python -m src.improvement_loop --iterations 5                         # 5 iterations
python -m src.cli eval --split train --config configs/final.yaml      # final train
python -m src.cli eval --split test  --config configs/final.yaml      # final test
```

Rough wall-clock on a laptop CPU: chunk encoding ~2 min (cached after first
run), each eval pass ~1 min.

## Assumptions

Documented in order of impact.

1. **"Index only the contracts the 194 queries refer to."** The brief says this
   explicitly, so we index 29 documents out of the 462-file CUAD corpus. The
   mini queries do not reference the other 433 files.
2. **Character offsets are UTF-8 default-text-mode offsets.** We read every
   corpus file with `Path.read_text(encoding="utf-8")` to match. Offset check
   (272/272 snippets) passes on this assumption.
3. **The train/test split is immutable.** We never call `assign_split()` with
   different arguments. All development uses the train portions of both the
   benchmark queries (140) and our unanswerable set (23). The test portions
   (54 / 12) are reserved for the single final-reporting run.
4. **Abstention off at baseline.** We first measure retrieval quality
   unconstrained, then tune abstention in the improvement loop. The baseline
   therefore has a wrong-abstention rate of 0 by construction; correct-abstention
   rate reflects the untuned rule.
5. **LLM is strictly auxiliary.** The LLM is only invoked for query rewriting
   or reranking candidate scoring. It never produces clause text; the output
   span is always a verbatim slice of the source document.
6. **Hosted Groq is acceptable for the take-home.** In production the brief's
   scenario (confidential JioStar agreements) would require a local
   deployment (vLLM / Ollama + Qwen) — see "Limitations" in REPORT.md.

## AI tool usage

- **Claude (via Claude Code)** was used throughout for scaffolding code, drafting
  the unanswerable query set, drafting documentation, and discussing design
  trade-offs.
- **Groq** (hosted inference) is used at runtime for optional query rewriting
  and reranking. See REPORT.md § "LLM use" for the justification and the
  measured impact on metrics.

Every file in this repository has been read by me, and the design choices
behind the retrieval, abstention, and loop code are my own. I am ready to walk
through and modify any part of the code live in the debrief.
