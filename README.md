# financebench-eval-harness

Evaluation harness for RAG and long-context LLMs on FinanceBench. It tracks retrieval quality, answer correctness, and evidence grounding as separate signals — you need all three to tell whether the model failed because retrieval missed the evidence or because it reasoned poorly over what it found. Seven pipeline stages, 24 CLI subcommands, every run snapshots its full config.

Python ≥ 3.10 | FAISS (CPU) | Ollama | 54 modules | 60 frozen dataclasses | 887 tests

---

## The Problem This Solves

Financial QA is harder than it looks. The questions aren't "summarize this paragraph" — they're things like "what was the net revenue in fiscal 2018, in millions?" That requires finding page 47 of a 200-page annual report, normalizing the unit, confirming the fiscal year isn't calendar year, and not confusing the number with a nearby figure that looks right. A model that scores fine on general-purpose QA often fails here, because the correct answer is a specific number, not a fluent sentence.

String-match scoring doesn't work: "1,577.0" and "$1.577 billion" are the same answer, but a naive metric marks them wrong. LLM judges have their own failure modes. And if you evaluate retrieval and generation as one thing, you lose the diagnostic — when the model fails, you can't tell if retrieval missed the evidence or if the model reasoned badly over what it found. The harness tracks them separately and correlates the two in a 2×2 matrix.

Reproducibility is its own problem. Change a prompt, swap an embedding model, or adjust chunk size, and prior results stop being comparable. Every run snapshots the full YAML config and validates corpus hash consistency across embedding comparisons — so when hit rates go up, you know it's the embedding model, not a difference in how the documents got chunked.

---

## Architecture and Pipeline

```
FinanceBench PDFs + questions.jsonl
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 0 · Data Ingestion                          │
│  extract-documents  →  pages.jsonl                 │
│  build-examples     →  examples.jsonl              │
│  (validate-data / validate-dataset /               │
│   validate-documents / validate-evidence-pages)    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1 · Chunking + Indexing                     │
│  chunk-documents   Recursive text split            │
│                    800 chars / 150 char overlap    │
│  build-index       FAISS IndexFlatL2               │
│                    → index.faiss                   │
│                    → index_metadata.json           │
│                      (corpus_hash, model, dim)     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2 · Retrieval                               │
│  retrieve          Dense top-k (L2 distance)       │
│                    → retrieval_results.jsonl       │
│  eval-retrieval    doc_hit@k, page_hit@k,          │
│                    evidence_text_hit@k, MRR        │
│                    → retrieval_scores.jsonl        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3 · RAG Generation                          │
│  run-rag           Retrieved chunks → LLM          │
│  (rag_dense | rag_oracle | rag_no_context)         │
│                    → rag_predictions.jsonl         │
│                    → run_metadata.json             │
│                    → config.yaml  (snapshot)       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4 · Scoring  (re-runnable independently)    │
│  score-rag         Lexical + answer judge +        │
│                    grounding judge                 │
│                    → rag_answer_scores.jsonl       │
│                    → rag_grounding_scores.jsonl    │
│                    → rag_combined_scores.jsonl     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5 · Metric Joining                          │
│  join-metrics      Retrieval hit × answer correct  │
│                    2×2 correlation matrix          │
│                    → joined_metrics.jsonl          │
│                    → joined_summary.json           │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 6 · Reporting                               │
│  report-rag        Markdown tables + failure       │
│                    breakdown + example Q/A         │
│  report-baseline   Closed-book / oracle reports    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 7 · Grounding and Hallucination Analysis    │
│  analyze-grounding  7-label grounding taxonomy     │
│                     5 rule-based hallucination     │
│                     flags + optional LLM judge     │
│                     classify_root_cause()          │
│                     → grounding_scores.jsonl       │
│                     → citation_scores.jsonl        │
│                     → failure_analysis.jsonl       │
│                     → hallucination_report.md      │
│  inspect-failure   Single-question drill-down      │
└─────────────────────────────────────────────────────┘

  Optional branch: compare-embeddings
  Runs Stages 1–2 across 5+ embedding models under identical
  chunking. corpus_hash validation ensures fair comparison.
  Produces embedding_comparison_*.md leaderboard.
```

Every run writes a complete YAML config snapshot into `runs/<run_id>/config.yaml`
so experiments are independently auditable, and scoring can be re-run with a
different judge without re-running expensive generation.

---

## Evaluation Framework

### Layer 1 — Retrieval (no LLM required)

| Metric | What it measures |
|---|---|
| `doc_hit@k` | Gold document name found in top-k retrieved chunks |
| `page_hit@k` | Chunk page in {gold_page, matched_page} (±1 tolerance for annotation drift) |
| `evidence_text_hit@k` | Gold evidence substring appears verbatim in a retrieved chunk |
| `answerable_hit@k` | Any of the above signals matches |
| `*_first_hit_rank` | Rank of earliest hit (lower = better retrieval) |
| `Jaccard overlap` | Token overlap between gold evidence and the best-matching chunk |

The ±1 page offset tolerance accounts for known annotation drift between
FinanceBench ground-truth page numbers and PDF extraction page indices.

### Layer 2 — Answer Correctness (deterministic + optional LLM judge)

Four deterministic metrics handle the unit-normalization problem:

- `exact_match` — byte-for-byte equality
- `normalized_string_match` — case, whitespace, and punctuation normalized
- `numeric_match` — extracts all numbers from both strings and compares (`$1.577B == 1577M`)
- `unit_match` — parses (number, unit_scale) pairs and scales to a common base

The LLM judge adds: `verdict` (correct / partially_correct / incorrect / not_answered),
`numeric_error` flag, and `unsupported_claims` flag. Scoring is decoupled from
generation — `score-rag` can re-run with a different judge config without
re-running the expensive LLM generation step.

### Layer 3 — Grounding and Hallucination (M7)

| Component | Labels |
|---|---|
| Grounding taxonomy | grounded / partially_grounded / ungrounded / contradicted / insufficient_evidence / over_refusal / under_refusal |
| Rule-based flags | no_citation / cited_chunk_not_retrieved / predicted_number_not_in_context / predicted_year_not_in_context / refusal_with_evidence |
| Citation quality | supports_answer / partially_supports / does_not_support / citation_missing / citation_invalid |
| Context sufficiency | context_sufficient / context_partially_sufficient / context_insufficient |
| Root cause hierarchy | retrieval_failure → generation_failure → citation_failure → hallucination_under_refusal → over_refusal → no_failure |

The `join-metrics` step produces a 2×2 matrix (retrieval_hit × answer_correct).
This separates four distinct outcomes — evidence retrieved and answered correctly,
evidence retrieved but answer wrong (generation/reasoning failure), evidence missed
but answered correctly (parametric knowledge), and evidence missed and answer wrong
— enabling targeted decisions about whether to improve retrieval or generation.

---

## Provider Abstraction

All three provider-facing types (`LLMClient`, `EmbeddingClient`, `VectorStore`)
are defined as Python `Protocol` classes. Any conforming implementation drops in
without changes to pipeline code.

### LLM providers

| Provider | Class | Default use |
|---|---|---|
| Mock (deterministic response queue) | `MockLLMClient` | Tests and CI |
| Ollama (local inference) | `OllamaLLMClient` | `llama3.2:3b` |

### Embedding providers

| Provider | Model | Dimensions |
|---|---|---|
| Mock (SHA-256 deterministic) | — | configurable |
| Ollama | `nomic-embed-text` | 768 |
| Ollama | `qwen3-embedding:0.6b` | 1024 |
| Ollama | `snowflake-arctic-embed:335m` | 1024 |
| Ollama | `granite-embedding:278m` | 768 |
| Ollama | `bge-m3` | 1024 |
| OpenAI | `text-embedding-3-*` | configurable |
| Voyage | `voyage-*` | configurable |

The embedding comparison runner validates `corpus_hash` consistency across models,
ensuring hit-rate differences reflect embedding quality rather than variation in
chunked input data.

---

## Design Decisions

1. **`score-rag` is a separate command from `run-rag`.** Generation is expensive;
   re-scoring with a new judge config takes seconds. The `score_config.run_dir`
   parameter points at a completed generation run.

2. **Two independent judges** (answer correctness + grounding). These are
   orthogonal signals. A model can be `correct + ungrounded` (lucky hallucination:
   right answer, wrong citation) or `grounded + incorrect` (faithfully cited bad
   context). Co-labeling would mask both failure modes.

3. **`over_refusal` suppresses `numeric_error`.** When a model refuses to answer,
   there is no numeric prediction to evaluate. Co-labeling would inflate the
   numeric_error rate with noise.

4. **60 frozen dataclasses for all configs.** Immutability prevents accidental
   config mutation between pipeline stages. Overrides (e.g., `--limit`) use
   `dataclasses.replace()` to produce a new config, keeping the original intact.

5. **Corpus hash validation in the embedding comparison runner.** The
   `corpus_hash` in `IndexMetadata` is a SHA-256 of sorted chunk IDs. When
   comparing embedding models, the runner validates that each model indexed
   the same corpus, making the comparison fair.

6. **SHA-256 deterministic mock embeddings.** `MockEmbeddingClient` derives stable
   float vectors from `SHA-256(text) → uint32s → normalized floats`. Every test
   with FAISS indexing uses real FAISS logic; only the embedding step is mocked.
   No external service is needed for offline testing.

7. **±1 page offset tolerance in retrieval scoring.** `_page_candidates()` in
   `retrieval_scoring.py` accepts `{gold_page_num, matched_page_num}` to avoid
   false negatives from known FinanceBench annotation drift vs. PDF page indices.

8. **Streaming JSONL I/O for all large outputs.** Predictions, scores, and
   retrieval results are written one JSON object per line. The pipeline can
   process arbitrarily large datasets without loading everything into memory.

9. **`join-metrics` is a required step before `report-rag`.** Decoupling
   correlation analysis from report rendering allows re-reporting at different
   k values without re-scoring.

10. **`config_templates/` vs `configs/`.** Templates have `# FILL:` markers for
    getting started. Committed configs under `configs/` are tested working defaults.

---

## Test Coverage

- **887 tests** across **43 test files** — all pass offline without any external service
- **54 source modules**, **60 frozen dataclasses**
- Tests cover all CLI subcommands, all embedding providers (via mock), all scoring
  functions, the complete grounding taxonomy, and the 6-case root cause classifier
- `MockLLMClient` + `MockEmbeddingClient` (SHA-256 deterministic) enable full
  pipeline integration tests including real FAISS indexing and retrieval offline
- 2 opt-in Ollama integration tests are skipped without a local server; no flaky
  tests in offline mode

---

## Skills Demonstrated

- **Evaluation framework design** — 7-stage composable pipeline separating
  retrieval quality, answer correctness, evidence grounding, and hallucination
  detection into independent, measurable metrics
- **RAG system diagnostics** — 2×2 correlation matrix (retrieval hit × answer
  correct) distinguishes retrieval failures from generation failures, enabling
  targeted improvement decisions
- **LLM-as-judge integration** — dual-judge architecture with decoupled scoring:
  answer correctness and grounding judges run independently, allowing judge swap
  without regeneration
- **Controlled embedding model comparison** — same retrieval experiment run across
  5+ local models with corpus hash validation ensuring apples-to-apples results
- **Domain-specific metric engineering** — numeric answer normalization across
  units and scales, fiscal year period detection, ±1 page offset tolerance for
  dataset annotation drift
- **Python Protocol-based provider abstraction** — three Protocol interfaces
  (`LLMClient`, `EmbeddingClient`, `VectorStore`) enable provider swaps without
  touching pipeline code
- **Reproducibility engineering** — full YAML config snapshot per run, corpus
  hash validates index provenance, re-scoring without re-generation is a
  first-class workflow
- **Testing discipline** — 887 tests, SHA-256 mock embeddings for offline FAISS
  tests, 0 flaky tests in offline mode
- **Failure taxonomy design** — 7-label grounding taxonomy, 6-case deterministic
  root cause hierarchy, and 5 rule-based hallucination flags form a structured
  failure ontology for financial QA

---

## Quick Start

### Zero-dependency mock run (no Ollama required)

```bash
pip install -e ".[dev]"
pytest -q   # 887 tests, no external services required

python -m financebench_eval run-baseline \
  --config configs/baseline_closed_book.yaml
# → runs/<run_id>/{config.yaml, predictions.jsonl, scores.jsonl}
# → reports/baseline_<run_id>.md
```

### Full local pipeline (Ollama required)

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Process data
financebench-harness extract-documents
financebench-harness build-examples

# Chunk, index, retrieve
python -m financebench_eval chunk-documents --config configs/retrieval.yaml
python -m financebench_eval build-index --config configs/retrieval.yaml --progress
python -m financebench_eval retrieve --config configs/retrieval.yaml --run-id run_001

# Generate, score, join, report
python -m financebench_eval run-rag --config configs/evaluation/rag_dense_local.yaml
python -m financebench_eval score-rag --config configs/evaluation/rag_score_local.yaml \
  --run-dir runs/<rag_run_id>
python -m financebench_eval join-metrics \
  --retrieval-scores runs/run_001/retrieval_scores.jsonl \
  --answer-scores runs/<rag_run_id>/rag_combined_scores.jsonl \
  --output-dir runs/<rag_run_id>/joined --k 5
python -m financebench_eval report-rag \
  --joined-dir runs/<rag_run_id>/joined --run-id <rag_run_id>
```

### Embedding model comparison

```bash
python -m financebench_eval compare-embeddings \
  --config configs/embedding_comparison.yaml
# Compares nomic-embed-text, qwen3-embedding, snowflake-arctic-embed,
# granite-embedding — ranked by evidence_text_hit@10 under identical chunking
```

`config_templates/` has annotated starter configs with `# FILL:` markers for
each pipeline stage.

---

## Project Status

| Stage | Milestone | Status |
|---|---|---|
| Data ingestion | M1 | Complete |
| Baseline eval (closed-book + oracle) | M2 | Complete |
| Chunking, indexing, retrieval | M3/M4 | Complete |
| RAG generation + scoring | M5 | Complete |
| Embedding model comparison | M6 | Complete |
| Grounding and hallucination analysis | M7 | Complete |
| Prompt and retrieval improvements | M8 | Planned |

Public FinanceBench sample only. Results in `reports/` are local experiment
artefacts, not published benchmark claims.

---

## Expected Data Layout

Place the public FinanceBench sample files under `data/raw/financebench/`:

```text
data/
  raw/
    financebench/
      questions.jsonl
      documents/
  processed/
    financebench/
```

- Put the public FinanceBench question records at `data/raw/financebench/questions.jsonl`.
- Put locally available source documents under `data/raw/financebench/documents/`.
- Keep derived data under `data/processed/financebench/`.

The `data/` directory is intentionally ignored by Git. Do not commit raw FinanceBench files, PDFs, processed data, vector stores, or generated evaluation artefacts.

## Validate Local Data

After placing the files, run:

```bash
financebench-harness validate-data
```

By default this reads `configs/datasets/financebench.yaml`:

```yaml
dataset:
  name: financebench
  questions_path: data/raw/financebench/questions.jsonl
  documents_dir: data/raw/financebench/documents
  processed_dir: data/processed/financebench
```

Use `--config PATH` to point at a different dataset config:

```bash
financebench-harness validate-data --config /path/to/dataset.yaml
```

Use `--data-root PATH` if the FinanceBench files live somewhere else:

```bash
financebench-harness validate-data --data-root /path/to/financebench
```

To validate the question schema and check for duplicate question IDs, run:

```bash
financebench-harness validate-dataset
```

To check that evidence document names resolve to local files, run:

```bash
financebench-harness validate-documents
```

To extract local PDF text page by page into `data/processed/financebench/pages.jsonl`, run:

```bash
financebench-harness extract-documents
```

To validate that each FinanceBench evidence item resolves to an extracted document page
and to print every evidence text match or mismatch, run:

```bash
financebench-harness validate-evidence-pages
```

To build the canonical processed examples file for downstream harness stages, run:

```bash
financebench-harness build-examples
```

This writes accepted examples to `data/processed/financebench/examples.jsonl` and
auditable rejected examples to `data/processed/financebench/examples.rejected.jsonl`.
Downstream retrieval and evaluation code should read `examples.jsonl` rather than
the raw FinanceBench question file. Each processed evidence item includes
`matched_page_num`, which is the exact `page_num` from `pages.jsonl`; look up the
matched page by canonical PDF filename plus `matched_page_num`.

## Baseline Evaluation Modes

M2 evaluation starts with two non-RAG baselines configured in
`configs/evaluation/baselines.yaml`. The config references versioned prompt
template files under `prompts/baselines/`:

- `closed_book`: the model receives only the FinanceBench question, with no
  retrieval results and no document context. This measures what the model can
  answer from parametric knowledge or reasoning alone.
- `oracle_context`: the model receives the question plus the gold evidence text
  from the processed example. This removes retrieval from the path and helps
  isolate generation and reasoning quality when the relevant evidence is already
  available.

These modes are comparison baselines, not benchmark claims. Recorded runs should
still capture the model provider, model name, prompt id, prompt version, dataset
slice, and evaluation settings before results are reported. Prompt rendering
returns run metadata with the evaluation mode, prompt id, prompt version, and
template path so future run artefacts can record which prompt was used.

## LLM Provider Configuration

LLM calls should go through the shared provider interface in
`financebench_eval_harness.llm` rather than calling a provider directly. The
default local LLM config lives at `configs/llm/local.yaml` and records provider,
model name, temperature, max tokens, timeout settings, and optional Ollama
`base_url`. Evaluation and CI still default to the mock provider via
`configs/evaluation/local_mock.yaml`. Tests can use `MockLLMClient` to exercise
harness code without making API calls.

## Run A Mock Evaluation

Use `configs/baseline_closed_book.yaml` for a deterministic local baseline run
that renders prompts, writes mock LLM responses, scores the predictions, and
creates a Markdown report without calling an external API:

```bash
python -m financebench_eval run-baseline --config configs/baseline_closed_book.yaml
```

Each run writes a directory under `runs/` containing `config.yaml`, the
normalized config snapshot used for reproducibility, `predictions.jsonl` with
one model prediction per evaluated example, `scores.jsonl` with automatic and
judge scores, and `run_metadata.json` with run-level settings and counts. The
baseline command also writes `reports/baseline_<run_id>.md`. Change
`eval.mode`, `eval.limit`, or the `model` settings in the YAML file to compare
configurations without editing code.

## Run An Optional Local Ollama Smoke Baseline

Mock remains the default committed baseline for tests and CI. For a cheap local
smoke run against a real model, use one of the dedicated Ollama configs:

```bash
python -m financebench_eval run-eval --config configs/evaluation/ollama_closed_book.yaml --limit 5
python -m financebench_eval run-eval --config configs/evaluation/ollama_oracle_context.yaml --limit 5
```

These configs expect a local Ollama server at `http://localhost:11434` and use
`llama3.2:3b` with `temperature: 0.0` for reproducible smoke checks. Both the
answer model and judge model use Ollama in these opt-in configs. If the server
is unavailable or the model is missing, the harness emits a readable error, and
the opt-in smoke test is expected to fail immediately with that prerequisite
message rather than silently skipping.
