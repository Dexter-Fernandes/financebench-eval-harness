# Project Status

## Current State

- The repository has completed local data tooling for PRD Milestone 1.1 through M1.9 and now has the M2 baseline evaluation path in place.
- Baseline evaluation supports `closed_book` and `oracle_context`, writes `predictions.jsonl`, `scores.jsonl`, and `run_metadata.json`, and records provider/model metadata for each run.
- The shared LLM interface now supports structured generation results, optional Ollama-backed local generation and judge calls, and provider token counts in `predictions.jsonl` when available.
- Numeric scoring is hardened for common financial negative formats so real oracle-context outputs do not crash evaluation runs.
- `configs/evaluation/local_mock.yaml` remains the default committed test and CI path, while dedicated Ollama configs provide opt-in local smoke coverage.
- Implemented capabilities also include expected dataset layout validation, YAML dataset config, FinanceBench question loading, dataset schema validation, document registry validation, PDF page extraction, evidence-to-page validation, and canonical processed example generation.
- `financebench-harness validate-evidence-pages` links each evidence item to a local document, extracted page, and deterministic text match/mismatch result.
- `financebench-harness build-examples` writes accepted examples to `data/processed/financebench/examples.jsonl` and rejected audit rows to `data/processed/financebench/examples.rejected.jsonl`.

## Intended Build Path

- Start with local Ollama models for development and debugging.
- Add hosted/frontier model support later through provider abstractions once the local harness is reproducible.
- Keep exact provider and model names configurable so comparisons can use whichever models are available at evaluation time.

## Next Engineering Milestone

- M3 is a standalone retrieval pipeline v1, not a new `run-eval` mode yet.
- Chunk extracted `pages.jsonl` text deterministically, generate local Ollama embeddings, and build a FAISS index plus chunk-id mapping.
- Retrieve top-k chunks for FinanceBench questions from `data/processed/financebench/examples.jsonl` and write reviewable retrieval artefacts such as `chunks.jsonl`, FAISS index files, `retrieval_results.jsonl`, and `retrieval_run_metadata.json`.
- Keep retrieval failures separate from generation, grounding, and scoring failures so later retrieval-backed evaluation runs remain interpretable.
- Keep the Ollama smoke configs as a cheap local wiring check for `closed_book` and `oracle_context`, not as the main quality baseline.

## Current Boundaries

- Use only public FinanceBench sample data.
- Keep local datasets, PDFs, processed page JSONL, processed examples JSONL, vector stores, model caches, secrets, and generated run artefacts out of version control.
- Treat local slices such as `data/raw/financebench/questions20.jsonl` as ignored development conveniences unless a committed sample/slice strategy is explicitly added.
- Do not publish benchmark claims or metrics until they are produced by reproducible runs.
