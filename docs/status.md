# Project Status

## Current State

- The repository has completed local data tooling for PRD Milestone 1.1 through M1.9.
- Implemented capabilities include expected dataset layout validation, YAML dataset config, FinanceBench question loading, dataset schema validation, document registry validation, PDF page extraction, evidence-to-page validation, and canonical processed example generation.
- `financebench-harness validate-evidence-pages` links each evidence item to a local document, extracted page, and deterministic text match/mismatch result.
- `financebench-harness build-examples` writes accepted examples to `data/processed/financebench/examples.jsonl` and rejected audit rows to `data/processed/financebench/examples.rejected.jsonl`.

## Intended Build Path

- Start with local Ollama models for development and debugging.
- Add hosted/frontier model support later through provider abstractions once the local harness is reproducible.
- Keep exact provider and model names configurable so comparisons can use whichever models are available at evaluation time.

## Next Engineering Milestone

- Continue toward retrieval/chunking and evaluation-run plumbing, using M1.9 `examples.jsonl` as the canonical downstream dataset.

## Current Boundaries

- Use only public FinanceBench sample data.
- Keep local datasets, PDFs, processed page JSONL, processed examples JSONL, vector stores, model caches, secrets, and generated run artefacts out of version control.
- Treat local slices such as `data/raw/financebench/questions20.jsonl` as ignored development conveniences unless a committed sample/slice strategy is explicitly added.
- Do not publish benchmark claims or metrics until they are produced by reproducible runs.
