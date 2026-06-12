# Project Status

## Current State

- M3 standalone retrieval pipeline is complete and review-clean on the `M3` branch.
- All 421 tests pass (2 pre-existing Ollama skips).
- Three-stage pipeline is runnable from a single config file:
  ```bash
  python -m financebench_eval chunk-documents --config configs/retrieval.yaml
  python -m financebench_eval build-index     --config configs/retrieval.yaml
  python -m financebench_eval retrieve        --config configs/retrieval.yaml
  ```
- Output layout: `data/processed/financebench/chunks.jsonl`, `data/indexes/financebench/{index.faiss,chunk_metadata.jsonl,index_metadata.json}`, `runs/<run_id>/{retrieval_results.jsonl,retrieval_run_metadata.json,config.yaml}`.
- M2 baseline evaluation (`closed_book`, `oracle_context`) and all earlier M1 data tooling remain intact.

## Intended Build Path

- Start with local Ollama models for development and debugging.
- Add hosted/frontier model support through provider abstractions once the local harness is reproducible.
- Keep exact provider and model names configurable so comparisons use whichever models are available at evaluation time.

## Next Engineering Milestone

- **M4**: wire retrieval into `run-eval` as a `retrieval_augmented` baseline mode.
- Compare retrieval-backed vs. closed-book vs. oracle-context on the FinanceBench sample.
- Keep retrieval failures (no chunk / wrong chunk) separate from generation and scoring failures.
- Merge `M3` → `main` before starting M4 work.

## Current Boundaries

- Use only public FinanceBench sample data.
- Keep local datasets, PDFs, processed page JSONL, processed examples JSONL, vector stores, model caches, secrets, and generated run artefacts out of version control.
- Treat local slices such as `data/raw/financebench/questions20.jsonl` as ignored development conveniences.
- Do not publish benchmark claims or metrics until they are produced by reproducible runs backed by committed configuration.
