# Project Status

## Current State

- M4 eval-retrieval milestone is complete on the `M4` branch (not yet merged to main).
- 581 tests pass (2 Ollama skips).
- Full eval pipeline operational:
  ```bash
  python -m financebench_eval chunk-documents  --config configs/retrieval.yaml
  python -m financebench_eval build-index      --config configs/retrieval.yaml
  python -m financebench_eval retrieve         --config configs/retrieval.yaml --run-id <run_id>
  python -m financebench_eval eval-retrieval   --config configs/retrieval.yaml --run-id <run_id>
  python -m financebench_eval inspect-retrieval-failure \
    --config configs/retrieval.yaml --run-id <run_id> --question-id <question_id>
  ```
- First baseline (run_004, nomic-embed-text, chunk_size=800, top_k=5):
  - `doc_hit@5 = 23.9%`, `page_hit@5 = 0%`, `evidence_text_hit@5 = 0%`
  - 76.1% `wrong_document`, 23.9% `right_document_wrong_page`
- Failure label taxonomy in place: `wrong_document`, `right_document_wrong_page`, `right_page_low_rank`, `evidence_not_in_chunk`, `table_extraction_issue`.
- `inspect-retrieval-failure` CLI available for per-question diagnosis.
- M2 baseline evaluation (`closed_book`, `oracle_context`) and M1 data tooling remain intact.

## Intended Build Path

- Start with local Ollama models for development and debugging.
- Add hosted/frontier model support through provider abstractions once the local harness is reproducible.
- Keep exact provider and model names configurable so comparisons use whichever models are available at evaluation time.

## Current Engineering Milestone

**M5: end-to-end RAG evaluation**

Answers: *Can the harness retrieve evidence, generate an answer, and evaluate whether the answer is correct and grounded?*

- Add `run-rag` command: retrieves top-k chunks, passes them to a generation model, writes `rag_predictions.jsonl`.
- Add `score-rag` command: evaluates predictions for answer correctness, grounding, and hallucination; writes `rag_answer_scores.jsonl`, `rag_grounding_scores.jsonl`, `rag_combined_scores.jsonl`, and `reports/rag_eval_<run_id>.md`.
- New configs: `configs/rag_eval.yaml` (generation settings) and `configs/rag_score.yaml` (scoring settings).
- Add generation provider abstraction (Ollama-backed for local dev; extensible to hosted models).
- Merge M4 to main before starting M5 work.

## Current Boundaries

- Use only public FinanceBench sample data.
- Keep local datasets, PDFs, processed page JSONL, processed examples JSONL, vector stores, model caches, secrets, and generated run artefacts out of version control.
- `reports/*.md` are gitignored; generated eval reports remain local unless intentionally curated.
- Do not publish benchmark claims or metrics until they are produced by reproducible runs backed by committed configuration.
