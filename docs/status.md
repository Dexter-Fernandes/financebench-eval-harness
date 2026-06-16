# Project Status

## Current State

- M6 complete on branch `M6` — PR not yet raised.
- 835 tests pass, 2 skipped (Ollama integration tests).
- Full pipeline operational through RAG evaluation (M5) and embedding comparison (M6).
- No live `evidence_hit@10` results recorded yet — dry-run only.

## Current Engineering Milestone

**M6 complete: Embedding model comparison**

```bash
# Full run (requires Ollama)
python -m financebench_eval compare-embeddings \
  --config configs/embedding_comparison.yaml

# Dry-run (no Ollama needed, writes stub report)
python -m financebench_eval compare-embeddings \
  --config configs/embedding_comparison.yaml --dry-run
```

Output layout:
```
runs/embedding_comparison/<run_id>/
  config.yaml                     ← corpus_hash + fixed settings snapshot
  model_runs/
    nomic-embed-text/
      retrieval_results.jsonl
      retrieval_scores.jsonl
      retrieval_summary.json
    granite-embedding__278m/
      retrieval_results.jsonl
      retrieval_scores.jsonl
      retrieval_summary.json
  embedding_leaderboard.csv
  embedding_leaderboard.json      ← sorted by evidence_hit@10
  embedding_decision.json         ← role assignments

reports/
  embedding_comparison_<run_id>.md
```

Active candidate models in `configs/embedding_comparison.yaml`:
- `nomic-embed-text` (ollama, 768-dim, local_baseline)
- `granite-embedding:278m` (ollama, 768-dim, open_source)

Commented-out candidates (ready to enable):
- `bge-m3` — NaN issue with GGUF, fallback implemented
- `mxbai-embed-large` — strong retrieval-focused model
- `snowflake-arctic-embed:335m` — asymmetric retrieval specialist
- `all-minilm` — fast floor baseline

## Next Steps

1. Raise M6 PR and merge to main.
2. Run `compare-embeddings` with real data; record `evidence_hit@10` baseline.
3. Optionally add `mxbai-embed-large` or `snowflake-arctic-embed:335m` for a stronger comparison.
4. M7: use winning embedding model for a full RAG comparison (generation quality vs retrieval quality).

---

## How to Run the Full M5 Pipeline

```bash
# Prerequisites
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Chunk + index (once)
python -m financebench_eval chunk-documents --config configs/retrieval.yaml
python -m financebench_eval build-index     --config configs/retrieval.yaml --progress

# Retrieve
python -m financebench_eval retrieve \
  --config configs/retrieval.yaml --run-id run_005

# RAG generation (update retrieval_results_path in config first)
python -m financebench_eval run-rag \
  --config configs/evaluation/rag_dense_local.yaml

# Score + report
python -m financebench_eval score-rag \
  --config configs/evaluation/rag_score_local.yaml \
  --run-dir runs/<rag_run_id>

python -m financebench_eval join-metrics \
  --retrieval-scores runs/<retrieval_run_id>/retrieval_scores.jsonl \
  --answer-scores    runs/<rag_run_id>/rag_combined_scores.jsonl \
  --output-dir       runs/<rag_run_id>/joined --k 5

python -m financebench_eval report-rag \
  --joined-dir        runs/<rag_run_id>/joined \
  --rag-run-dir       runs/<rag_run_id> \
  --retrieval-summary runs/<retrieval_run_id>/retrieval_summary.json \
  --run-id            <rag_run_id>
```

## Current Boundaries

- Public FinanceBench sample data only.
- Local datasets, PDFs, vector stores, model caches, secrets, and run artefacts stay out of version control.
- `reports/*.md` gitignored; generated reports remain local unless intentionally curated.
- Do not publish benchmark claims until backed by reproducible runs with committed configuration.
