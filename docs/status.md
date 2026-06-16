# Project Status

## Current State

- M7 complete on branch `M7` — PR not yet raised.
- 885 tests pass, 2 skipped (Ollama integration tests).
- Full pipeline operational through hallucination and grounding analysis (M7).
- No live grounding analysis run recorded yet — all M7 tests use mock data.

## Current Engineering Milestone

**M7 complete: Hallucination and Grounding Analysis**

```bash
# Run grounding analysis on a completed RAG run
# (update run_dir in configs/grounding_analysis.yaml first)
python -m financebench_eval analyze-grounding \
  --config configs/grounding_analysis.yaml

# Inspect one question's failure details
python -m financebench_eval inspect-failure \
  --run runs/<run_id> --question-id financebench_001
```

Output layout:
```
runs/<run_id>/
  grounding_scores.jsonl          ← per-question grounding label + rule flags
  citation_scores.jsonl           ← citation quality per question
  failure_analysis.jsonl          ← joined signals + root_cause per question
  failure_summary.json            ← aggregate root cause counts
  grounding_analysis_config.yaml  ← config snapshot

reports/
  hallucination_analysis_<run_id>.md
```

Config: `configs/grounding_analysis.yaml`
- Set `grounding_analysis.run_dir` to the target RAG run directory
- Set `grounding_judge.enabled: true` to activate LLM-as-judge (requires Ollama)

## Next Steps

1. Raise M7 PR and merge to main.
2. Run `analyze-grounding` against a real RAG run; record root cause breakdown.
3. Enable `grounding_judge` with a local Ollama model for richer per-example labels.
4. Use failure analysis to guide M8 improvements (prompt tuning or retrieval improvements).

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
