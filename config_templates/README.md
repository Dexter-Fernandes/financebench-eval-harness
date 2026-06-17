# Config Templates

Barebones starting configs for each pipeline stage. Copy the one you need into `configs/`, fill in the `FILL:` fields, then run the corresponding command.

## Pipeline order

| File | Stage | Command |
|------|-------|---------|
| `01_dataset.yaml` | Load raw FinanceBench data | `python -m financebench_eval load-dataset` |
| `02_retrieval.yaml` | Chunk, index, retrieve | `chunk-documents` → `build-index` → `retrieve` |
| `03_rag_run.yaml` | Generate answers with an LLM | `python -m financebench_eval run-rag` |
| `04_rag_score.yaml` | Score answer correctness + grounding | `python -m financebench_eval score-rag` |
| `05_grounding_analysis.yaml` | Hallucination detection + root cause | `python -m financebench_eval analyze-grounding` |
| `06_embedding_comparison.yaml` | Compare embedding models side by side | `python -m financebench_eval compare-embeddings` |
| `07_baseline_eval.yaml` | Closed-book or oracle-context baselines | `python -m financebench_eval run-baseline` |

## What to fill in

Every parameter marked `# FILL:` must be set before running. Parameters without that marker have working defaults you can leave as-is.

## Ollama setup (for local runs)

```bash
ollama serve                        # start the server
ollama pull llama3.2:3b             # generator / judge model
ollama pull nomic-embed-text        # embedding model
```

## Minimal end-to-end run

```bash
# 1. Process raw data
python -m financebench_eval load-dataset --config configs/01_dataset.yaml

# 2. Chunk + index + retrieve
python -m financebench_eval chunk-documents --config configs/02_retrieval.yaml
python -m financebench_eval build-index     --config configs/02_retrieval.yaml --progress
python -m financebench_eval retrieve        --config configs/02_retrieval.yaml --run-id run_001

# 3. Generate answers
# (edit 03_rag_run.yaml: set retrieval_results_path to runs/run_001/retrieval_results.jsonl)
python -m financebench_eval run-rag --config configs/03_rag_run.yaml

# 4. Score
# (edit 04_rag_score.yaml: set run_dir to the timestamp dir written by run-rag)
python -m financebench_eval score-rag --config configs/04_rag_score.yaml

# 5. Analyse grounding
# (edit 05_grounding_analysis.yaml: set run_dir same as step 4)
python -m financebench_eval analyze-grounding --config configs/05_grounding_analysis.yaml
```
