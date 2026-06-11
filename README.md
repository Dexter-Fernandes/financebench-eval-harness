# financebench-eval-harness
Evaluation harness for RAG and long-context LLMs on FinanceBench, focusing on answer correctness, evidence grounding, hallucination detection, and failure analysis.

## Current Scope

The project is currently implementing PRD Milestone 1: local FinanceBench data layout, dataset validation, experiment metadata conventions, and a minimal run artefact shape.

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

Use `--data-root PATH` if the FinanceBench files live somewhere else:

```bash
financebench-harness validate-data --data-root /path/to/financebench
```
