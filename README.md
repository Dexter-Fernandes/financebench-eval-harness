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
