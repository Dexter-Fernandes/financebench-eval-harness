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
