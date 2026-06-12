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

## Baseline Evaluation Modes

M2 evaluation starts with two non-RAG baselines configured in
`configs/evaluation/baselines.yaml`. The config references versioned prompt
template files under `prompts/baselines/`:

- `closed_book`: the model receives only the FinanceBench question, with no
  retrieval results and no document context. This measures what the model can
  answer from parametric knowledge or reasoning alone.
- `oracle_context`: the model receives the question plus the gold evidence text
  from the processed example. This removes retrieval from the path and helps
  isolate generation and reasoning quality when the relevant evidence is already
  available.

These modes are comparison baselines, not benchmark claims. Recorded runs should
still capture the model provider, model name, prompt id, prompt version, dataset
slice, and evaluation settings before results are reported. Prompt rendering
returns run metadata with the evaluation mode, prompt id, prompt version, and
template path so future run artefacts can record which prompt was used.

## LLM Provider Configuration

LLM calls should go through the shared provider interface in
`financebench_eval_harness.llm` rather than calling a provider directly. The
default local config lives at `configs/llm/local.yaml` and records provider,
model name, temperature, max tokens, and timeout settings. Tests can use
`MockLLMClient` to exercise harness code without making API calls.

## Run A Mock Evaluation

Use `configs/baseline_closed_book.yaml` for a deterministic local baseline run
that renders prompts, writes mock LLM responses, scores the predictions, and
creates a Markdown report without calling an external API:

```bash
python -m financebench_eval run-baseline --config configs/baseline_closed_book.yaml
```

Each run writes a directory under `runs/` containing `config.yaml`, the
normalized config snapshot used for reproducibility, `predictions.jsonl` with
one model prediction per evaluated example, `scores.jsonl` with automatic and
judge scores, and `run_metadata.json` with run-level settings and counts. The
baseline command also writes `reports/baseline_<run_id>.md`. Change
`eval.mode`, `eval.limit`, or the `model` settings in the YAML file to compare
configurations without editing code.
