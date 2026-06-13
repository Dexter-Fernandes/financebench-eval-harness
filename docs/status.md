# Project Status

## Current State

- M5.1–M5.6 complete on `M5` branch (not merged to main).
- 688 tests pass (2 Ollama skips).
- Full pipeline operational: `build-index → retrieve → run-rag`.
- First retrieval baseline (run_004, nomic-embed-text, chunk_size=800, top_k=5):
  - `doc_hit@5 = 23.9%`, `page_hit@5 = 0%`, `evidence_text_hit@5 = 0%`

## Current Engineering Milestone

**M5.7 next: RAG scoring and markdown report**

A `score-rag` command that reads `predictions.jsonl` and produces answer correctness + grounding scores plus a markdown report.

---

## How to Run M5 End-to-End

### Prerequisites

**1. Ollama running with required models:**
```bash
ollama serve                    # start server if not running
ollama pull llama3.2:3b         # generation model
ollama pull nomic-embed-text    # embedding model (for retrieval)
```

**2. Processed examples** at `data/processed/financebench/examples.jsonl`:
```bash
# Rebuild if missing:
python -m financebench_eval_harness.cli build-examples --config configs/dataset.yaml
```

**3. Chunk index** built from document pages:
```bash
python -m financebench_eval_harness.cli chunk-documents --config configs/retrieval.yaml
python -m financebench_eval_harness.cli build-index     --config configs/retrieval.yaml --progress
```

**4. Retrieval results** from a `retrieve` run:
```bash
python -m financebench_eval_harness.cli retrieve \
  --config configs/retrieval.yaml --run-id run_005
# Output: runs/run_005/retrieval_results.jsonl
```

**5. Update `configs/evaluation/rag_dense_local.yaml`** to point at that run:
```yaml
rag_eval:
  retrieval_results_path: runs/run_005/retrieval_results.jsonl
```

---

### Running RAG Generation

```bash
python -m financebench_eval_harness.cli run-rag \
  --config configs/evaluation/rag_dense_local.yaml \
  --run-id <run_id> \
  [--limit N]        # optional: N examples for smoke tests
```

**Outputs written to `runs/<run_id>/`:**

| File | Contents |
|---|---|
| `config.yaml` | Full config snapshot for reproducibility |
| `predictions.jsonl` | One row per question: `question_id`, `question`, `gold_answer`, `prediction`, `mode`, `model_provider`, `model_name`, `prompt_id`, `prompt_version`, `top_k`, `retrieved_chunk_ids`, `latency_ms`, `input_tokens`, `output_tokens`, `status`, `error` |
| `scores.jsonl` | One row per question: `question_id`, `scores` (exact_match, normalized_string_match, contains_gold_answer, numeric_match), `judge`, `status` |
| `run_metadata.json` | `run_id`, `mode`, `top_k`, `max_context_chars`, model config, `attempted_count`, `success_count`, `error_count`, `score_summary`, `duration_ms` |

---

### Config Reference

**`configs/evaluation/rag_dense_local.yaml`:**
```yaml
rag_eval:
  examples_path: data/processed/financebench/examples.jsonl
  retrieval_results_path: runs/<run_id>/retrieval_results.jsonl  # ← update per run
  output_dir: runs
  mode: rag_dense          # or: rag_oracle, rag_no_context
  top_k: 5                 # number of chunks passed to LLM
  max_context_chars: 4000  # optional char budget; omit for no limit
  eval_config_path: configs/evaluation/rag_modes.yaml  # optional override

model:
  provider: ollama          # or: mock (dry-run, no server needed)
  model_name: llama3.2:3b
  temperature: 0.0
  max_tokens: 512
  timeout_seconds: 60
  base_url: http://localhost:11434

judge:
  enabled: false            # set true and fill in prompt to enable LLM-as-judge
  provider: ollama
  model_name: llama3.2:3b
  temperature: 0.0
  max_tokens: 256
  timeout_seconds: 60
  base_url: http://localhost:11434
  prompt:
    id: answer_correctness_v1
    version: v1
    template_path: prompts/judges/answer_correctness_v1.txt
```

---

### Available RAG Modes

| Mode | Context source | Prompt file |
|---|---|---|
| `rag_dense` | Top-k dense retrieved chunks | `prompts/rag/rag_dense_v2.txt` |
| `rag_oracle` | Gold evidence text (upper bound) | `prompts/rag/rag_oracle_v1.txt` |
| `rag_no_context` | None — closed-book | `prompts/rag/rag_no_context_v1.txt` |

Mode definitions live in `configs/evaluation/rag_modes.yaml`.

---

### Switching Providers

**Mock provider — no Ollama needed, useful for pipeline wiring checks:**
```yaml
model:
  provider: mock
  model_name: mock-llm
  temperature: 0.0
  max_tokens: 512
  timeout_seconds: 30
```

**Different Ollama model:**
```yaml
model:
  provider: ollama
  model_name: mistral:7b    # any model available in your Ollama instance
  base_url: http://localhost:11434
```

---

### Dry-Run Smoke Test (no Ollama, no data files)

```bash
pytest tests/test_rag_run.py tests/test_rag_run_config.py -v
```

---

### Inspecting Results

```bash
# Overall summary
cat runs/<run_id>/run_metadata.json | python -m json.tool

# First prediction row (formatted)
head -1 runs/<run_id>/predictions.jsonl | python -m json.tool

# Score summary only
python -c "
import json; from pathlib import Path
meta = json.loads(Path('runs/<run_id>/run_metadata.json').read_text())
print(json.dumps(meta['score_summary'], indent=2))
"

# Which questions had LLM errors
python -c "
import json; from pathlib import Path
rows = [json.loads(l) for l in Path('runs/<run_id>/predictions.jsonl').read_text().splitlines() if l.strip()]
errors = [r for r in rows if r['status'] == 'error']
print(f'{len(errors)} errors'); [print(r['question_id'], r['error']) for r in errors]
"
```

---

## Complete Pipeline Reference

```bash
# 1. (Once) prepare data and index
python -m financebench_eval_harness.cli chunk-documents --config configs/retrieval.yaml
python -m financebench_eval_harness.cli build-index     --config configs/retrieval.yaml --progress

# 2. Run retrieval
python -m financebench_eval_harness.cli retrieve \
  --config configs/retrieval.yaml --run-id <retrieval_run_id>

# 3. Score retrieval (optional but recommended before generation)
python -m financebench_eval_harness.cli eval-retrieval \
  --config configs/retrieval.yaml --run-id <retrieval_run_id>

# 4. Run RAG generation
#    (update retrieval_results_path in rag_dense_local.yaml first)
python -m financebench_eval_harness.cli run-rag \
  --config configs/evaluation/rag_dense_local.yaml \
  --run-id <rag_run_id> [--limit N]

# 5. (Future: M5.7) Score RAG answers
# python -m financebench_eval_harness.cli score-rag --run-id <rag_run_id>
```

## Intended Build Path

- Local Ollama models for development and debugging.
- Hosted/frontier model support via provider abstraction once local harness is reproducible.
- Keep provider, model name, prompt version, and dataset slice configurable; record all in `run_metadata.json` and the config snapshot for every run.

## Current Boundaries

- Use only public FinanceBench sample data.
- Keep local datasets, PDFs, processed pages, vector stores, model caches, secrets, and generated run artefacts out of version control.
- `reports/*.md` are gitignored; generated eval reports remain local unless intentionally curated.
- Do not publish benchmark claims until produced by reproducible runs backed by committed configuration.
