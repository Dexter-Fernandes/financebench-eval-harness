M8: Experiment Reporting

M8 should answer:

Can the harness turn raw eval outputs into clear, reproducible reports that show what happened, what worked, what failed, and what should be improved next?

This milestone is where your project becomes easy to show to employers. Your repo’s stated scope is an eval harness for FinanceBench focused on answer correctness, evidence grounding, hallucination detection, and failure analysis, so M8 is about packaging those results into readable reports.

M8.1: Define report types

Goal: decide which reports the harness should generate.

Suggested report types:

ingestion_report
baseline_eval_report
retrieval_eval_report
rag_eval_report
embedding_comparison_report
hallucination_analysis_report
experiment_summary_report

Done when:

- Each report type has a clear purpose
- Each report has required input files
- Each report has a standard output path

Suggested structure:

reports/
  ingestion/
  baseline/
  retrieval/
  rag/
  embeddings/
  grounding/
  summaries/
M8.2: Define standard run directory structure

Goal: make every experiment easy to inspect later.

Suggested structure:

runs/
  <run_id>/
    config.yaml
    run_metadata.json
    predictions.jsonl
    retrieval_results.jsonl
    retrieval_scores.jsonl
    rag_predictions.jsonl
    rag_scores.jsonl
    failure_analysis.jsonl
    summary.json

Done when:

- Every run has a unique run_id
- Every run stores the exact config used
- Every run stores machine-readable outputs
- Reports can be regenerated from run files
M8.3: Add run metadata schema

Goal: make reports reproducible.

Useful metadata:

{
  "run_id": "2026-06-12_rag_dense_bge_m3",
  "created_at": "2026-06-12T12:00:00Z",
  "dataset": "financebench",
  "num_examples": 100,
  "retrieval_mode": "dense",
  "embedding_model": "BAAI/bge-m3",
  "chunk_size": 800,
  "chunk_overlap": 150,
  "top_k": 10,
  "generator_model": "gpt-4o-mini",
  "judge_model": "gpt-4o-mini",
  "prompt_versions": {
    "rag_answer": "rag_answer_v1",
    "judge": "judge_v1"
  }
}

Done when:

- Reports can show what setup produced the results
- Metadata is saved as JSON
- Missing metadata fields are handled cleanly
M8.4: Create report data loader

Goal: load all files needed for reporting.

Input examples:

retrieval_summary.json
retrieval_scores.jsonl
rag_predictions.jsonl
rag_answer_scores.jsonl
grounding_scores.jsonl
failure_analysis.jsonl
run_metadata.json
config.yaml

Done when:

- Report generator can load run outputs
- Missing optional files produce warnings
- Missing required files produce clear errors
M8.5: Generate baseline evaluation report

Goal: summarize M2 non-RAG results.

Report sections:

Run metadata
Evaluation mode
Model used
Number of questions
Correct / partially correct / incorrect counts
Accuracy
Average latency
Average token usage
Examples of correct answers
Examples of failures

Example table:

Metric	Value
Questions evaluated	100
Correct	35
Partially correct	18
Incorrect	47
Accuracy	35%
Avg latency	1.2s

Done when:

- Markdown report is generated from baseline outputs
- Report includes aggregate metrics
- Report includes a few example predictions
M8.6: Generate retrieval evaluation report

Goal: summarize M4 retrieval results.

Report sections:

Retriever setup
Embedding model
Chunk size and overlap
Top-k values
doc_hit@k
page_hit@k
evidence_hit@k
MRR@k
Best retrieval examples
Worst retrieval examples
Retrieval failure labels

Example table:

Metric	@1	@5	@10	@20
doc_hit	0.62	0.81	0.88	0.91
page_hit	0.41	0.67	0.74	0.80
evidence_hit	0.30	0.49	0.58	0.63

Done when:

- Retrieval metrics are readable in Markdown
- Report identifies where retrieval fails
- Report can be compared across retrieval runs
M8.7: Generate end-to-end RAG report

Goal: summarize M5 full RAG results.

Report sections:

Retriever setup
Generator model
Prompt version
Number of questions
Answer accuracy
Grounding accuracy
Citation accuracy
Correct when evidence retrieved
Correct when evidence missed
Common failure types
Representative examples

Example table:

Metric	Value
Questions evaluated	100
Answer correct	64%
Grounded answers	71%
Unsupported answers	11%
Evidence hit@10	76%
Correct when evidence retrieved	82%
Correct when evidence missed	21%

Done when:

- RAG report links retrieval results to answer results
- Report shows whether RAG improved over baseline
- Report includes failure examples
M8.8: Generate embedding comparison report

Goal: summarize M6 model comparison results.

Report sections:

Experiment goal
Candidate embedding models
Fixed settings
Retrieval leaderboard
Cost comparison
Latency comparison
Index size comparison
Recommended default model
Best cheap model
Best quality model
Best local model

Example table:

Model	evidence_hit@10	page_hit@10	MRR@10	Cost	Role
text-embedding-3-small	0.58	0.74	0.52	Low	Cheap baseline
text-embedding-3-large	0.64	0.79	0.57	Medium	Quality baseline
bge-m3	0.61	0.76	0.55	Local	Local baseline

Done when:

- Leaderboard is generated in Markdown and CSV
- Models are sorted by primary metric
- Report explains trade-offs clearly
M8.9: Generate hallucination and grounding report

Goal: summarize M7 failure analysis.

Report sections:

Grounding labels
Citation quality
Context sufficiency
Hallucination types
Root cause breakdown
Retrieval failures vs generation failures
Examples of unsupported answers
Examples of bad citations
Examples of wrong-number failures
Recommended fixes

Example table:

Failure type	Count
retrieval_miss	18
wrong_number	12
bad_citation	9
unsupported_claim	7
over_refusal	3

Done when:

- Report separates retrieval failures from generation failures
- Report identifies hallucination patterns
- Report suggests next improvements
M8.10: Add experiment summary report

Goal: create one high-level report that summarizes the whole project.

This is the most portfolio-friendly report.

Suggested sections:

Project objective
Dataset
Experimental setup
Baselines
Retrieval results
RAG results
Embedding comparison
Hallucination analysis
Key findings
Limitations
Next steps

Example key findings:

- RAG improved answer correctness over closed-book baseline.
- Evidence retrieval quality strongly correlated with final answer correctness.
- Most incorrect RAG answers came from retrieval misses and numeric reasoning errors.
- Hybrid retrieval improved page_hit@10 over dense-only retrieval.

Done when:

- One Markdown file summarizes the full experiment
- It can be linked from the README
- It explains results in plain English, not just metrics

Suggested output:

reports/experiment_summary_<run_id>.md
M8.11: Add CSV and JSON exports

Goal: make results easy to analyze later.

Suggested outputs:

retrieval_leaderboard.csv
rag_leaderboard.csv
embedding_leaderboard.csv
failure_breakdown.csv
summary_metrics.json

Done when:

- Every major report has machine-readable exports
- CSV files can be opened in Excel or pandas
- JSON files preserve full metric structure
M8.12: Add report templates

Goal: keep report formatting clean and consistent.

Suggested templates:

templates/
  baseline_report.md.j2
  retrieval_report.md.j2
  rag_report.md.j2
  embedding_comparison_report.md.j2
  hallucination_report.md.j2
  experiment_summary_report.md.j2

Done when:

- Reports use templates instead of hardcoded strings
- Tables are consistently formatted
- New report sections are easy to add

Jinja2 would work well here, but simple Python string formatting is fine for v1.

M8.13: Add report generation CLI

Goal: generate reports from the command line.

Suggested commands:

python -m financebench_eval report-baseline --run runs/<run_id>
python -m financebench_eval report-retrieval --run runs/<run_id>
python -m financebench_eval report-rag --run runs/<run_id>
python -m financebench_eval report-grounding --run runs/<run_id>
python -m financebench_eval report-experiment --run runs/<run_id>

Or one generic command:

python -m financebench_eval generate-report \
  --type rag \
  --run runs/<run_id>

Done when:

- Reports can be generated without editing code
- CLI validates required files
- Output path is printed after generation
M8.14: Add comparison across runs

Goal: compare multiple experiments side by side.

Example command:

python -m financebench_eval compare-runs \
  --runs runs/run_a runs/run_b runs/run_c \
  --output reports/run_comparison.md

Useful comparisons:

closed_book vs rag_dense
dense retrieval vs hybrid retrieval
embedding_model_A vs embedding_model_B
top_k=5 vs top_k=10
chunk_size=400 vs chunk_size=800

Done when:

- Multiple runs can be loaded together
- Key metrics are shown side by side
- Best run is highlighted by selected metric

Example table:

Run	Mode	Embedding	Top-k	evidence_hit@10	Accuracy	Grounded
run_a	dense	small	10	0.58	0.52	0.60
run_b	dense	large	10	0.64	0.57	0.67
run_c	hybrid	large	10	0.71	0.63	0.74
M8.15: Add README links to reports

Goal: make the repo easy to navigate.

README should include:

Project goal
Current milestone status
How to run a smoke eval
How to generate reports
Example reports
Key results

Example section:

## Example Reports

- Retrieval Evaluation Report
- End-to-End RAG Evaluation Report
- Embedding Model Comparison
- Hallucination and Grounding Analysis

Done when:

- README links to generated reports
- A recruiter can understand the project quickly
- Example outputs are visible without running the full benchmark
M8.16: Add tests for reporting

Goal: make sure reports do not break as schemas evolve.

Test cases:

- baseline report renders with mock data
- retrieval report renders hit@k table
- RAG report renders answer metrics
- embedding leaderboard sorts correctly
- missing optional files produce warnings
- missing required files fail clearly
- generated Markdown contains expected sections

Done when:

pytest tests/test_reports.py

passes.

M8 final deliverable

At the end of M8, you should be able to run:

python -m financebench_eval generate-report \
  --type experiment-summary \
  --run runs/<run_id>

and get:

reports/
  baseline_<run_id>.md
  retrieval_eval_<run_id>.md
  rag_eval_<run_id>.md
  embedding_comparison_<run_id>.md
  hallucination_analysis_<run_id>.md
  experiment_summary_<run_id>.md

runs/<run_id>/
  summary_metrics.json
  failure_breakdown.csv
  leaderboard.csv