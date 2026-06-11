# Product Requirements Document: FinanceBench Eval Harness

## 1. Product Name

**financebench-eval-harness**

## 2. One-line Summary

A research-engineering evaluation harness for testing document-grounded LLM systems on FinanceBench, with emphasis on answer correctness, evidence grounding, hallucination detection, reproducible experiments, and actionable failure analysis.

## 3. Documentation Change Note

This PRD revision is documentation-only. It does not introduce code changes, public APIs, CLI commands, data schemas, model integrations, or runtime behaviour by itself.

## 4. Problem Statement

LLM and RAG systems can appear useful in demos while failing under structured evaluation. In document-grounded financial QA, user trust depends on whether the system can locate relevant evidence, reason over it correctly, and explain its answer without unsupported claims.

The project exists to de-risk document-grounded AI systems through repeatable evaluation rather than anecdotal testing. It should make model, prompt, and retrieval changes measurable, expose failure modes early, and produce clear findings that a product or engineering team could use to decide what to improve next.

Common failures to measure include:

- Retrieving irrelevant or incomplete evidence.
- Retrieving the right evidence but reasoning incorrectly.
- Producing unsupported claims or fabricated details.
- Refusing to answer when sufficient evidence exists.
- Answering correctly but failing to cite or ground the response.
- Producing inconsistent results across model, prompt, or retrieval changes.

## 5. Target Users

### Primary User

A research engineer, AI engineer, or ML engineer who wants to evaluate document-grounded QA systems with trusted benchmark data and repeatable experiments.

### Secondary Users

- Hiring managers reviewing the project as evidence of applied AI research-engineering judgement.
- Product and engineering stakeholders who need concise evidence about model readiness, failure modes, and tradeoffs.
- Developers comparing local and hosted LLM pipelines.
- Practitioners learning RAG evaluation, retrieval metrics, RAGAS, and structured failure analysis.

## 6. Goals

### Product Goals

1. Load and validate the public FinanceBench sample.
2. Run document-grounded QA experiments over selected FinanceBench examples.
3. Support repeatable comparisons across retrieval settings, prompts, and model providers.
4. Support local inference through Ollama and optional hosted providers available at evaluation time.
5. Evaluate outputs using a mix of automated metrics, deterministic checks, and qualitative review.
6. Separate retrieval failures from generation and reasoning failures.
7. Generate experiment reports that explain what changed, what improved, what regressed, and what remains uncertain.
8. Provide a failure taxonomy suitable for technical interviews, portfolio review, and engineering discussion.

### Learning and Portfolio Goals

1. Demonstrate practical understanding of evaluation beyond "looks good" manual testing.
2. Show disciplined experiment design: hypotheses, controlled comparisons, metrics, and findings.
3. Build evidence-grounded evaluation metrics for financial document QA.
4. Compare smaller local models against larger hosted models with explicit tradeoffs.
5. Communicate technical findings in a format useful to non-specialist stakeholders.

## 7. Non-goals

This project will not:

- Train or fine-tune a new LLM.
- Claim benchmark-leading FinanceBench performance.
- Use the full private FinanceBench dataset.
- Build a production-grade enterprise RAG platform.
- Commit FinanceBench PDFs, vector stores, model caches, API keys, or large generated outputs to GitHub.
- Use private, confidential, or proprietary financial documents.
- Present generated metrics as externally validated benchmark results without reproducible evidence.

## 8. Dataset and Data Boundaries

### Primary Dataset

The project uses the public FinanceBench open-source sample.

Expected local data layout:

```text
data/
  raw/
    financebench/
      questions.jsonl
      documents/
  processed/
    financebench/
```

The public FinanceBench question records should be placed at `data/raw/financebench/questions.jsonl`. Public or locally available source documents should be placed under `data/raw/financebench/documents/`. Derived, normalized, or cached data should live under `data/processed/financebench/`.

### Data Handling Requirements

- Keep source data paths configurable so the repository can remain lightweight.
- Validate required fields before running an experiment.
- Record dataset version, sample selection, and document availability in each report.
- Treat all generated outputs as experiment artefacts, not source data.
- Avoid committing large, regenerated, or licensed artefacts unless explicitly allowed by the repository policy.

## 9. Product Requirements

### Data Ingestion and Validation

- Load FinanceBench examples and associated document metadata from local JSONL files.
- Validate that each selected question has an identifier, question text, expected answer, source document reference, and evidence metadata when available.
- Produce clear validation errors for missing files, malformed records, or examples that cannot be evaluated.

### Document-grounded QA Pipeline

- Provide a baseline RAG-style pipeline that retrieves relevant context and generates an answer from that context.
- Allow a long-context baseline where the model receives larger document excerpts when feasible.
- Keep prompts and retrieval settings configurable so experiments can compare changes without editing code.
- Capture raw model outputs, retrieved context, citations, and runtime metadata for evaluation.

### Evaluation Framework

- Measure answer correctness against reference answers using automated and reviewable signals.
- Measure evidence grounding by checking whether the response is supported by retrieved or supplied context.
- Track retrieval quality separately from answer quality, including whether expected evidence appears in the retrieved context.
- Detect likely hallucinations, unsupported claims, and answer refusals.
- Use RAGAS where appropriate as one evaluation layer, supplemented by project-specific checks for FinanceBench-style answers.

### Experiment Tracking and Reporting

- Represent each run with a stable experiment name, configuration, dataset slice, model settings, and timestamp.
- Generate a structured report with aggregate metrics, per-question outcomes, failure categories, representative examples, and recommended next steps.
- Make regressions visible when comparing a new run to a baseline.
- Keep report language readable for both technical reviewers and product stakeholders.

## 10. Evaluation Methodology

Each experiment should begin with a clear hypothesis, such as "a reranking step improves evidence recall without materially increasing hallucinations" or "a larger hosted model improves numeric reasoning on multi-hop financial questions."

Experiments should control for:

- Dataset slice and document availability.
- Retrieval configuration.
- Prompt version.
- Model provider, model name, and generation settings.
- Evaluation metric versions.

Core evaluation dimensions:

- **Answer correctness:** whether the final answer matches the expected answer in substance.
- **Evidence recall:** whether expected supporting evidence appears in retrieved or supplied context.
- **Groundedness:** whether answer claims are supported by the available context.
- **Hallucination risk:** whether the answer introduces unsupported figures, entities, or reasoning.
- **Refusal behaviour:** whether the model declines appropriately when evidence is insufficient.
- **Regression risk:** whether a change improves one dimension while degrading another.

## 11. Failure Taxonomy

Each failed or uncertain answer should be assigned one primary category:

- **Retrieval miss:** expected evidence was not retrieved or supplied.
- **Partial evidence:** some relevant evidence was available, but key details were missing.
- **Reasoning error:** evidence was available, but the model drew the wrong conclusion.
- **Calculation error:** evidence was available, but arithmetic or unit handling failed.
- **Unsupported claim:** answer included material not grounded in context.
- **Citation or grounding failure:** answer was correct but poorly supported.
- **Over-refusal:** model declined despite sufficient evidence.
- **Under-refusal:** model answered despite insufficient evidence.
- **Evaluation ambiguity:** reference answer, evidence, or scoring rule needs human review.

## 12. Success Criteria and Acceptance Tests

The project is successful when it can demonstrate:

- A reproducible evaluation run over a documented FinanceBench sample slice.
- A baseline report that includes aggregate metrics and per-question outcomes.
- At least one controlled comparison between two model, prompt, or retrieval configurations.
- Clear separation between retrieval failures and generation or reasoning failures.
- A qualitative failure analysis with representative examples and recommended improvements.
- Documentation that allows another engineer or hiring reviewer to understand the purpose, methodology, and limits of the work.

Acceptance tests for the PRD-defined product:

- Given valid public FinanceBench sample files, the harness can load and validate the selected examples.
- Given an experiment configuration, the harness can run the configured QA pipeline and save outputs for evaluation.
- Given completed outputs, the harness can produce metric summaries and a failure taxonomy.
- Given two comparable runs, the harness can identify metric changes and highlight likely regressions.
- Given insufficient or missing evidence, the report can distinguish model uncertainty from pipeline failure.

## 13. Milestones

### Milestone 1: Evaluation Skeleton

- Load and validate the public sample dataset.
- Define experiment configuration conventions.
- Produce a minimal run artefact with questions, answers, retrieved context, and metadata.

### Milestone 2: Baseline QA and Metrics

- Implement a baseline document-grounded QA flow.
- Add retrieval and answer-quality metrics.
- Produce a first structured report over a small dataset slice.

### Milestone 3: Failure Analysis and Comparisons

- Add the failure taxonomy and per-question review fields.
- Compare at least two model, prompt, or retrieval configurations.
- Summarise findings as technical recommendations.

### Milestone 4: Portfolio-ready Research Report

- Polish documentation, examples, and reproducibility instructions.
- Include representative successes, failures, limitations, and next experiments.
- Keep claims bounded to the actual public sample and recorded results.

## 14. Risks and Mitigations

- **Small public sample size:** Treat results as research signals, not leaderboard claims.
- **LLM-as-judge unreliability:** Combine automated scoring with deterministic checks and human-reviewable examples.
- **Provider variability:** Record model versions, generation settings, and run metadata.
- **Data availability:** Keep PDFs and generated artefacts outside the repository unless licensing and size constraints are clear.
- **Overfitting to benchmark examples:** Prioritise methodology, diagnostics, and transparent limitations over inflated scores.

## 15. Employer-facing Evidence

The finished project should make the following capabilities visible to reviewers:

- Designing evaluation frameworks for AI systems in a domain where correctness and trust matter.
- Turning ambiguous model-quality questions into measurable experiments.
- Using Python and AI tooling to build reproducible evaluation workflows.
- Diagnosing failures across retrieval, reasoning, grounding, and refusal behaviour.
- Communicating findings as practical recommendations rather than raw metrics alone.
- Maintaining credible data boundaries and avoiding exaggerated performance claims.
