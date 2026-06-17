M7: Hallucination and Grounding Analysis

M7 should answer:

When the RAG system gives an answer, is that answer actually supported by the retrieved evidence?

This milestone builds on M5. By this point, your harness can retrieve chunks, generate answers, and score correctness. M7 goes deeper by explaining why answers are wrong, unsupported, or misleading.

This fits your repo goal: evaluating RAG and long-context LLMs on FinanceBench for answer correctness, evidence grounding, hallucination detection, and failure analysis.

M7.1: Define hallucination and grounding labels

Goal: create a clear taxonomy for answer support.

Suggested labels:

Label	Meaning
grounded	Answer is fully supported by retrieved context
partially_grounded	Some claims are supported, but some are missing or vague
ungrounded	Answer is not supported by retrieved context
contradicted	Answer conflicts with retrieved context
insufficient_evidence	Retrieved context does not contain enough information
over_refusal	Model refused even though evidence was available
under_refusal	Model answered even though evidence was insufficient

Done when:

- Each grounding label has a written definition
- Labels are mutually understandable
- The evaluator can assign one primary grounding label per answer
M7.2: Define hallucination failure types

Goal: make failure analysis more specific than just “wrong answer”.

Suggested failure types:

Failure type	Meaning
wrong_number	Incorrect financial value, percentage, year, or count
wrong_unit	Confuses dollars, millions, billions, percentages, etc.
wrong_period	Uses the wrong fiscal year, quarter, or reporting period
wrong_metric	Answers with a related but incorrect financial metric
unsupported_claim	Adds information not present in the retrieved context
contradicted_by_context	Says something the retrieved context disproves
bad_citation	Cites a chunk that does not support the answer
missing_citation	Gives an answer without citing evidence
retrieval_miss	Correct evidence was not retrieved
generation_error	Evidence was present, but the LLM reasoned incorrectly
format_error	Answer is not in the expected format

Done when:

- Each failure type has a definition
- Multiple failure types can be attached to one answer
- Failure labels are saved in structured output
M7.3: Define claim extraction format

Goal: break generated answers into smaller claims that can be checked against evidence.

Example answer:

3M's FY2018 capital expenditure was $1,577 million, based on purchases of property, plant and equipment in the cash flow statement.

Extracted claims:

[
  {
    "claim_id": "claim_001",
    "claim_text": "3M's FY2018 capital expenditure was $1,577 million.",
    "claim_type": "numeric"
  },
  {
    "claim_id": "claim_002",
    "claim_text": "The value comes from purchases of property, plant and equipment in the cash flow statement.",
    "claim_type": "evidence_reference"
  }
]

Done when:

- Generated answers can be split into checkable claims
- Numeric claims are identified
- Citation or evidence claims are identified

For v1, this can be simple. You do not need perfect claim extraction immediately.

M7.4: Add numeric consistency checks

Goal: detect wrong financial values automatically.

FinanceBench is full of numeric answers, so this is one of the highest-value parts of M7.

Checks:

- Does the predicted number match the gold answer?
- Does the predicted number appear in retrieved context?
- Does the unit match?
- Does the scale match?
- Does the fiscal year or period match?

Examples:

Gold	Prediction	Issue
$1,577 million	$1.577 billion	Equivalent, likely correct
$1,577 million	$157.7 million	Scale error
15.2%	15.2 million	Unit error
FY2022	FY2021	Period error

Done when:

- Numeric values are extracted from gold answer, prediction, and context
- Equivalent scales can be normalized where possible
- Wrong number, wrong unit, and wrong period are flagged
M7.5: Add citation extraction

Goal: detect which chunks the model claims to rely on.

If your RAG prompt asks the model to cite chunk IDs, extract them from the answer.

Example answer:

The FY2018 capital expenditure was $1,577 million. [3M_2018_10K_p45_c02]

Parsed output:

{
  "cited_chunk_ids": ["3M_2018_10K_p45_c02"]
}

Done when:

- Cited chunk IDs can be extracted from model answers
- Missing citations are detected
- Invalid chunk IDs are detected
- Citation list is saved with each prediction
M7.6: Add citation support checking

Goal: check whether cited chunks actually support the answer.

For each answer:

answer -> cited chunk IDs -> cited chunk text -> support check

Support labels:

Label	Meaning
supports_answer	Cited chunk contains the answer evidence
partially_supports_answer	Cited chunk is relevant but incomplete
does_not_support_answer	Cited chunk does not justify the answer
citation_missing	No citation provided
citation_invalid	Citation does not map to a retrieved chunk

Done when:

- Each cited chunk is checked against the answer
- Bad citations are flagged
- Citation correctness is saved per answer
M7.7: Add context sufficiency check

Goal: distinguish hallucination from retrieval failure.

Before blaming the LLM, check whether the retrieved context actually contained enough evidence.

Possible labels:

context_sufficient
context_partially_sufficient
context_insufficient

Example distinction:

Case A:
Correct evidence was not retrieved.
Failure = retrieval_miss

Case B:
Correct evidence was retrieved, but model gave wrong answer.
Failure = generation_error

Case C:
Correct evidence was retrieved, answer correct, but citation wrong.
Failure = bad_citation

Done when:

- Each answer has a context sufficiency label
- The evaluator separates retrieval failures from generation failures
- Report shows correctness conditional on context sufficiency
M7.8: Add LLM-as-judge grounding evaluator

Goal: use a judge model to assess support more flexibly than rules alone.

Judge input:

Question
Gold answer
Generated answer
Retrieved context
Cited chunks
Gold evidence, optional

Expected judge output:

{
  "grounding_label": "partially_grounded",
  "failure_types": ["wrong_number", "unsupported_claim"],
  "context_sufficiency": "context_sufficient",
  "citation_quality": "does_not_support_answer",
  "reason": "The retrieved context contains the correct capital expenditure figure, but the answer gives a different value."
}

Done when:

- Grounding judge prompt is versioned
- Judge model is configurable
- Judge returns structured JSON
- Invalid judge responses are logged
M7.9: Add rule-based hallucination checks

Goal: avoid depending only on LLM-as-judge.

Useful rule-based checks:

- predicted number not found in retrieved context
- cited chunk ID does not exist
- answer has no citation
- predicted year not found in retrieved context
- prediction contains unsupported financial terms
- answer says "cannot determine" when gold evidence was retrieved

Done when:

- Rule-based checks produce flags
- Flags are saved alongside judge labels
- Rule-based and judge-based labels can be compared

This makes your harness more credible because it does not blindly trust another LLM.

M7.10: Join retrieval, answer, and grounding results

Goal: create one combined analysis row per question.

Input files:

retrieval_scores.jsonl
rag_predictions.jsonl
rag_answer_scores.jsonl
rag_grounding_scores.jsonl

Combined output example:

{
  "question_id": "financebench_001",
  "page_hit@10": true,
  "evidence_hit@10": true,
  "prediction": "$1,577 million",
  "gold_answer": "$1577.00",
  "answer_verdict": "correct",
  "grounding_label": "grounded",
  "context_sufficiency": "context_sufficient",
  "citation_quality": "supports_answer",
  "failure_types": []
}

Done when:

- Retrieval and generation results are joined by question_id
- Each question has one combined analysis row
- Combined rows can be filtered by failure type

Suggested output:

runs/<run_id>/failure_analysis.jsonl
M7.11: Add failure classification logic

Goal: automatically classify the root cause of each failed example.

Suggested hierarchy:

1. If evidence_hit@k is false:
      root_cause = retrieval_failure

2. If evidence_hit@k is true but answer is wrong:
      root_cause = generation_failure

3. If answer is correct but citation is wrong:
      root_cause = citation_failure

4. If answer is correct and citation supports it:
      root_cause = no_failure

5. If context is insufficient but model answered anyway:
      root_cause = hallucination_under_refusal

6. If context is sufficient but model refused:
      root_cause = over_refusal

Done when:

- Each example receives a root cause label
- Root cause labels are deterministic where possible
- Judge labels can override or enrich rule labels if configured
M7.12: Generate hallucination and grounding report

Goal: produce a readable analysis report.

Report sections:

Run metadata
Dataset size
Retriever used
Generator model
Judge model
Answer correctness summary
Grounding summary
Citation quality summary
Context sufficiency summary
Failure type counts
Root cause breakdown
Examples of grounded correct answers
Examples of retrieval failures
Examples of hallucinations
Examples of bad citations
Recommended next fixes

Example summary table:

Metric	Value
Total questions	100
Correct answers	64
Grounded answers	58
Ungrounded answers	19
Contradicted answers	6
Missing citations	22
Bad citations	11
Retrieval failures	18
Generation failures	12

Done when:

- Markdown report is generated automatically
- Report includes failure counts and example cases
- Report explains whether failures are mostly retrieval or generation related

Suggested output:

reports/hallucination_analysis_<run_id>.md
M7.13: Add hallucination inspection CLI

Goal: inspect a single failed example quickly.

Suggested command:

python -m financebench_eval inspect-failure \
  --run runs/<run_id> \
  --question-id financebench_001

Output should show:

Question
Gold answer
Generated answer
Answer verdict
Grounding label
Root cause
Failure types
Gold evidence
Retrieved context
Cited chunks
Judge reason
Rule-based flags

Done when:

- One failed example can be inspected from the CLI
- Retrieved context and gold evidence are shown side by side
- Failure labels are easy to understand
M7.14: Add failure slicing

Goal: understand which types of questions or documents fail most often.

Useful slices:

company
document type
question type
reasoning type
answer type
numeric vs textual
retrieval success vs retrieval failure
short context vs long context
table-heavy vs prose-heavy evidence

Example outputs:

Accuracy by question_type
Grounding rate by company
Hallucination rate when evidence_hit@10 = false
Wrong-number rate for numeric questions

Done when:

- Failure analysis can group by metadata fields
- Report includes at least 2 useful slices
- Slices help identify what to improve next

This is a strong portfolio feature because it shows real evaluation thinking.

M7.15: Add tests for hallucination analysis

Goal: make sure grounding logic is reliable.

Test cases:

- correct answer with supporting citation is grounded
- correct answer with bad citation is citation_failure
- wrong answer despite evidence present is generation_failure
- wrong answer with evidence missing is retrieval_failure
- answer with unsupported number is unsupported_claim or wrong_number
- refusal with sufficient context is over_refusal
- answer with insufficient context is under_refusal
- invalid cited chunk ID is citation_invalid

Done when:

pytest tests/test_hallucination_analysis.py

passes.

M7 final deliverable

At the end of M7, you should be able to run:

python -m financebench_eval analyze-grounding \
  --config configs/grounding_analysis.yaml

and get:

runs/<run_id>/
  grounding_scores.jsonl
  citation_scores.jsonl
  failure_analysis.jsonl
  failure_summary.json
  grounding_analysis_config.yaml

reports/
  hallucination_analysis_<run_id>.md