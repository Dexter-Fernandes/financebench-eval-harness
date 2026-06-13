"""M5.15 — RAG evaluation integration tests using mock LLM responses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from financebench_eval_harness.analysis import join_retrieval_and_answer_scores
from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import LLMGenerationConfig, MockLLMClient
from financebench_eval_harness.rag_report import generate_rag_report
from financebench_eval_harness.rag_run import RAGRunResult, run_rag_from_config
from financebench_eval_harness.rag_run_config import RAGEvalSettings, RAGRunConfig
from financebench_eval_harness.rag_types import RAGContextChunk, format_retrieved_context
from financebench_eval_harness.run_config import JudgeConfig, JudgePromptConfig
from financebench_eval_harness.scoring import score_prediction


# ---------------------------------------------------------------------------
# Shared fixtures (adapted from test_rag_run.py)
# ---------------------------------------------------------------------------

_EVAL_CONFIG_PATH = Path("configs/evaluation/rag_modes.yaml")

_EXAMPLES = [
    {
        "question_id": "q001",
        "question": "What was the revenue?",
        "gold_answer": "$10 million",
        "evidence": [{"evidence_text": "Revenue was $10 million."}],
    },
    {
        "question_id": "q002",
        "question": "What were the operating expenses?",
        "gold_answer": "$1.8 billion",
        "evidence": [{"evidence_text": "Operating expenses rose to $1.8 billion."}],
    },
]

_RETRIEVAL = [
    {
        "question_id": "q001",
        "retrieved": [
            {
                "rank": 1,
                "chunk_id": "doc_p1_c01",
                "doc_name": "doc.pdf",
                "page_num": 1,
                "text": "Revenue was $10 million.",
                "score": 0.9,
            },
        ],
    },
    {
        "question_id": "q002",
        "retrieved": [
            {
                "rank": 1,
                "chunk_id": "doc_p3_c01",
                "doc_name": "doc.pdf",
                "page_num": 3,
                "text": "Operating expenses rose to $1.8 billion.",
                "score": 0.85,
            },
        ],
    },
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_model_config() -> LLMGenerationConfig:
    return LLMGenerationConfig(
        provider="mock",
        model_name="mock-llm",
        temperature=0.0,
        max_tokens=512,
        timeout_seconds=30.0,
    )


def _make_config(tmp_path: Path, *, judge: JudgeConfig | None = None) -> RAGRunConfig:
    examples_path = tmp_path / "examples.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    _write_jsonl(examples_path, _EXAMPLES)
    _write_jsonl(retrieval_path, _RETRIEVAL)
    return RAGRunConfig(
        settings=RAGEvalSettings(
            examples_path=examples_path,
            retrieval_results_path=retrieval_path,
            output_dir=tmp_path / "runs",
            mode=EvaluationMode.RAG_DENSE,
            top_k=3,
            retrieval_run_id="test-retrieval-run",
            eval_config_path=_EVAL_CONFIG_PATH,
        ),
        model=_make_model_config(),
        judge=judge,
    )


def _make_judge_config() -> JudgeConfig:
    return JudgeConfig(
        enabled=True,
        model=LLMGenerationConfig(
            provider="mock",
            model_name="mock-judge",
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=30.0,
        ),
        prompt=JudgePromptConfig(
            id="answer_correctness_v1",
            version="v1",
            template_path=Path("prompts/judges/answer_correctness_v1.txt"),
        ),
    )


def _make_joined_dir(tmp_path: Path) -> Path:
    joined_dir = tmp_path / "analysis"
    joined_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "question_id": "q001",
            "retrieval": {"page_hit@5": True, "best_evidence_overlap": 0.9},
            "answer": {"numeric_match": True, "unit_match": True},
            "judge_verdict": "correct",
            "judge_numeric_error": None,
            "judge_unsupported_claims": None,
            "category": "retrieval_hit_answer_correct",
            "failure_labels": [],
        },
        {
            "question_id": "q002",
            "retrieval": {"page_hit@5": False, "best_evidence_overlap": 0.1},
            "answer": {"numeric_match": False, "unit_match": False},
            "judge_verdict": "incorrect",
            "judge_numeric_error": None,
            "judge_unsupported_claims": None,
            "category": "retrieval_miss_answer_wrong",
            "failure_labels": ["retrieval_miss"],
        },
    ]
    summary = {
        "example_count": 2,
        "retrieval_hit_answer_correct": 1,
        "retrieval_hit_answer_wrong": 0,
        "retrieval_miss_answer_correct": 0,
        "retrieval_miss_answer_wrong": 1,
        "retrieval_hit_count": 1,
        "answer_correct_count": 1,
        "retrieval_miss_count": 1,
        "numeric_error_count": 0,
        "unsupported_claim_count": 0,
        "over_refusal_count": 0,
        "reasoning_error_count": 0,
        "no_failure_label_count": 1,
    }
    (joined_dir / "joined_metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (joined_dir / "joined_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return joined_dir


# ---------------------------------------------------------------------------
# Test 1 — RAG prompt is built from retrieved chunks
# ---------------------------------------------------------------------------


def test_rag_prompt_built_from_retrieved_chunks(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    mock = MockLLMClient(config.model, responses=["mock answer"] * 10)

    run_rag_from_config(config, mock, run_id="t1")

    assert len(mock.calls) >= 1
    first_prompt = mock.calls[0]
    assert "What was the revenue?" in first_prompt
    assert "Revenue was $10 million" in first_prompt


# ---------------------------------------------------------------------------
# Test 2 — context block is formatted correctly
# ---------------------------------------------------------------------------


def test_context_formatted_correctly() -> None:
    chunk = RAGContextChunk(
        rank=1,
        chunk_id="c1",
        doc_name="doc.pdf",
        page_num=3,
        text="Revenue was $10M.",
        score=0.9,
    )
    ctx = format_retrieved_context([chunk])

    assert "doc.pdf" in ctx
    assert "3" in ctx
    assert "Revenue was $10M" in ctx


# ---------------------------------------------------------------------------
# Test 3 — mock LLM generates prediction written to rag_predictions.jsonl
# ---------------------------------------------------------------------------


def test_mock_llm_generates_prediction(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    mock = MockLLMClient(config.model, responses=["$10 million", "$1.8 billion"])

    result = run_rag_from_config(config, mock, run_id="t3")

    rows = _read_jsonl(result.predictions_path)
    assert rows[0]["question_id"] == "q001"
    assert rows[0]["prediction"] == "$10 million"
    assert rows[1]["prediction"] == "$1.8 billion"


# ---------------------------------------------------------------------------
# Test 4 — predictions are saved as JSONL with required fields
# ---------------------------------------------------------------------------


def test_predictions_saved_as_jsonl(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(
        config, MockLLMClient(config.model, responses=["ans"] * 10), run_id="t4"
    )

    assert result.predictions_path.is_file()
    rows = _read_jsonl(result.predictions_path)
    assert len(rows) == 2
    for row in rows:
        assert "question_id" in row
        assert "question" in row
        assert "gold_answer" in row
        assert "prediction" in row
        assert "context_chunk_ids" in row
        assert "status" in row
        assert row["status"] == "success"


# ---------------------------------------------------------------------------
# Test 5 — numeric scoring works correctly
# ---------------------------------------------------------------------------


def test_numeric_scoring_works() -> None:
    correct = score_prediction("$10 million", "$10 million")
    assert correct["numeric_match"] is True

    wrong_value = score_prediction("$10 million", "$5 million")
    assert wrong_value["numeric_match"] is False

    # Unit equivalence: 1 billion == 1000 million
    unit_equiv = score_prediction("$1 billion", "$1,000 million")
    assert unit_equiv["unit_match"] is True


# ---------------------------------------------------------------------------
# Test 6 — grounding labels (judge verdict) work on simple examples
# ---------------------------------------------------------------------------


def test_grounding_labels_work_on_simple_examples(tmp_path: Path) -> None:
    judge_cfg = _make_judge_config()
    config = _make_config(tmp_path, judge=judge_cfg)
    mock = MockLLMClient(config.model, responses=["$10 million", "$1.8 billion"])
    judge_mock = MockLLMClient(
        judge_cfg.model,
        responses=[
            '{"verdict": "correct", "reason": "Matches gold."}',
            '{"verdict": "correct", "reason": "Matches gold."}',
        ],
    )

    result = run_rag_from_config(config, mock, judge_client=judge_mock, run_id="t6")

    score_rows = _read_jsonl(result.scores_path)
    assert score_rows[0]["judge"]["verdict"] == "correct"
    assert score_rows[0]["judge"]["status"] == "success"


# ---------------------------------------------------------------------------
# Test 7 — combined retrieval + answer scores are saved
# ---------------------------------------------------------------------------


def test_combined_retrieval_and_answer_scores_are_saved() -> None:
    retrieval_scores = [
        {
            "question_id": "q001",
            "page_hit@5": True,
            "doc_hit@5": True,
            "evidence_text_hit@5": True,
            "best_evidence_overlap": 0.9,
            "page_first_hit_rank": 1,
        }
    ]
    answer_scores = [
        {
            "question_id": "q001",
            "scores": {
                "exact_match": True,
                "normalized_string_match": True,
                "numeric_match": True,
                "unit_match": True,
                "gold_numeric_values": [10.0],
            },
            "judge": {"verdict": "correct", "numeric_error": False, "unsupported_claims": False},
            "status": "success",
        }
    ]

    rows = join_retrieval_and_answer_scores(retrieval_scores, answer_scores, k=5)

    assert len(rows) == 1
    row = rows[0]
    assert "retrieval" in row
    assert "answer" in row
    assert row["judge_verdict"] == "correct"
    assert row["category"] == "retrieval_hit_answer_correct"
    assert isinstance(row["failure_labels"], list)


# ---------------------------------------------------------------------------
# Test 8 — report file is created
# ---------------------------------------------------------------------------


def test_report_file_is_created(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)

    result = generate_rag_report(joined_dir, output_dir=tmp_path / "reports", run_id="smoke")

    assert result.report_path.is_file()
    assert result.report_path.suffix == ".md"
    assert result.example_count == 2
    text = result.report_path.read_text(encoding="utf-8")
    assert "## Answer Accuracy" in text
    assert "## Failure Breakdown" in text
