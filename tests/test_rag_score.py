"""Tests for score_rag_run (score-rag command)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from financebench_eval_harness.llm import LLMGenerationConfig, MockLLMClient
from financebench_eval_harness.rag_score import RAGScoreError, RAGScoreResult, score_rag_run
from financebench_eval_harness.rag_score_config import RAGScoreConfig, RAGScoreSettings
from financebench_eval_harness.run_config import JudgeConfig, JudgePromptConfig


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_PREDICTIONS = [
    {
        "question_id": "q001",
        "question": "What was the revenue?",
        "gold_answer": "$10 million",
        "prediction": "$10 million",
        "context_chunk_ids": ["doc_p1_c01"],
        "status": "success",
    },
    {
        "question_id": "q002",
        "question": "What were the operating expenses?",
        "gold_answer": "$1.8 billion",
        "prediction": "$5 million",
        "context_chunk_ids": ["doc_p3_c01"],
        "status": "success",
    },
]

_RETRIEVAL = [
    {
        "question_id": "q001",
        "retrieved": [
            {
                "rank": 1,
                "chunk_id": "doc_p1_c01",
                "doc_name": "annual_report.pdf",
                "page_num": 1,
                "text": "Revenue was $10 million.",
                "score": 0.9,
            }
        ],
    },
    {
        "question_id": "q002",
        "retrieved": [
            {
                "rank": 1,
                "chunk_id": "doc_p3_c01",
                "doc_name": "annual_report.pdf",
                "page_num": 3,
                "text": "Operating expenses rose to $1.8 billion.",
                "score": 0.85,
            }
        ],
    },
]

_ANSWER_JUDGE_PROMPT = Path("prompts/judges/answer_correctness_v2.txt")
_GROUNDING_JUDGE_PROMPT = Path("prompts/judges/answer_grounding_v1.txt")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    _write_jsonl(run_dir / "rag_predictions.jsonl", _PREDICTIONS)
    return run_dir


def _make_retrieval_file(tmp_path: Path) -> Path:
    ret_path = tmp_path / "retrieval_results.jsonl"
    _write_jsonl(ret_path, _RETRIEVAL)
    return ret_path


def _make_model_config(name: str = "mock-llm") -> LLMGenerationConfig:
    return LLMGenerationConfig(
        provider="mock",
        model_name=name,
        temperature=0.0,
        max_tokens=256,
        timeout_seconds=30.0,
    )


def _make_judge_config(prompt_path: Path) -> JudgeConfig:
    return JudgeConfig(
        enabled=True,
        model=_make_model_config("mock-judge"),
        prompt=JudgePromptConfig(
            id="test_judge",
            version="v1",
            template_path=prompt_path,
        ),
    )


def _make_config(
    run_dir: Path,
    *,
    retrieval_results_path: Path | None = None,
    answer_judge: JudgeConfig | None = None,
    grounding_judge: JudgeConfig | None = None,
) -> RAGScoreConfig:
    return RAGScoreConfig(
        settings=RAGScoreSettings(
            run_dir=run_dir,
            retrieval_results_path=retrieval_results_path,
        ),
        answer_judge=answer_judge,
        grounding_judge=grounding_judge,
    )


# ---------------------------------------------------------------------------
# Test 1 — reads predictions from run dir
# ---------------------------------------------------------------------------


def test_reads_predictions_from_run_dir(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    config = _make_config(run_dir)

    result = score_rag_run(config)

    assert isinstance(result, RAGScoreResult)
    assert result.example_count == 2
    assert result.scored_count == 2
    assert result.error_count == 0


# ---------------------------------------------------------------------------
# Test 2 — answer scores file written with required fields
# ---------------------------------------------------------------------------


def test_answer_scores_written_to_jsonl(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    config = _make_config(run_dir)

    result = score_rag_run(config)

    assert result.answer_scores_path.is_file()
    rows = _read_jsonl(result.answer_scores_path)
    assert len(rows) == 2
    required = {"question_id", "gold_answer", "prediction", "exact_match", "numeric_match", "status"}
    for row in rows:
        assert required.issubset(row.keys()), f"Missing fields in: {row}"


# ---------------------------------------------------------------------------
# Test 3 — lexical scores computed correctly
# ---------------------------------------------------------------------------


def test_lexical_scores_correct(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    config = _make_config(run_dir)

    result = score_rag_run(config)

    rows = _read_jsonl(result.answer_scores_path)
    q001 = next(r for r in rows if r["question_id"] == "q001")
    q002 = next(r for r in rows if r["question_id"] == "q002")

    assert q001["numeric_match"] is True
    assert q001["unit_match"] is True
    assert q002["numeric_match"] is False


# ---------------------------------------------------------------------------
# Test 4 — answer judge called with question and prediction text
# ---------------------------------------------------------------------------


def test_answer_judge_called_with_question_and_prediction(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    judge_cfg = _make_judge_config(_ANSWER_JUDGE_PROMPT)
    mock_judge = MockLLMClient(
        judge_cfg.model,
        responses=[
            '{"verdict": "correct", "reason": "Matches.", "numeric_error": false, "unsupported_claims": false}',
            '{"verdict": "incorrect", "reason": "Wrong value.", "numeric_error": true, "unsupported_claims": false}',
        ],
    )
    config = _make_config(run_dir, answer_judge=judge_cfg)

    score_rag_run(config, answer_judge_client=mock_judge)

    assert len(mock_judge.calls) == 2
    assert "What was the revenue?" in mock_judge.calls[0]
    assert "$10 million" in mock_judge.calls[0]


# ---------------------------------------------------------------------------
# Test 5 — answer verdict written to answer scores
# ---------------------------------------------------------------------------


def test_answer_verdict_written_to_answer_scores(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    judge_cfg = _make_judge_config(_ANSWER_JUDGE_PROMPT)
    mock_judge = MockLLMClient(
        judge_cfg.model,
        responses=[
            '{"verdict": "correct", "reason": "Good.", "numeric_error": false, "unsupported_claims": false}',
            '{"verdict": "incorrect", "reason": "Bad.", "numeric_error": true, "unsupported_claims": false}',
        ],
    )
    config = _make_config(run_dir, answer_judge=judge_cfg)

    result = score_rag_run(config, answer_judge_client=mock_judge)

    rows = _read_jsonl(result.answer_scores_path)
    q001 = next(r for r in rows if r["question_id"] == "q001")
    assert q001["answer_verdict"] == "correct"
    assert q001["answer_judge_status"] == "success"


# ---------------------------------------------------------------------------
# Test 6 — grounding judge called with retrieved context text
# ---------------------------------------------------------------------------


def test_grounding_judge_called_with_retrieved_context(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    ret_path = _make_retrieval_file(tmp_path)
    grounding_cfg = _make_judge_config(_GROUNDING_JUDGE_PROMPT)
    mock_grounding = MockLLMClient(
        grounding_cfg.model,
        responses=[
            '{"verdict": "grounded", "citation_correct": null, "unsupported_claims": null, "reason": "Supported."}',
            '{"verdict": "ungrounded", "citation_correct": null, "unsupported_claims": "Wrong value cited.", "reason": "Not in context."}',
        ],
    )
    config = _make_config(run_dir, retrieval_results_path=ret_path, grounding_judge=grounding_cfg)

    score_rag_run(config, grounding_judge_client=mock_grounding)

    assert len(mock_grounding.calls) == 2
    assert "Revenue was $10 million" in mock_grounding.calls[0]


# ---------------------------------------------------------------------------
# Test 7 — grounding verdict written to grounding scores file
# ---------------------------------------------------------------------------


def test_grounding_verdict_written_to_grounding_scores(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    ret_path = _make_retrieval_file(tmp_path)
    grounding_cfg = _make_judge_config(_GROUNDING_JUDGE_PROMPT)
    mock_grounding = MockLLMClient(
        grounding_cfg.model,
        responses=[
            '{"verdict": "grounded", "citation_correct": null, "unsupported_claims": null, "reason": "OK."}',
            '{"verdict": "ungrounded", "citation_correct": null, "unsupported_claims": null, "reason": "Not supported."}',
        ],
    )
    config = _make_config(run_dir, retrieval_results_path=ret_path, grounding_judge=grounding_cfg)

    result = score_rag_run(config, grounding_judge_client=mock_grounding)

    assert result.grounding_scores_path.is_file()
    rows = _read_jsonl(result.grounding_scores_path)
    q001 = next(r for r in rows if r["question_id"] == "q001")
    assert q001["grounding_verdict"] == "grounded"
    assert q001["grounding_judge_status"] == "success"


# ---------------------------------------------------------------------------
# Test 8 — combined scores merge answer and grounding fields
# ---------------------------------------------------------------------------


def test_combined_scores_merge_answer_and_grounding(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    ret_path = _make_retrieval_file(tmp_path)
    judge_cfg = _make_judge_config(_ANSWER_JUDGE_PROMPT)
    grounding_cfg = _make_judge_config(_GROUNDING_JUDGE_PROMPT)
    mock_judge = MockLLMClient(
        judge_cfg.model,
        responses=[
            '{"verdict": "correct", "reason": "OK.", "numeric_error": false, "unsupported_claims": false}',
            '{"verdict": "incorrect", "reason": "Bad.", "numeric_error": true, "unsupported_claims": false}',
        ],
    )
    mock_grounding = MockLLMClient(
        grounding_cfg.model,
        responses=[
            '{"verdict": "grounded", "citation_correct": null, "unsupported_claims": null, "reason": "Good."}',
            '{"verdict": "ungrounded", "citation_correct": null, "unsupported_claims": null, "reason": "No."}',
        ],
    )
    config = _make_config(
        run_dir,
        retrieval_results_path=ret_path,
        answer_judge=judge_cfg,
        grounding_judge=grounding_cfg,
    )

    result = score_rag_run(
        config, answer_judge_client=mock_judge, grounding_judge_client=mock_grounding
    )

    assert result.combined_scores_path.is_file()
    rows = _read_jsonl(result.combined_scores_path)
    q001 = next(r for r in rows if r["question_id"] == "q001")
    assert "answer_verdict" in q001
    assert "grounding_verdict" in q001
    assert q001["answer_verdict"] == "correct"
    assert q001["grounding_verdict"] == "grounded"


# ---------------------------------------------------------------------------
# Test 9 — failure labels assigned in combined scores
# ---------------------------------------------------------------------------


def test_failure_labels_assigned_in_combined(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    # Add a third row: refusal on a numeric question
    refusal_row = {
        "question_id": "q003",
        "question": "What was net income?",
        "gold_answer": "$500 million",
        "prediction": "I don't know.",
        "context_chunk_ids": [],
        "status": "success",
    }
    with (run_dir / "rag_predictions.jsonl").open("a", encoding="utf-8") as f:
        f.write("\n" + json.dumps(refusal_row))

    judge_cfg = _make_judge_config(_ANSWER_JUDGE_PROMPT)
    mock_judge = MockLLMClient(
        judge_cfg.model,
        responses=[
            '{"verdict": "correct", "reason": "OK.", "numeric_error": false, "unsupported_claims": false}',
            '{"verdict": "incorrect", "reason": "Bad.", "numeric_error": true, "unsupported_claims": false}',
            '{"verdict": "not_answered", "reason": "Refused.", "numeric_error": false, "unsupported_claims": false}',
        ],
    )
    config = _make_config(run_dir, answer_judge=judge_cfg)

    result = score_rag_run(config, answer_judge_client=mock_judge)

    rows = _read_jsonl(result.combined_scores_path)
    q001 = next(r for r in rows if r["question_id"] == "q001")
    q002 = next(r for r in rows if r["question_id"] == "q002")
    q003 = next(r for r in rows if r["question_id"] == "q003")

    assert isinstance(q001["failure_labels"], list)
    assert q001["failure_labels"] == []
    assert "numeric_error" in q002["failure_labels"]

    # Refusal: over_refusal fires, numeric_error must NOT fire even though gold has numbers
    assert "over_refusal" in q003["failure_labels"]
    assert "numeric_error" not in q003["failure_labels"]


# ---------------------------------------------------------------------------
# Test 10 — CLI returns 0 and writes all three output files
# ---------------------------------------------------------------------------


def test_cli_score_rag_returns_0(tmp_path: Path) -> None:
    from financebench_eval_harness.cli import main

    run_dir = _make_run_dir(tmp_path)

    config_path = tmp_path / "rag_score.yaml"
    config_path.write_text(
        f"rag_score:\n  run_dir: {run_dir}\n  k: 5\n",
        encoding="utf-8",
    )

    exit_code = main(["score-rag", "--config", str(config_path)])

    assert exit_code == 0
    assert (run_dir / "rag_answer_scores.jsonl").is_file()
    assert (run_dir / "rag_grounding_scores.jsonl").is_file()
    assert (run_dir / "rag_combined_scores.jsonl").is_file()
