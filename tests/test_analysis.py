from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.analysis import (
    categorize_join_row,
    join_retrieval_and_answer_scores,
    summarize_joined_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _retrieval_row(question_id: str, *, page_hit: bool = True, k: int = 5) -> dict:
    return {
        "question_id": question_id,
        f"doc_hit@{k}": page_hit,
        f"page_hit@{k}": page_hit,
        f"evidence_text_hit@{k}": page_hit,
        "best_evidence_overlap": 0.85 if page_hit else 0.10,
        "page_first_hit_rank": 2 if page_hit else None,
    }


def _answer_row(
    question_id: str,
    *,
    numeric_match: bool = True,
    verdict: str | None = "correct",
) -> dict:
    return {
        "question_id": question_id,
        "scores": {
            "exact_match": False,
            "normalized_string_match": numeric_match,
            "numeric_match": numeric_match,
            "unit_match": numeric_match,
            "gold_numeric_values": [1.0],
            "prediction_numeric_values": [1.0],
        },
        "judge": (
            {"status": "success", "verdict": verdict, "reason": "Test reason."}
            if verdict is not None
            else None
        ),
        "status": "success",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Cycle 1 — join produces one row per retrieval question
# ---------------------------------------------------------------------------


def test_join_returns_one_row_per_retrieval_question() -> None:
    retrieval = [_retrieval_row("q1"), _retrieval_row("q2")]
    answers = [_answer_row("q1"), _answer_row("q2")]

    result = join_retrieval_and_answer_scores(retrieval, answers)

    assert len(result) == 2
    assert {r["question_id"] for r in result} == {"q1", "q2"}


# ---------------------------------------------------------------------------
# Cycle 2 — join merges answer fields by question_id
# ---------------------------------------------------------------------------


def test_join_merges_answer_fields_by_question_id() -> None:
    retrieval = [_retrieval_row("q1", page_hit=True)]
    answers = [_answer_row("q1", numeric_match=True, verdict="correct")]

    result = join_retrieval_and_answer_scores(retrieval, answers, k=5)
    row = result[0]

    assert row["question_id"] == "q1"
    assert row["retrieval"]["page_hit@5"] is True
    assert row["retrieval"]["best_evidence_overlap"] == pytest.approx(0.85)
    assert row["answer"]["numeric_match"] is True
    assert row["answer"]["exact_match"] is False
    assert row["judge_verdict"] == "correct"


# ---------------------------------------------------------------------------
# Cycle 3 — missing answer row → answer: None, judge_verdict: None
# ---------------------------------------------------------------------------


def test_join_handles_missing_answer_row_gracefully() -> None:
    retrieval = [_retrieval_row("q1")]
    answers = [_answer_row("q2")]  # different question_id

    result = join_retrieval_and_answer_scores(retrieval, answers)
    row = result[0]

    assert row["question_id"] == "q1"
    assert row["answer"] is None
    assert row["judge_verdict"] is None


# ---------------------------------------------------------------------------
# Cycles 4-7 — categorize_join_row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("page_hit", "numeric_match", "expected_category"),
    [
        (True, True, "retrieval_hit_answer_correct"),
        (True, False, "retrieval_hit_answer_wrong"),
        (False, True, "retrieval_miss_answer_correct"),
        (False, False, "retrieval_miss_answer_wrong"),
    ],
)
def test_categorize_join_row(
    page_hit: bool, numeric_match: bool, expected_category: str
) -> None:
    row: dict = {
        "retrieval": {"page_hit@5": page_hit},
        "answer": {"numeric_match": numeric_match, "unit_match": False},
    }
    assert categorize_join_row(row, k=5) == expected_category


def test_categorize_treats_missing_answer_as_wrong() -> None:
    row: dict = {
        "retrieval": {"page_hit@5": True},
        "answer": None,
    }
    assert categorize_join_row(row, k=5) == "retrieval_hit_answer_wrong"


def test_categorize_unit_match_counts_as_correct() -> None:
    row: dict = {
        "retrieval": {"page_hit@5": False},
        "answer": {"numeric_match": False, "unit_match": True},
    }
    assert categorize_join_row(row, k=5) == "retrieval_miss_answer_correct"


# ---------------------------------------------------------------------------
# Cycle 8 — summarize counts all four categories
# ---------------------------------------------------------------------------


def test_summarize_counts_all_four_categories() -> None:
    rows = [
        {"category": "retrieval_hit_answer_correct"},
        {"category": "retrieval_hit_answer_wrong"},
        {"category": "retrieval_miss_answer_correct"},
        {"category": "retrieval_miss_answer_wrong"},
    ]

    summary = summarize_joined_metrics(rows)

    assert summary["example_count"] == 4
    assert summary["retrieval_hit_answer_correct"] == 1
    assert summary["retrieval_hit_answer_wrong"] == 1
    assert summary["retrieval_miss_answer_correct"] == 1
    assert summary["retrieval_miss_answer_wrong"] == 1
    assert summary["retrieval_hit_count"] == 2
    assert summary["answer_correct_count"] == 2


# ---------------------------------------------------------------------------
# Cycle 9 — join assigns category field on each row
# ---------------------------------------------------------------------------


def test_join_assigns_category_on_each_row() -> None:
    retrieval = [
        _retrieval_row("hit_correct", page_hit=True),
        _retrieval_row("miss_wrong", page_hit=False),
    ]
    answers = [
        _answer_row("hit_correct", numeric_match=True),
        _answer_row("miss_wrong", numeric_match=False),
    ]

    result = join_retrieval_and_answer_scores(retrieval, answers, k=5)
    by_id = {r["question_id"]: r for r in result}

    assert by_id["hit_correct"]["category"] == "retrieval_hit_answer_correct"
    assert by_id["miss_wrong"]["category"] == "retrieval_miss_answer_wrong"


# ---------------------------------------------------------------------------
# Cycle 10 — CLI join-metrics writes output files (end-to-end)
# ---------------------------------------------------------------------------


def test_cli_join_metrics_writes_output_files(tmp_path: Path) -> None:
    from financebench_eval_harness.cli import main

    retrieval_scores_path = tmp_path / "retrieval_scores.jsonl"
    answer_scores_path = tmp_path / "scores.jsonl"
    output_dir = tmp_path / "analysis"

    retrieval_scores_path.write_text(
        json.dumps(_retrieval_row("q1", page_hit=True)) + "\n"
        + json.dumps(_retrieval_row("q2", page_hit=False)) + "\n",
        encoding="utf-8",
    )
    answer_scores_path.write_text(
        json.dumps(_answer_row("q1", numeric_match=True)) + "\n"
        + json.dumps(_answer_row("q2", numeric_match=False)) + "\n",
        encoding="utf-8",
    )

    exit_code = main([
        "join-metrics",
        "--retrieval-scores", str(retrieval_scores_path),
        "--answer-scores", str(answer_scores_path),
        "--output-dir", str(output_dir),
        "--k", "5",
    ])

    assert exit_code == 0
    joined_path = output_dir / "joined_metrics.jsonl"
    summary_path = output_dir / "joined_summary.json"
    assert joined_path.is_file()
    assert summary_path.is_file()

    rows = [json.loads(line) for line in joined_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2

    by_id = {r["question_id"]: r for r in rows}
    assert by_id["q1"]["category"] == "retrieval_hit_answer_correct"
    assert by_id["q2"]["category"] == "retrieval_miss_answer_wrong"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["example_count"] == 2
    assert summary["retrieval_hit_answer_correct"] == 1
    assert summary["retrieval_miss_answer_wrong"] == 1
