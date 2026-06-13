from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.rag_report import RagReportError, generate_rag_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _joined_row(
    question_id: str,
    *,
    category: str = "retrieval_hit_answer_correct",
    judge_verdict: str | None = "correct",
    failure_labels: list[str] | None = None,
) -> dict:
    page_hit = category.startswith("retrieval_hit")
    return {
        "question_id": question_id,
        "retrieval": {"page_hit@5": page_hit, "best_evidence_overlap": 0.8},
        "answer": {
            "exact_match": False,
            "normalized_string_match": True,
            "numeric_match": True,
            "unit_match": True,
            "gold_numeric_values": [100.0],
        },
        "judge_verdict": judge_verdict,
        "judge_numeric_error": None,
        "judge_unsupported_claims": None,
        "category": category,
        "failure_labels": failure_labels or [],
    }


def _make_joined_dir(
    tmp_path: Path,
    *,
    rows: list[dict] | None = None,
    summary: dict | None = None,
) -> Path:
    if rows is None:
        rows = [
            _joined_row("q1", category="retrieval_hit_answer_correct", judge_verdict="correct"),
            _joined_row("q2", category="retrieval_hit_answer_wrong", judge_verdict="incorrect", failure_labels=["reasoning_error"]),
            _joined_row("q3", category="retrieval_miss_answer_wrong", judge_verdict="incorrect", failure_labels=["retrieval_miss"]),
        ]
    if summary is None:
        summary = {
            "example_count": 3,
            "retrieval_hit_answer_correct": 1,
            "retrieval_hit_answer_wrong": 1,
            "retrieval_miss_answer_correct": 0,
            "retrieval_miss_answer_wrong": 1,
            "retrieval_hit_count": 2,
            "answer_correct_count": 1,
            "retrieval_miss_count": 1,
            "numeric_error_count": 0,
            "unsupported_claim_count": 0,
            "over_refusal_count": 0,
            "reasoning_error_count": 1,
            "no_failure_label_count": 1,
        }
    joined_dir = tmp_path / "analysis"
    joined_dir.mkdir()
    (joined_dir / "joined_metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (joined_dir / "joined_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return joined_dir


def _make_rag_run_dir(tmp_path: Path) -> Path:
    rag_dir = tmp_path / "rag_run"
    rag_dir.mkdir()
    metadata = {
        "model_provider": "ollama",
        "model_name": "llama3.2:3b",
        "top_k": 5,
        "retrieval_run_id": "run_001",
        "judge": {"enabled": False},
        "judge_summary": {
            "attempted_count": 0,
            "correct_count": 0,
            "partially_correct_count": 0,
            "incorrect_count": 0,
            "not_answered_count": 0,
            "error_count": 0,
        },
    }
    (rag_dir / "rag_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    predictions = [
        {
            "question_id": "q1",
            "question": "What was revenue?",
            "gold_answer": "$100M",
            "prediction": "$100 million",
            "prompt_version": "v1",
        },
        {
            "question_id": "q2",
            "question": "What was net income?",
            "gold_answer": "$50M",
            "prediction": "$25M",
            "prompt_version": "v1",
        },
        {
            "question_id": "q3",
            "question": "What was capex?",
            "gold_answer": "$30M",
            "prediction": "$60M",
            "prompt_version": "v1",
        },
    ]
    (rag_dir / "rag_predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8"
    )
    return rag_dir


# ---------------------------------------------------------------------------
# Cycle 1 — report writes a markdown file
# ---------------------------------------------------------------------------


def test_generate_rag_report_writes_markdown_file(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)
    output_dir = tmp_path / "reports"

    result = generate_rag_report(joined_dir, output_dir=output_dir, run_id="test-run")

    assert result.report_path.is_file()
    assert result.report_path.suffix == ".md"
    assert result.example_count == 3


# ---------------------------------------------------------------------------
# Cycle 2 — report includes Run Metadata section
# ---------------------------------------------------------------------------


def test_report_includes_run_metadata_section(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)

    result = generate_rag_report(joined_dir, output_dir=tmp_path / "reports", run_id="r1")
    text = result.report_path.read_text(encoding="utf-8")

    assert "## Run Metadata" in text
    assert "Questions evaluated" in text
    assert "3" in text


# ---------------------------------------------------------------------------
# Cycle 3 — report includes Answer Accuracy section
# ---------------------------------------------------------------------------


def test_report_includes_answer_accuracy_table(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)

    result = generate_rag_report(joined_dir, output_dir=tmp_path / "reports")
    text = result.report_path.read_text(encoding="utf-8")

    assert "## Answer Accuracy" in text
    assert "Correct" in text


# ---------------------------------------------------------------------------
# Cycle 4 — report includes Retrieval vs Answer Correlation
# ---------------------------------------------------------------------------


def test_report_includes_retrieval_vs_answer_correlation(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)

    result = generate_rag_report(joined_dir, output_dir=tmp_path / "reports")
    text = result.report_path.read_text(encoding="utf-8")

    assert "## Retrieval vs Answer" in text
    assert "Retrieval hit" in text
    assert "Retrieval miss" in text


# ---------------------------------------------------------------------------
# Cycle 5 — report includes Failure Breakdown
# ---------------------------------------------------------------------------


def test_report_includes_failure_breakdown(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)

    result = generate_rag_report(joined_dir, output_dir=tmp_path / "reports")
    text = result.report_path.read_text(encoding="utf-8")

    assert "## Failure Breakdown" in text
    assert "retrieval_miss" in text
    assert "reasoning_error" in text


# ---------------------------------------------------------------------------
# Cycle 6 — report includes Best and Worst Examples sections
# ---------------------------------------------------------------------------


def test_report_includes_best_and_worst_examples(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)

    result = generate_rag_report(joined_dir, output_dir=tmp_path / "reports")
    text = result.report_path.read_text(encoding="utf-8")

    assert "## Best Examples" in text
    assert "## Worst Examples" in text


# ---------------------------------------------------------------------------
# Cycle 7 — degrades gracefully when rag_run_dir is None
# ---------------------------------------------------------------------------


def test_report_degrades_gracefully_without_rag_run_dir(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)

    result = generate_rag_report(
        joined_dir,
        rag_run_dir=None,
        output_dir=tmp_path / "reports",
    )

    assert result.report_path.is_file()
    text = result.report_path.read_text(encoding="utf-8")
    assert "## Run Metadata" in text


# ---------------------------------------------------------------------------
# Cycle 8 — examples enriched with question text from predictions
# ---------------------------------------------------------------------------


def test_report_enriches_examples_from_predictions(tmp_path: Path) -> None:
    joined_dir = _make_joined_dir(tmp_path)
    rag_dir = _make_rag_run_dir(tmp_path)

    result = generate_rag_report(
        joined_dir,
        rag_run_dir=rag_dir,
        output_dir=tmp_path / "reports",
    )
    text = result.report_path.read_text(encoding="utf-8")

    assert "What was revenue?" in text or "What was net income?" in text


# ---------------------------------------------------------------------------
# Cycle 9 — CLI report-rag command writes file and prints path
# ---------------------------------------------------------------------------


def test_cli_report_rag_writes_file_and_prints_path(tmp_path: Path) -> None:
    from financebench_eval_harness.cli import main

    joined_dir = _make_joined_dir(tmp_path)
    output_dir = tmp_path / "reports"

    exit_code = main([
        "report-rag",
        "--joined-dir", str(joined_dir),
        "--output-dir", str(output_dir),
        "--run-id", "smoke",
    ])

    assert exit_code == 0
    reports = list(output_dir.glob("*.md"))
    assert len(reports) == 1
