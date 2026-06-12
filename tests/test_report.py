import json
from pathlib import Path

import pytest

from financebench_eval_harness.report import (
    BaselineReportError,
    generate_baseline_report,
)


def test_generate_baseline_report_writes_markdown_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fixed-run"
    _write_run(
        run_dir,
        rows=[
            _output_row(
                question_id="q1",
                question="What was revenue?",
                prediction="$123",
                verdict="correct",
                latency_ms=1000,
            ),
            _output_row(
                question_id="q2",
                question="What was gross profit?",
                prediction="$400",
                verdict="incorrect",
                latency_ms=2000,
            ),
            _output_row(
                question_id="q3",
                question="What was operating income?",
                prediction="It was not disclosed.",
                verdict="not_answered",
                latency_ms=3000,
            ),
            _output_row(
                question_id="q4",
                question="What year was cited?",
                prediction="2023",
                verdict="partially_correct",
                latency_ms=4000,
            ),
        ],
    )

    result = generate_baseline_report(run_dir, output_dir=tmp_path / "reports" / "generated")

    assert result.run_dir == run_dir
    assert result.evaluated_count == 4
    assert result.report_path == (
        tmp_path / "reports" / "generated" / "baseline_closed_book_mock-model.md"
    )
    markdown = result.report_path.read_text(encoding="utf-8")
    assert "# Baseline Report: closed_book / mock-model" in markdown
    assert "| Questions evaluated | 4 |" in markdown
    assert "| Model | mock / mock-model |" in markdown
    assert "| Evaluation mode | closed_book |" in markdown
    assert "| Correct | 1 |" in markdown
    assert "| Partially correct | 1 |" in markdown
    assert "| Incorrect | 1 |" in markdown
    assert "| Not answered | 1 |" in markdown
    assert "| Judge errors | 0 |" in markdown
    assert "| Accuracy estimate | 25% |" in markdown
    assert "| Avg latency | 2.5s |" in markdown
    assert "| Avg input tokens | N/A |" in markdown
    assert "| Avg output tokens | N/A |" in markdown
    assert "## Common Failure Examples" in markdown
    assert markdown.index("q2") < markdown.index("q3") < markdown.index("q4")


def test_generate_baseline_report_averages_present_token_usage(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "token-run"
    _write_run(
        run_dir,
        rows=[
            _output_row("q1", "Question 1?", "answer", "correct", input_tokens=10),
            _output_row("q2", "Question 2?", "answer", "correct", input_tokens=None),
            _output_row("q3", "Question 3?", "answer", "correct", output_tokens=6),
        ],
    )

    result = generate_baseline_report(run_dir, output_dir=tmp_path / "reports")

    markdown = result.report_path.read_text(encoding="utf-8")
    assert "| Avg input tokens | 10 |" in markdown
    assert "| Avg output tokens | 6 |" in markdown


def test_generate_baseline_report_reports_missing_run_files(tmp_path: Path) -> None:
    with pytest.raises(BaselineReportError) as exc_info:
        generate_baseline_report(tmp_path / "missing-run")

    assert "Baseline report run metadata file not found" in str(exc_info.value)


def test_generate_baseline_report_requires_judge_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "no-judge-run"
    _write_json(run_dir / "run_metadata.json", {"mode": "closed_book"})
    _write_jsonl(run_dir / "outputs.jsonl", [])

    with pytest.raises(BaselineReportError) as exc_info:
        generate_baseline_report(run_dir)

    assert "Baseline report requires judge_summary in run metadata" in str(exc_info.value)


def _write_run(run_dir: Path, *, rows: list[dict[str, object]]) -> None:
    correct_count = sum(1 for row in rows if row["judge"]["verdict"] == "correct")
    partial_count = sum(
        1 for row in rows if row["judge"]["verdict"] == "partially_correct"
    )
    incorrect_count = sum(1 for row in rows if row["judge"]["verdict"] == "incorrect")
    not_answered_count = sum(
        1 for row in rows if row["judge"]["verdict"] == "not_answered"
    )
    metadata = {
        "mode": "closed_book",
        "model_provider": "mock",
        "model_name": "mock-model",
        "attempted_count": len(rows),
        "judge_summary": {
            "attempted_count": len(rows),
            "success_count": len(rows),
            "error_count": 0,
            "correct_count": correct_count,
            "correct_rate": correct_count / len(rows) if rows else 0.0,
            "partially_correct_count": partial_count,
            "partially_correct_rate": partial_count / len(rows) if rows else 0.0,
            "incorrect_count": incorrect_count,
            "incorrect_rate": incorrect_count / len(rows) if rows else 0.0,
            "not_answered_count": not_answered_count,
            "not_answered_rate": not_answered_count / len(rows) if rows else 0.0,
        },
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    _write_jsonl(run_dir / "outputs.jsonl", rows)


def _output_row(
    question_id: str,
    question: str,
    prediction: str,
    verdict: str,
    *,
    latency_ms: int = 1000,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": question,
        "gold_answer": "Gold answer",
        "prediction": prediction,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "judge": {
            "status": "success",
            "verdict": verdict,
            "reason": f"Judge marked this {verdict}.",
            "error": None,
        },
    }


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
