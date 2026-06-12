from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
import re
from typing import Any


DEFAULT_REPORT_OUTPUT_DIR = Path("reports/generated")


@dataclass(frozen=True)
class BaselineReportResult:
    report_path: Path
    run_dir: Path
    evaluated_count: int


class BaselineReportError(ValueError):
    """Raised when a baseline report cannot be generated."""


def generate_baseline_report(
    run_dir: str | Path,
    *,
    output_dir: str | Path = DEFAULT_REPORT_OUTPUT_DIR,
) -> BaselineReportResult:
    resolved_run_dir = Path(run_dir)
    metadata = _load_run_metadata(resolved_run_dir)
    rows = _load_output_rows(resolved_run_dir)
    judge_summary = _judge_summary_from_metadata(metadata)

    mode = _metadata_string(metadata, "mode")
    model_provider = _metadata_string(metadata, "model_provider")
    model_name = _metadata_string(metadata, "model_name")
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"baseline_{_slug(mode)}_{_slug(model_name)}.md"

    report_path.write_text(
        _render_markdown_report(
            metadata=metadata,
            rows=rows,
            judge_summary=judge_summary,
            mode=mode,
            model_provider=model_provider,
            model_name=model_name,
        ),
        encoding="utf-8",
    )
    return BaselineReportResult(
        report_path=report_path,
        run_dir=resolved_run_dir,
        evaluated_count=_int_value(judge_summary, "attempted_count"),
    )


def _load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise BaselineReportError(
            f"Baseline report run metadata file not found: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise BaselineReportError(
            f"Baseline report run metadata was not valid JSON: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise BaselineReportError(
            f"Baseline report run metadata must be a JSON object: {metadata_path}"
        )
    return metadata


def _load_output_rows(run_dir: Path) -> list[dict[str, Any]]:
    outputs_path = run_dir / "outputs.jsonl"
    if not outputs_path.is_file():
        raise BaselineReportError(
            f"Baseline report outputs file not found: {outputs_path}"
        )
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            outputs_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise BaselineReportError(
                    f"Baseline report output row must be an object at line {line_number}"
                )
            rows.append(row)
    except JSONDecodeError as exc:
        raise BaselineReportError(
            f"Baseline report outputs file was not valid JSONL at line {line_number}"
        ) from exc
    return rows


def _judge_summary_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    judge_summary = metadata.get("judge_summary")
    if not isinstance(judge_summary, dict):
        raise BaselineReportError(
            "Baseline report requires judge_summary in run metadata"
        )
    return judge_summary


def _render_markdown_report(
    *,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    judge_summary: dict[str, Any],
    mode: str,
    model_provider: str,
    model_name: str,
) -> str:
    evaluated_count = _int_value(judge_summary, "attempted_count")
    correct_count = _int_value(judge_summary, "correct_count")
    partial_count = _int_value(judge_summary, "partially_correct_count")
    incorrect_count = _int_value(judge_summary, "incorrect_count")
    not_answered_count = _int_value(judge_summary, "not_answered_count")
    judge_error_count = _int_value(judge_summary, "error_count")
    accuracy = correct_count / evaluated_count if evaluated_count else 0.0

    lines = [
        f"# Baseline Report: {mode} / {model_name}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Questions evaluated | {evaluated_count} |",
        f"| Model | {model_provider} / {model_name} |",
        f"| Evaluation mode | {mode} |",
        f"| Correct | {correct_count} |",
        f"| Partially correct | {partial_count} |",
        f"| Incorrect | {incorrect_count} |",
        f"| Not answered | {not_answered_count} |",
        f"| Judge errors | {judge_error_count} |",
        f"| Accuracy estimate | {_format_percent(accuracy)} |",
        f"| Avg latency | {_format_seconds(_average_numeric(rows, 'latency_ms'))} |",
        f"| Avg input tokens | {_format_number(_average_numeric(rows, 'input_tokens'))} |",
        f"| Avg output tokens | {_format_number(_average_numeric(rows, 'output_tokens'))} |",
        "",
        "## Common Failure Examples",
        "",
    ]

    failure_examples = _failure_examples(rows)
    if not failure_examples:
        lines.append("No judge-marked failure examples found.")
    else:
        for row in failure_examples:
            judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
            lines.extend(
                [
                    f"### {row.get('question_id', 'unknown')}",
                    "",
                    f"- Verdict: {judge.get('verdict', 'unknown')}",
                    f"- Question: {_escape_markdown_text(str(row.get('question', '')))}",
                    f"- Gold answer: {_escape_markdown_text(str(row.get('gold_answer', '')))}",
                    f"- Prediction: {_escape_markdown_text(str(row.get('prediction', '')))}",
                    f"- Judge reason: {_escape_markdown_text(str(judge.get('reason', '')))}",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def _failure_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prioritized_verdicts = ("incorrect", "not_answered", "partially_correct")
    examples: list[dict[str, Any]] = []
    for verdict in prioritized_verdicts:
        for row in rows:
            judge = row.get("judge")
            if isinstance(judge, dict) and judge.get("verdict") == verdict:
                examples.append(row)
                if len(examples) == 5:
                    return examples
    return examples


def _average_numeric(rows: list[dict[str, Any]], field_name: str) -> float | None:
    values = [
        float(row[field_name])
        for row in rows
        if type(row.get(field_name)) in (int, float)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _metadata_string(metadata: dict[str, Any], field_name: str) -> str:
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise BaselineReportError(
            f"Baseline report metadata missing string field: {field_name}"
        )
    return value


def _int_value(mapping: dict[str, Any], field_name: str) -> int:
    value = mapping.get(field_name)
    if type(value) is not int:
        raise BaselineReportError(
            f"Baseline report summary missing integer field: {field_name}"
        )
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "unknown"


def _format_percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 1000:.1f}s"


def _format_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _escape_markdown_text(value: str) -> str:
    return value.replace("\n", " ").strip()


__all__ = [
    "BaselineReportError",
    "BaselineReportResult",
    "DEFAULT_REPORT_OUTPUT_DIR",
    "generate_baseline_report",
]
