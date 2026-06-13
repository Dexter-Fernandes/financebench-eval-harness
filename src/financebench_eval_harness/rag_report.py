from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financebench_eval_harness.report import _escape_markdown_text, _format_percent, _slug

_FAILURE_LABELS = (
    "retrieval_miss",
    "numeric_error",
    "unsupported_claim",
    "over_refusal",
    "reasoning_error",
)

DEFAULT_RAG_REPORT_OUTPUT_DIR = Path("reports")


@dataclass(frozen=True)
class RagReportResult:
    report_path: Path
    example_count: int


class RagReportError(ValueError):
    """Raised when a RAG report cannot be generated."""


def generate_rag_report(
    joined_dir: Path,
    *,
    rag_run_dir: Path | None = None,
    retrieval_summary_path: Path | None = None,
    output_dir: Path = DEFAULT_RAG_REPORT_OUTPUT_DIR,
    run_id: str | None = None,
) -> RagReportResult:
    rows, summary = _load_joined_data(joined_dir)
    rag_meta = _load_rag_metadata(rag_run_dir)
    predictions_by_id = _load_predictions_by_id(rag_run_dir)
    retrieval_summary = _load_retrieval_summary(retrieval_summary_path)

    resolved_run_id = run_id or joined_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"rag_eval_{_slug(resolved_run_id)}.md"
    report_path = output_dir / report_filename

    report_path.write_text(
        _render_rag_report(
            rows=rows,
            summary=summary,
            rag_meta=rag_meta,
            predictions_by_id=predictions_by_id,
            retrieval_summary=retrieval_summary,
            run_id=resolved_run_id,
        ),
        encoding="utf-8",
    )
    return RagReportResult(
        report_path=report_path,
        example_count=int(summary.get("example_count", 0)),
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_joined_data(joined_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_path = joined_dir / "joined_metrics.jsonl"
    summary_path = joined_dir / "joined_summary.json"
    if not metrics_path.is_file():
        raise RagReportError(f"joined_metrics.jsonl not found: {metrics_path}")
    if not summary_path.is_file():
        raise RagReportError(f"joined_summary.json not found: {summary_path}")
    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    return rows, summary


def _load_rag_metadata(rag_run_dir: Path | None) -> dict[str, Any]:
    if rag_run_dir is None:
        return {}
    path = rag_run_dir / "rag_run_metadata.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions_by_id(rag_run_dir: Path | None) -> dict[str, dict[str, Any]]:
    if rag_run_dir is None:
        return {}
    path = rag_run_dir / "rag_predictions.jsonl"
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("question_id")
        if isinstance(qid, str):
            result[qid] = row
    return result


def _load_retrieval_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_rag_report(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    rag_meta: dict[str, Any],
    predictions_by_id: dict[str, dict[str, Any]],
    retrieval_summary: dict[str, Any],
    run_id: str,
) -> str:
    n = int(summary.get("example_count", len(rows)))
    lines: list[str] = [f"# RAG Evaluation Report — {run_id}", ""]

    # Run Metadata
    lines += ["## Run Metadata", "", "| Setting | Value |", "| --- | --- |"]
    lines.append(f"| Questions evaluated | {n} |")
    if rag_meta.get("model_provider") or rag_meta.get("model_name"):
        provider = rag_meta.get("model_provider", "")
        model = rag_meta.get("model_name", "")
        lines.append(f"| Generator model | {provider} / {model} |")
    if rag_meta.get("top_k") is not None:
        lines.append(f"| Top-k | {rag_meta['top_k']} |")
    if rag_meta.get("retrieval_run_id"):
        lines.append(f"| Retrieval run | {rag_meta['retrieval_run_id']} |")
    # Prompt version from first prediction if available
    if predictions_by_id:
        first_pred = next(iter(predictions_by_id.values()))
        pv = first_pred.get("prompt_version")
        if pv:
            lines.append(f"| Prompt version | {pv} |")
    lines.append("")

    # Answer Accuracy — derive verdict counts from joined rows directly
    verdict_counts: dict[str, int] = {}
    for row in rows:
        v = row.get("judge_verdict")
        if isinstance(v, str):
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

    lines += ["## Answer Accuracy", "", "| Metric | Value |", "| --- | ---: |"]
    if verdict_counts:
        judge_n = sum(verdict_counts.values())
        for label, key in [
            ("Correct", "correct"),
            ("Partially correct", "partially_correct"),
            ("Incorrect", "incorrect"),
            ("Not answered", "not_answered"),
        ]:
            count = verdict_counts.get(key, 0)
            lines.append(f"| {label} | {count} ({_format_percent(count / judge_n)}) |")
        correct_count = verdict_counts.get("correct", 0)
        lines.append(f"| Accuracy | {_format_percent(correct_count / judge_n)} |")
    else:
        answer_correct = int(summary.get("answer_correct_count", 0))
        accuracy = answer_correct / n if n else 0.0
        lines += [
            f"| Correct | {answer_correct} ({_format_percent(accuracy)}) |",
            f"| Wrong | {n - answer_correct} ({_format_percent(1 - accuracy)}) |",
        ]
    lines.append("")

    # Retrieval Performance (from retrieval_summary if provided)
    ret_keys = [k for k in retrieval_summary if k.endswith("_rate") or "@" in k]
    if ret_keys:
        lines += ["## Retrieval Performance", "", "| Metric | Value |", "| --- | --- |"]
        for key in sorted(ret_keys):
            val = retrieval_summary[key]
            if isinstance(val, float):
                lines.append(f"| {key} | {_format_percent(val)} |")
        lines.append("")

    # Retrieval vs Answer Correlation
    rhac = int(summary.get("retrieval_hit_answer_correct", 0))
    rhaw = int(summary.get("retrieval_hit_answer_wrong", 0))
    rmac = int(summary.get("retrieval_miss_answer_correct", 0))
    rmaw = int(summary.get("retrieval_miss_answer_wrong", 0))
    hit_total = rhac + rhaw
    miss_total = rmac + rmaw

    lines += [
        "## Retrieval vs Answer Correlation",
        "",
        "| Category | Count | Rate |",
        "| --- | ---: | ---: |",
        f"| Retrieval hit → correct | {rhac} | {_format_percent(rhac / hit_total) if hit_total else 'N/A'} of hits |",
        f"| Retrieval hit → wrong | {rhaw} | {_format_percent(rhaw / hit_total) if hit_total else 'N/A'} of hits |",
        f"| Retrieval miss → correct | {rmac} | {_format_percent(rmac / miss_total) if miss_total else 'N/A'} of misses |",
        f"| Retrieval miss → wrong | {rmaw} | {_format_percent(rmaw / miss_total) if miss_total else 'N/A'} of misses |",
        "",
    ]

    # Failure Breakdown
    lines += ["## Failure Breakdown", "", "| Label | Count | Rate |", "| --- | ---: | ---: |"]
    for label in _FAILURE_LABELS:
        count = int(summary.get(f"{label}_count", 0))
        rate = count / n if n else 0.0
        lines.append(f"| {label} | {count} | {_format_percent(rate)} |")
    lines.append("")

    # Best Examples
    best = _best_examples(rows, predictions_by_id)
    lines += ["## Best Examples", ""]
    if best:
        for row in best:
            lines += _format_example(row, predictions_by_id)
    else:
        lines.append("No correct examples with retrieval hit found.")
    lines.append("")

    # Worst Examples
    worst = _worst_examples(rows, predictions_by_id)
    lines += ["## Worst Examples", ""]
    if worst:
        for row in worst:
            lines += _format_example(row, predictions_by_id)
    else:
        lines.append("No failed examples found.")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _best_examples(
    rows: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    n: int = 3,
) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("category") == "retrieval_hit_answer_correct"][:n]


def _worst_examples(
    rows: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    n: int = 3,
) -> list[dict[str, Any]]:
    # Prefer retrieval hits with wrong answers (more diagnostic), then misses
    hits_wrong = [r for r in rows if r.get("category") == "retrieval_hit_answer_wrong"]
    misses_wrong = [r for r in rows if r.get("category") == "retrieval_miss_answer_wrong"]
    candidates = hits_wrong + misses_wrong
    return candidates[:n]


def _format_example(
    row: dict[str, Any],
    predictions_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    qid = str(row.get("question_id", "unknown"))
    pred = predictions_by_id.get(qid, {})
    lines = [f"### {qid}", ""]
    if pred.get("question"):
        lines.append(f"- Question: {_escape_markdown_text(str(pred['question']))}")
    if pred.get("gold_answer"):
        lines.append(f"- Gold answer: {_escape_markdown_text(str(pred['gold_answer']))}")
    if pred.get("prediction"):
        lines.append(f"- Prediction: {_escape_markdown_text(str(pred['prediction']))}")
    verdict = row.get("judge_verdict")
    if verdict:
        lines.append(f"- Verdict: {verdict}")
    labels = row.get("failure_labels") or []
    if labels:
        lines.append(f"- Failure labels: {', '.join(str(l) for l in labels)}")
    lines.append("")
    return lines


__all__ = [
    "DEFAULT_RAG_REPORT_OUTPUT_DIR",
    "RagReportError",
    "RagReportResult",
    "generate_rag_report",
]
