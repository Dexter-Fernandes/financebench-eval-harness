"""M7.13: Inspect a single failed example from a grounding analysis run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FailureInspectionResult:
    question_id: str
    question: str | None
    gold_answer: str | None
    prediction: str | None
    answer_verdict: str | None
    grounding_label: str | None
    root_cause: str | None
    failure_types: list[str]
    context_sufficiency: str | None
    citation_quality: str | None
    cited_chunk_ids: list[str]
    gold_evidence: str | None
    retrieved_chunks: list[dict[str, Any]]
    judge_reason: str | None
    rule_flags: list[str]


def load_failure_inspection(run_dir: Path, question_id: str) -> FailureInspectionResult:
    """Load all grounding analysis signals for one question from a run directory."""
    failure_row = _find_row(run_dir / "failure_analysis.jsonl", question_id)
    grounding_row = _find_row(run_dir / "grounding_scores.jsonl", question_id)
    citation_row = _find_row(run_dir / "citation_scores.jsonl", question_id)
    prediction_row = _find_row(run_dir / "rag_predictions.jsonl", question_id)

    retrieved_chunks: list[dict[str, Any]] = []
    if prediction_row:
        rc = prediction_row.get("retrieved_chunks")
        if isinstance(rc, list):
            retrieved_chunks = rc

    return FailureInspectionResult(
        question_id=question_id,
        question=_str_or_none(prediction_row, "question") if prediction_row else None,
        gold_answer=_str_or_none(prediction_row, "gold_answer") if prediction_row else None,
        prediction=_str_or_none(prediction_row, "prediction") if prediction_row else None,
        answer_verdict=_str_or_none(failure_row, "answer_verdict") if failure_row else None,
        grounding_label=_str_or_none(failure_row, "grounding_label") if failure_row else None,
        root_cause=_str_or_none(failure_row, "root_cause") if failure_row else None,
        failure_types=list(failure_row.get("failure_types") or []) if failure_row else [],
        context_sufficiency=_str_or_none(failure_row, "context_sufficiency") if failure_row else None,
        citation_quality=_str_or_none(citation_row, "citation_quality") if citation_row else None,
        cited_chunk_ids=list(citation_row.get("cited_chunk_ids") or []) if citation_row else [],
        gold_evidence=_str_or_none(prediction_row, "gold_evidence") if prediction_row else None,
        retrieved_chunks=retrieved_chunks,
        judge_reason=_str_or_none(grounding_row, "judge_reason") if grounding_row else None,
        rule_flags=list(grounding_row.get("rule_flags") or []) if grounding_row else [],
    )


def format_failure_inspection(result: FailureInspectionResult) -> str:
    """Format a FailureInspectionResult for terminal display."""
    lines = [
        f"=== Failure Inspection: {result.question_id} ===",
        "",
        f"Question:         {result.question or '(not available)'}",
        f"Gold Answer:      {result.gold_answer or '(not available)'}",
        f"Generated Answer: {result.prediction or '(not available)'}",
        "",
        f"Answer Verdict:   {result.answer_verdict or '—'}",
        f"Grounding Label:  {result.grounding_label or '—'}",
        f"Root Cause:       {result.root_cause or '—'}",
        f"Failure Types:    {', '.join(result.failure_types) or 'none'}",
        "",
        f"Context Sufficiency: {result.context_sufficiency or '—'}",
        f"Citation Quality:    {result.citation_quality or '—'}",
        f"Cited Chunks:        {', '.join(result.cited_chunk_ids) or 'none'}",
        "",
        f"Gold Evidence:",
        f"  {result.gold_evidence or '(not available)'}",
        "",
        "Retrieved Context:",
    ]
    if result.retrieved_chunks:
        for i, chunk in enumerate(result.retrieved_chunks[:5], start=1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = str(chunk.get("text", ""))[:200]
            lines.append(f"  [{i}] {chunk_id}: {text}")
    else:
        lines.append("  (none)")
    lines += [
        "",
        f"Judge Reason:     {result.judge_reason or '(no judge run)'}",
        f"Rule-based Flags: {', '.join(result.rule_flags) or 'none'}",
        "",
    ]
    return "\n".join(lines)


def _find_row(path: Path, question_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if str(row.get("question_id", "")) == question_id:
                return row
        except json.JSONDecodeError:
            pass
    return None


def _str_or_none(row: dict[str, Any], key: str) -> str | None:
    val = row.get(key)
    return str(val) if val is not None else None


__all__ = [
    "FailureInspectionResult",
    "format_failure_inspection",
    "load_failure_inspection",
]
