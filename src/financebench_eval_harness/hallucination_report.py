"""M7.12: Generate hallucination and grounding analysis report."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financebench_eval_harness.grounding_types import (
    GROUNDING_LABELS,
    HALLUCINATION_FAILURE_TYPES,
    ROOT_CAUSE_LABELS,
)


@dataclass(frozen=True)
class HallucinationReportResult:
    report_path: Path
    run_id: str
    total_examples: int


def generate_hallucination_report(
    failure_analysis_path: Path,
    failure_summary_path: Path,
    *,
    run_id: str,
    output_dir: Path = Path("reports"),
) -> HallucinationReportResult:
    """Produce reports/hallucination_analysis_<run_id>.md."""
    rows = _load_jsonl(failure_analysis_path)
    summary = _load_json(failure_summary_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"hallucination_analysis_{run_id}.md"

    content = _render_report(rows, summary, run_id=run_id)
    report_path.write_text(content, encoding="utf-8")

    return HallucinationReportResult(
        report_path=report_path,
        run_id=run_id,
        total_examples=len(rows),
    )


def _render_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    run_id: str,
) -> str:
    total = len(rows)
    lines: list[str] = []

    lines += [f"# Hallucination and Grounding Analysis: {run_id}", ""]

    # Run metadata
    lines += ["## Run Metadata", "", f"| Field | Value |", f"|-------|-------|",
              f"| run_id | {run_id} |",
              f"| total_examples | {total} |", ""]

    # Answer correctness
    correct = sum(1 for r in rows if r.get("answer_verdict") in ("correct", "partially_correct"))
    lines += ["## Answer Correctness", "",
              f"| Metric | Count | Rate |",
              f"|--------|-------|------|",
              f"| Correct answers | {correct} | {correct/total:.1%} |" if total else "| Correct answers | 0 | — |",
              ""]

    # Grounding summary
    lines += ["## Grounding Summary", "",
              "| Grounding Label | Count |",
              "|----------------|-------|"]
    grounding_counts = Counter(r.get("grounding_label") for r in rows)
    for label in GROUNDING_LABELS:
        lines.append(f"| {label} | {grounding_counts.get(label, 0)} |")
    lines.append("")

    # Citation quality summary
    cit_counts = Counter(r.get("citation_quality") for r in rows)
    lines += ["## Citation Quality Summary", "",
              "| Citation Quality | Count |",
              "|-----------------|-------|"]
    for label in ("supports_answer", "partially_supports_answer", "does_not_support_answer",
                  "citation_missing", "citation_invalid"):
        lines.append(f"| {label} | {cit_counts.get(label, 0)} |")
    lines.append("")

    # Context sufficiency summary
    ctx_counts = Counter(r.get("context_sufficiency") for r in rows)
    lines += ["## Context Sufficiency Summary", "",
              "| Sufficiency | Count |",
              "|------------|-------|",
              f"| context_sufficient | {ctx_counts.get('context_sufficient', 0)} |",
              f"| context_partially_sufficient | {ctx_counts.get('context_partially_sufficient', 0)} |",
              f"| context_insufficient | {ctx_counts.get('context_insufficient', 0)} |",
              ""]

    # Failure type counts
    lines += ["## Failure Type Counts", "",
              "| Failure Type | Count |",
              "|-------------|-------|"]
    ft_counter: Counter = Counter()
    for r in rows:
        for ft in r.get("failure_types") or []:
            ft_counter[ft] += 1
    for ft in HALLUCINATION_FAILURE_TYPES:
        lines.append(f"| {ft} | {ft_counter.get(ft, 0)} |")
    lines.append("")

    # Root cause breakdown
    lines += ["## Root Cause Breakdown", "",
              "| Root Cause | Count | Rate |",
              "|-----------|-------|------|"]
    rc_counts = Counter(r.get("root_cause") for r in rows)
    for rc in ROOT_CAUSE_LABELS:
        count = rc_counts.get(rc, 0)
        rate = f"{count/total:.1%}" if total else "—"
        lines.append(f"| {rc} | {count} | {rate} |")
    lines.append("")

    # Example cases
    lines += _example_section("Grounded Correct Answers",
                               rows, lambda r: r.get("root_cause") == "no_failure")
    lines += _example_section("Retrieval Failures",
                               rows, lambda r: r.get("root_cause") == "retrieval_failure")
    lines += _example_section("Hallucinations (Generation Failures)",
                               rows, lambda r: r.get("root_cause") == "generation_failure")
    lines += _example_section("Citation Failures",
                               rows, lambda r: r.get("root_cause") == "citation_failure")

    # Failure slicing (M7.14)
    lines += _slice_section("Grounding Rate by Company", rows, "company", "grounding_label",
                             target_value="grounded")
    lines += _slice_section("Retrieval Failure Rate by Question Type", rows, "question_type",
                             "root_cause", target_value="retrieval_failure")

    # Recommended fixes
    dominant = rc_counts.most_common(1)
    dominant_cause = dominant[0][0] if dominant else "unknown"
    lines += ["## Recommended Next Fixes", ""]
    lines += _recommendations(dominant_cause, rc_counts, total)

    return "\n".join(lines) + "\n"


def _example_section(
    heading: str, rows: list[dict[str, Any]], predicate
) -> list[str]:
    examples = [r for r in rows if predicate(r)][:3]
    if not examples:
        return []
    lines = [f"## Examples: {heading}", ""]
    for ex in examples:
        qid = ex.get("question_id", "?")
        root = ex.get("root_cause", "?")
        verdict = ex.get("answer_verdict", "?")
        lines.append(f"- **{qid}** — verdict: `{verdict}`, root_cause: `{root}`")
    lines.append("")
    return lines


def _slice_section(
    heading: str,
    rows: list[dict[str, Any]],
    group_field: str,
    target_field: str,
    target_value: str,
) -> list[str]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = str(r.get(group_field) or "unknown")
        groups.setdefault(key, []).append(r)
    if not groups:
        return []
    lines = [f"## {heading}", "", f"| {group_field} | {target_value} rate | count |",
             f"|{'-'*len(group_field)}|{'-'*20}|-------|"]
    for key, group_rows in sorted(groups.items()):
        n = len(group_rows)
        hits = sum(1 for r in group_rows if r.get(target_field) == target_value)
        rate = f"{hits/n:.1%}" if n else "—"
        lines.append(f"| {key} | {rate} | {n} |")
    lines.append("")
    return lines


def _recommendations(dominant_cause: str, rc_counts: Counter, total: int) -> list[str]:
    if dominant_cause == "retrieval_failure":
        return [
            "The dominant failure mode is **retrieval failure**.",
            "- Improve chunking strategy or embedding model to increase evidence_hit@k.",
            "- Consider adding more documents or tuning the retrieval top-k.",
            "",
        ]
    if dominant_cause == "generation_failure":
        return [
            "The dominant failure mode is **generation failure** (evidence retrieved but model reasoned incorrectly).",
            "- Improve the RAG prompt to better guide the model toward the evidence.",
            "- Consider a stronger generator model.",
            "",
        ]
    if dominant_cause == "citation_failure":
        return [
            "The dominant failure mode is **citation failure** (answers correct but citations unsupported).",
            "- Update the prompt to require correct citation of chunk_ids.",
            "",
        ]
    return [
        f"Primary failure mode: `{dominant_cause}`.",
        "- Review examples above to identify patterns.",
        "",
    ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


__all__ = [
    "HallucinationReportResult",
    "generate_hallucination_report",
]
