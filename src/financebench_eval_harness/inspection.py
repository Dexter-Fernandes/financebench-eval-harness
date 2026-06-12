from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RESULTS_FILE = "retrieval_results.jsonl"
_METADATA_FILE = "run_metadata.json"


class InspectionError(RuntimeError):
    """Raised when an inspection cannot be loaded."""


@dataclass(frozen=True)
class InspectedEvidence:
    doc_name: str
    page_num: int | None
    text: str


@dataclass(frozen=True)
class InspectedQuestion:
    question_id: str
    question_text: str
    gold_evidence: list[InspectedEvidence]
    query: str
    retrieved: list[dict[str, Any]]


def load_inspection(
    question_id: str,
    run_dir: Path,
    *,
    examples_path: Path | None = None,
) -> InspectedQuestion:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise InspectionError(f"Run directory not found: {run_dir}")

    results_file = run_dir / _RESULTS_FILE
    if not results_file.is_file():
        raise InspectionError(
            f"Retrieval results file not found: {results_file}"
        )

    row = _find_result_row(results_file, question_id)

    resolved_examples = examples_path or _examples_path_from_metadata(run_dir)
    question_text, gold_evidence = _load_gold(question_id, resolved_examples)

    return InspectedQuestion(
        question_id=question_id,
        question_text=question_text,
        gold_evidence=gold_evidence,
        query=row["query"],
        retrieved=row["retrieved"],
    )


def format_inspection(
    inspection: InspectedQuestion,
    *,
    preview_chars: int = 300,
) -> str:
    lines: list[str] = []

    lines.append(f"Question ID : {inspection.question_id}")
    if inspection.question_text:
        lines.append(f"Question    : {inspection.question_text}")
    lines.append(f"Query       : {inspection.query}")

    if inspection.gold_evidence:
        lines.append("")
        lines.append(f"Gold Evidence ({len(inspection.gold_evidence)}):")
        for i, ev in enumerate(inspection.gold_evidence, start=1):
            page = f"p.{ev.page_num}" if ev.page_num is not None else "p.?"
            lines.append(f"  [{i}] {ev.doc_name}  {page}")
            if ev.text:
                lines.append(f"      {ev.text[:preview_chars]}")

    lines.append("")
    lines.append(f"Retrieved Chunks (top {len(inspection.retrieved)}):")
    for chunk in inspection.retrieved:
        page = f"p.{chunk['page_num']}"
        score = f"{chunk['score']:.3f}"
        lines.append(
            f"  #{chunk['rank']}  score={score}  {chunk['doc_name']}  {page}"
        )
        lines.append(f"      {chunk['text'][:preview_chars]}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_result_row(results_file: Path, question_id: str) -> dict[str, Any]:
    for line in results_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("question_id") == question_id:
            return row
    raise InspectionError(f"question_id not found in results: {question_id}")


def _examples_path_from_metadata(run_dir: Path) -> Path | None:
    meta_file = run_dir / _METADATA_FILE
    if not meta_file.is_file():
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    dataset_path = meta.get("dataset_path")
    if not dataset_path:
        return None
    return Path(dataset_path)


def _load_gold(
    question_id: str,
    examples_path: Path | None,
) -> tuple[str, list[InspectedEvidence]]:
    if examples_path is None or not examples_path.is_file():
        return "", []

    for line in examples_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("question_id") != question_id:
            continue
        question_text = row.get("question", "")
        evidence = [
            InspectedEvidence(
                doc_name=ev.get("doc_name", ""),
                page_num=ev.get("matched_page_num"),
                text=ev.get("evidence_text", ""),
            )
            for ev in row.get("evidence", [])
        ]
        return question_text, evidence

    return "", []
