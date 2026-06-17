"""M7.10 / M7.11 / M7.14: Grounding analysis orchestrator.

Reads rag_predictions.jsonl + rag_grounding_scores.jsonl + retrieval_scores.jsonl
from a completed run directory and produces:
  grounding_scores.jsonl, citation_scores.jsonl, failure_analysis.jsonl,
  failure_summary.json, grounding_analysis_config.yaml
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financebench_eval_harness.analysis import classify_root_cause
from financebench_eval_harness.citation_checker import score_citations
from financebench_eval_harness.context_sufficiency import check_context_sufficiency
from financebench_eval_harness.grounding_analysis_config import GroundingAnalysisConfig
from financebench_eval_harness.grounding_types import (
    ROOT_CAUSE_LABELS,
    GroundingAnalysisError,
)
from financebench_eval_harness.hallucination_checks import run_rule_based_checks
from financebench_eval_harness.scoring import extract_numeric_values


@dataclass(frozen=True)
class GroundingAnalysisResult:
    grounding_scores_path: Path
    citation_scores_path: Path
    failure_analysis_path: Path
    failure_summary_path: Path
    config_snapshot_path: Path
    example_count: int
    grounding_scored_count: int
    error_count: int


def analyze_grounding(
    config: GroundingAnalysisConfig,
    *,
    judge_client: Any = None,
) -> GroundingAnalysisResult:
    """Main entry point for M7 grounding analysis."""
    run_dir = config.settings.run_dir
    output_dir = config.settings.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    k = config.settings.k

    predictions = _load_jsonl(run_dir / "rag_predictions.jsonl")
    grounding_scores_raw = _load_jsonl_by_qid(run_dir / "rag_grounding_scores.jsonl")
    answer_scores_raw = _load_jsonl_by_qid(run_dir / "rag_answer_scores.jsonl")

    retrieval_path = (
        config.settings.retrieval_results_path or run_dir / "retrieval_scores.jsonl"
    )
    retrieval_scores_raw = _load_jsonl_by_qid(retrieval_path) if retrieval_path.is_file() else {}

    grounding_rows: list[dict[str, Any]] = []
    citation_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    error_count = 0

    for pred in predictions:
        qid = str(pred.get("question_id", ""))
        try:
            g_row, c_row, f_row = _process_prediction(
                pred=pred,
                grounding_score_raw=grounding_scores_raw.get(qid),
                answer_score_raw=answer_scores_raw.get(qid),
                retrieval_score_raw=retrieval_scores_raw.get(qid),
                k=k,
            )
            grounding_rows.append(g_row)
            citation_rows.append(c_row)
            failure_rows.append(f_row)
        except Exception:
            error_count += 1
            failure_rows.append({"question_id": qid, "root_cause": "no_failure", "error": True})

    grounding_path = output_dir / "grounding_scores.jsonl"
    citation_path = output_dir / "citation_scores.jsonl"
    failure_path = output_dir / "failure_analysis.jsonl"
    summary_path = output_dir / "failure_summary.json"
    config_path = output_dir / "grounding_analysis_config.yaml"

    _write_jsonl(grounding_path, grounding_rows)
    _write_jsonl(citation_path, citation_rows)
    _write_jsonl(failure_path, failure_rows)

    summary = _summarize(failure_rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import yaml
    config_path.write_text(
        yaml.dump({"k": k, "run_dir": str(run_dir)}), encoding="utf-8"
    )

    return GroundingAnalysisResult(
        grounding_scores_path=grounding_path,
        citation_scores_path=citation_path,
        failure_analysis_path=failure_path,
        failure_summary_path=summary_path,
        config_snapshot_path=config_path,
        example_count=len(predictions),
        grounding_scored_count=len(grounding_rows),
        error_count=error_count,
    )


def _process_prediction(
    *,
    pred: dict[str, Any],
    grounding_score_raw: dict[str, Any] | None,
    answer_score_raw: dict[str, Any] | None,
    retrieval_score_raw: dict[str, Any] | None,
    k: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    qid = str(pred.get("question_id", ""))
    prediction_text = str(pred.get("prediction", ""))
    gold_answer = str(pred.get("gold_answer", ""))
    retrieved_chunks = pred.get("retrieved_chunks") or []
    if not isinstance(retrieved_chunks, list):
        retrieved_chunks = []

    evidence_hit = _get_evidence_hit(retrieval_score_raw, k)
    context_sufficiency = check_context_sufficiency(
        gold_answer=gold_answer,
        retrieved_chunks=retrieved_chunks,
        evidence_hit=evidence_hit,
    )

    citation_result = score_citations(
        answer_text=prediction_text, retrieved_chunks=retrieved_chunks
    )
    cited_ids = citation_result.get("cited_chunk_ids") or []
    all_chunk_ids = [str(c.get("chunk_id", "")) for c in retrieved_chunks]
    gold_nums = extract_numeric_values(gold_answer)
    pred_nums = extract_numeric_values(prediction_text)
    rule_flags = run_rule_based_checks(
        prediction=prediction_text,
        gold_answer=gold_answer,
        retrieved_chunks=retrieved_chunks,
        cited_chunk_ids=cited_ids,
        all_chunk_ids=all_chunk_ids,
        gold_numeric_values=gold_nums,
        prediction_numeric_values=pred_nums,
    )

    existing_verdict = None
    if grounding_score_raw:
        existing_verdict = grounding_score_raw.get("verdict") or grounding_score_raw.get(
            "grounding_label"
        )

    grounding_label = _resolve_grounding_label(existing_verdict, context_sufficiency)

    g_row: dict[str, Any] = {
        "question_id": qid,
        "grounding_label": grounding_label,
        "context_sufficiency": context_sufficiency,
        "rule_flags": rule_flags,
        "judge_grounding_label": None,
        "judge_failure_types": [],
        "judge_citation_quality": None,
        "judge_reason": None,
        "cited_chunk_ids": cited_ids,
    }

    c_row: dict[str, Any] = {"question_id": qid, **citation_result}

    answer_verdict = None
    if answer_score_raw:
        answer_verdict = answer_score_raw.get("answer_verdict")

    f_row = join_all_signals(
        qid=qid,
        grounding_row=g_row,
        citation_row=c_row,
        answer_score_row=answer_score_raw,
        retrieval_score_row=retrieval_score_raw,
        k=k,
    )
    return g_row, c_row, f_row


def join_all_signals(
    *,
    qid: str,
    grounding_row: dict[str, Any],
    citation_row: dict[str, Any],
    answer_score_row: dict[str, Any] | None,
    retrieval_score_row: dict[str, Any] | None,
    k: int,
) -> dict[str, Any]:
    """M7.10: produce one failure_analysis row per question with all signals joined."""
    evidence_hit = _get_evidence_hit(retrieval_score_row, k)
    answer_verdict = None
    company = None
    question_type = None
    if answer_score_row:
        answer_verdict = answer_score_row.get("answer_verdict")
        company = answer_score_row.get("company")
        question_type = answer_score_row.get("question_type")

    grounding_label = grounding_row.get("grounding_label")
    context_sufficiency = grounding_row.get("context_sufficiency")
    citation_quality = citation_row.get("citation_quality")
    rule_flags = list(grounding_row.get("rule_flags") or [])
    failure_types = list(grounding_row.get("judge_failure_types") or [])

    row: dict[str, Any] = {
        "question_id": qid,
        f"evidence_hit@{k}": evidence_hit,
        "answer_verdict": answer_verdict,
        "grounding_label": grounding_label,
        "context_sufficiency": context_sufficiency,
        "citation_quality": citation_quality,
        "failure_types": failure_types,
        "rule_flags": rule_flags,
        "company": company,
        "question_type": question_type,
    }
    row["root_cause"] = classify_root_cause(row, k=k)
    return row


def _resolve_grounding_label(
    existing_verdict: str | None, context_sufficiency: str
) -> str:
    if existing_verdict in (
        "grounded", "partially_grounded", "ungrounded",
        "contradicted", "insufficient_evidence", "over_refusal", "under_refusal",
    ):
        return existing_verdict
    if context_sufficiency == "context_insufficient":
        return "insufficient_evidence"
    if existing_verdict == "grounded":
        return "grounded"
    return "ungrounded"


def _get_evidence_hit(retrieval_row: dict[str, Any] | None, k: int) -> bool | None:
    if retrieval_row is None:
        return None
    val = retrieval_row.get(f"evidence_text_hit@{k}") or retrieval_row.get(f"page_hit@{k}")
    if val is None:
        return None
    return bool(val)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    summary: dict[str, Any] = {"total": total}
    for label in ROOT_CAUSE_LABELS:
        summary[f"{label}_count"] = sum(1 for r in rows if r.get("root_cause") == label)
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _load_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("question_id", "")): row
        for row in _load_jsonl(path)
        if "question_id" in row
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


__all__ = [
    "GroundingAnalysisResult",
    "analyze_grounding",
    "join_all_signals",
]
