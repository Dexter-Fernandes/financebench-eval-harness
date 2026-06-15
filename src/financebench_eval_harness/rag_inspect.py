from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RagInspectError(ValueError):
    """Raised when inspect-rag input data is missing or invalid."""


@dataclass(frozen=True)
class RagInspectionResult:
    question_id: str
    question: str | None
    gold_answer: str | None
    prediction: str | None
    judge_verdict: str | None
    judge_numeric_error: bool | None
    judge_unsupported_claims: bool | None
    numeric_match: bool | None
    exact_match: bool | None
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[dict[str, Any]]
    gold_doc_name: str | None
    gold_page_num: int | None
    page_hit_at_k: bool | None
    best_evidence_overlap: float | None
    failure_labels: list[str]
    category: str | None


def load_rag_inspection(
    rag_run_dir: Path,
    question_id: str,
    *,
    retrieval_run_dir: Path | None = None,
    joined_dir: Path | None = None,
) -> RagInspectionResult:
    rag_run_dir = Path(rag_run_dir)

    if not rag_run_dir.is_dir():
        raise RagInspectError(f"RAG run directory not found: {rag_run_dir}")

    pred_path = rag_run_dir / "rag_predictions.jsonl"
    if not pred_path.is_file():
        raise RagInspectError(f"rag_predictions.jsonl not found in: {rag_run_dir}")

    score_path = rag_run_dir / "scores.jsonl"
    if not score_path.is_file():
        raise RagInspectError(f"scores.jsonl not found in: {rag_run_dir}")

    pred_row = _find_row(pred_path, question_id)
    if pred_row is None:
        raise RagInspectError(f"Question ID not found: {question_id!r}")

    score_row = _find_row(score_path, question_id)

    question = pred_row.get("question") if pred_row else None
    gold_answer = pred_row.get("gold_answer") if pred_row else None
    prediction = pred_row.get("prediction") if pred_row else None
    retrieved_chunk_ids: list[str] = []
    if pred_row:
        raw_ids = pred_row.get("retrieved_chunk_ids") or []
        retrieved_chunk_ids = [str(x) for x in raw_ids if isinstance(x, str)]

    numeric_match: bool | None = None
    exact_match: bool | None = None
    judge_verdict: str | None = None
    judge_numeric_error: bool | None = None
    judge_unsupported_claims: bool | None = None
    if score_row:
        scores = score_row.get("scores") or {}
        if isinstance(scores, dict):
            nm = scores.get("numeric_match")
            if isinstance(nm, bool):
                numeric_match = nm
            em = scores.get("exact_match")
            if isinstance(em, bool):
                exact_match = em
        judge = score_row.get("judge")
        if isinstance(judge, dict):
            v = judge.get("verdict")
            if isinstance(v, str):
                judge_verdict = v
            ne = judge.get("numeric_error")
            if isinstance(ne, bool):
                judge_numeric_error = ne
            uc = judge.get("unsupported_claims")
            if isinstance(uc, bool):
                judge_unsupported_claims = uc

    # Auto-resolve retrieval run dir from rag_run_metadata.json
    if retrieval_run_dir is None:
        meta_path = rag_run_dir / "rag_run_metadata.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ret_run_id = meta.get("retrieval_run_id")
            if ret_run_id:
                candidate = rag_run_dir.parent / str(ret_run_id)
                if (candidate / "retrieval_results.jsonl").is_file():
                    retrieval_run_dir = candidate

    retrieved_chunks: list[dict[str, Any]] = []
    gold_doc_name: str | None = None
    gold_page_num: int | None = None
    page_hit_at_k: bool | None = None
    ret_score_row: dict[str, Any] | None = None

    if retrieval_run_dir is not None:
        retrieval_run_dir = Path(retrieval_run_dir)
        ret_result_row = _find_row(retrieval_run_dir / "retrieval_results.jsonl", question_id)
        ret_score_row = _find_row(retrieval_run_dir / "retrieval_scores.jsonl", question_id)
        if ret_result_row:
            raw_chunks = ret_result_row.get("retrieved") or []
            retrieved_chunks = [c for c in raw_chunks if isinstance(c, dict)]
        if ret_score_row:
            gold_doc_name = ret_score_row.get("gold_doc_name")
            gp = ret_score_row.get("gold_page_num")
            if isinstance(gp, int):
                gold_page_num = gp
            # Find any hit@k key
            for key, val in ret_score_row.items():
                if key.startswith("page_hit@") and isinstance(val, bool):
                    page_hit_at_k = val
                    break

    best_evidence_overlap: float | None = None
    failure_labels: list[str] = []
    category: str | None = None

    if joined_dir is not None:
        joined_row = _find_row(Path(joined_dir) / "joined_metrics.jsonl", question_id)
        if joined_row:
            fl = joined_row.get("failure_labels")
            if isinstance(fl, list):
                failure_labels = [str(x) for x in fl]
            cat = joined_row.get("category")
            if isinstance(cat, str):
                category = cat
            ret_section = joined_row.get("retrieval")
            if isinstance(ret_section, dict):
                ov = ret_section.get("best_evidence_overlap")
                if isinstance(ov, float):
                    best_evidence_overlap = ov
                if page_hit_at_k is None:
                    for key, val in ret_section.items():
                        if key.startswith("page_hit@") and isinstance(val, bool):
                            page_hit_at_k = val
                            break

    # Fallback: get best_evidence_overlap from retrieval_scores.jsonl
    if best_evidence_overlap is None and ret_score_row:
        ov = ret_score_row.get("best_evidence_overlap")
        if isinstance(ov, float):
            best_evidence_overlap = ov

    return RagInspectionResult(
        question_id=question_id,
        question=question,
        gold_answer=gold_answer,
        prediction=prediction,
        judge_verdict=judge_verdict,
        judge_numeric_error=judge_numeric_error,
        judge_unsupported_claims=judge_unsupported_claims,
        numeric_match=numeric_match,
        exact_match=exact_match,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_chunks=retrieved_chunks,
        gold_doc_name=gold_doc_name,
        gold_page_num=gold_page_num,
        page_hit_at_k=page_hit_at_k,
        best_evidence_overlap=best_evidence_overlap,
        failure_labels=failure_labels,
        category=category,
    )


def format_rag_inspection(result: RagInspectionResult, *, preview_chars: int = 400) -> str:
    lines: list[str] = []
    lines.append(f"=== RAG Inspection: {result.question_id} ===")
    lines.append("")

    lines.append("QUESTION")
    lines.append(f"  {result.question or '—'}")
    lines.append("")

    lines.append("GOLD ANSWER")
    lines.append(f"  {result.gold_answer or '—'}")
    lines.append("")

    lines.append("GENERATED ANSWER")
    lines.append(f"  {result.prediction or '—'}")
    lines.append("")

    lines.append("ANSWER SCORES")
    lines.append(f"  judge_verdict:       {result.judge_verdict or '—'}")
    lines.append(f"  numeric_match:       {result.numeric_match}")
    lines.append(f"  exact_match:         {result.exact_match}")
    lines.append(f"  judge_numeric_error: {result.judge_numeric_error}")
    lines.append(f"  judge_unsupported:   {result.judge_unsupported_claims}")
    lines.append("")

    labels_str = ", ".join(result.failure_labels) if result.failure_labels else "(none)"
    category_str = f"  [category: {result.category}]" if result.category else ""
    lines.append("FAILURE LABELS")
    lines.append(f"  {labels_str}{category_str}")
    lines.append("")

    if result.gold_doc_name or result.gold_page_num is not None:
        lines.append("GOLD EVIDENCE")
        doc_str = result.gold_doc_name or "—"
        page_str = str(result.gold_page_num) if result.gold_page_num is not None else "—"
        lines.append(f"  doc: {doc_str}  page: {page_str}")
        if result.best_evidence_overlap is not None:
            hit_str = str(result.page_hit_at_k) if result.page_hit_at_k is not None else "—"
            lines.append(
                f"  best_evidence_overlap: {result.best_evidence_overlap:.2f}"
                f"  page_hit@k: {hit_str}"
            )
        lines.append("")

    if result.retrieved_chunks:
        lines.append(f"RETRIEVED CHUNKS (top {len(result.retrieved_chunks)})")
        gold_page = result.gold_page_num
        gold_doc = result.gold_doc_name
        for chunk in result.retrieved_chunks:
            rank = chunk.get("rank", "?")
            doc = chunk.get("doc_name", "—")
            page = chunk.get("page_num", "—")
            score = chunk.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            is_hit = (
                doc == gold_doc and page == gold_page
                if gold_doc is not None and gold_page is not None
                else False
            )
            hit_marker = "  [PAGE HIT]" if is_hit else ""
            lines.append(f"  [{rank}] {doc} p.{page}  score={score_str}{hit_marker}")
            text = str(chunk.get("text", ""))
            if text:
                preview = text[:preview_chars].replace("\n", " ")
                lines.append(f'      "{preview}"')
        lines.append("")

    if result.retrieved_chunk_ids:
        lines.append("CHUNK IDS FED TO MODEL")
        lines.append(f"  {', '.join(result.retrieved_chunk_ids)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _find_row(path: Path, question_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("question_id") == question_id:
            return row
    return None


__all__ = [
    "RagInspectError",
    "RagInspectionResult",
    "load_rag_inspection",
    "format_rag_inspection",
]
