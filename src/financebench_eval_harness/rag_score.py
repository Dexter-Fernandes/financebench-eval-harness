from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financebench_eval_harness.judge import (
    JudgeError,
    parse_grounding_response,
    parse_judge_response,
    render_grounding_prompt,
    render_judge_prompt_for_processed_example,
)
from financebench_eval_harness.llm import LLMClient
from financebench_eval_harness.rag_score_config import RAGScoreConfig
from financebench_eval_harness.rag_types import RAGContextChunk, format_retrieved_context
from financebench_eval_harness.scoring import score_prediction


class RAGScoreError(ValueError):
    """Raised when score_rag_run encounters an unrecoverable error."""


@dataclass(frozen=True)
class RAGScoreResult:
    output_dir: Path
    answer_scores_path: Path
    grounding_scores_path: Path
    combined_scores_path: Path
    example_count: int
    scored_count: int
    error_count: int


def score_rag_run(
    config: RAGScoreConfig,
    *,
    answer_judge_client: LLMClient | None = None,
    grounding_judge_client: LLMClient | None = None,
) -> RAGScoreResult:
    run_dir = config.settings.run_dir
    if not run_dir.is_dir():
        raise RAGScoreError(f"RAG run directory not found: {run_dir}")

    pred_path = run_dir / "rag_predictions.jsonl"
    if not pred_path.is_file():
        raise RAGScoreError(f"rag_predictions.jsonl not found in: {run_dir}")

    predictions = _load_jsonl(pred_path)

    retrieval_by_qid: dict[str, list[dict[str, Any]]] = {}
    if config.settings.retrieval_results_path is not None:
        ret_path = config.settings.retrieval_results_path
        if ret_path.is_file():
            for row in _load_jsonl(ret_path):
                qid = row.get("question_id")
                if isinstance(qid, str):
                    retrieval_by_qid[qid] = [
                        c for c in (row.get("retrieved") or []) if isinstance(c, dict)
                    ]

    output_dir = config.settings.output_dir if config.settings.output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_scores_path = output_dir / "rag_answer_scores.jsonl"
    grounding_scores_path = output_dir / "rag_grounding_scores.jsonl"
    combined_scores_path = output_dir / "rag_combined_scores.jsonl"

    answer_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    scored_count = 0
    error_count = 0

    for pred in predictions:
        qid = str(pred.get("question_id", ""))
        question = pred.get("question", "")
        gold_answer = pred.get("gold_answer", "")
        prediction_text = pred.get("prediction", "")
        pred_status = pred.get("status", "success")

        chunks = _chunks_from_retrieval(retrieval_by_qid.get(qid, []))
        retrieved_context_str = format_retrieved_context(chunks) if chunks else None

        if pred_status != "success" or not isinstance(prediction_text, str) or not prediction_text.strip():
            error_count += 1
            err_row: dict[str, Any] = {"question_id": qid, "status": "error"}
            answer_rows.append(err_row)
            grounding_rows.append(err_row.copy())
            combined_rows.append(err_row.copy())
            continue

        lexical = score_prediction(str(gold_answer), str(prediction_text))

        answer_row = _build_answer_row(
            qid=qid,
            gold_answer=str(gold_answer),
            prediction=str(prediction_text),
            lexical=lexical,
            question=str(question),
            retrieved_context=retrieved_context_str,
            judge_config=config.answer_judge,
            judge_client=answer_judge_client,
            pred=pred,
        )
        answer_rows.append(answer_row)

        grounding_row = _build_grounding_row(
            qid=qid,
            question=str(question),
            prediction=str(prediction_text),
            retrieved_context=retrieved_context_str,
            judge_config=config.grounding_judge,
            judge_client=grounding_judge_client,
        )
        grounding_rows.append(grounding_row)

        combined_row = _build_combined_row(answer_row, grounding_row, k=config.settings.k)
        combined_rows.append(combined_row)
        scored_count += 1

    _write_jsonl(answer_scores_path, answer_rows)
    _write_jsonl(grounding_scores_path, grounding_rows)
    _write_jsonl(combined_scores_path, combined_rows)

    return RAGScoreResult(
        output_dir=output_dir,
        answer_scores_path=answer_scores_path,
        grounding_scores_path=grounding_scores_path,
        combined_scores_path=combined_scores_path,
        example_count=len(predictions),
        scored_count=scored_count,
        error_count=error_count,
    )


# ---------------------------------------------------------------------------
# Per-row builders
# ---------------------------------------------------------------------------


def _build_answer_row(
    *,
    qid: str,
    gold_answer: str,
    prediction: str,
    lexical: dict[str, Any],
    question: str,
    retrieved_context: str | None,
    judge_config: Any,
    judge_client: LLMClient | None,
    pred: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": qid,
        "gold_answer": gold_answer,
        "prediction": prediction,
        "exact_match": lexical.get("exact_match"),
        "normalized_string_match": lexical.get("normalized_string_match"),
        "numeric_match": lexical.get("numeric_match"),
        "unit_match": lexical.get("unit_match"),
        "gold_numeric_values": lexical.get("gold_numeric_values", []),
        "answer_verdict": None,
        "answer_reason": None,
        "answer_numeric_error": None,
        "answer_unsupported_claims": None,
        "answer_judge_status": "skipped",
        "status": "success",
    }

    if judge_config is not None and judge_client is not None:
        processed_example = {
            "question": question,
            "gold_answer": gold_answer,
            "evidence": pred.get("evidence") or [],
        }
        try:
            rendered = render_judge_prompt_for_processed_example(
                judge_config.prompt,
                processed_example,
                prediction=prediction,
                retrieved_context=retrieved_context,
            )
            gen_result = judge_client.generate(rendered.text)
            parsed = parse_judge_response(gen_result.text)
            row["answer_verdict"] = parsed.get("verdict")
            row["answer_reason"] = parsed.get("reason")
            row["answer_numeric_error"] = parsed.get("numeric_error")
            row["answer_unsupported_claims"] = parsed.get("unsupported_claims")
            row["answer_judge_status"] = "success"
        except (JudgeError, Exception) as exc:
            row["answer_judge_status"] = "error"
            row["answer_judge_error"] = str(exc)

    return row


def _build_grounding_row(
    *,
    qid: str,
    question: str,
    prediction: str,
    retrieved_context: str | None,
    judge_config: Any,
    judge_client: LLMClient | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": qid,
        "grounding_verdict": None,
        "citation_correct": None,
        "unsupported_claims": None,
        "grounding_reason": None,
        "grounding_judge_status": "skipped",
        "status": "success",
    }

    if judge_config is None or judge_client is None:
        return row

    if not retrieved_context:
        row["grounding_judge_status"] = "skipped"
        return row

    try:
        rendered = render_grounding_prompt(
            judge_config.prompt,
            question=question,
            prediction=prediction,
            retrieved_context=retrieved_context,
        )
        gen_result = judge_client.generate(rendered.text)
        parsed = parse_grounding_response(gen_result.text)
        row["grounding_verdict"] = parsed.get("verdict")
        row["citation_correct"] = parsed.get("citation_correct")
        row["unsupported_claims"] = parsed.get("unsupported_claims")
        row["grounding_reason"] = parsed.get("reason")
        row["grounding_judge_status"] = "success"
    except (JudgeError, Exception) as exc:
        row["grounding_judge_status"] = "error"
        row["grounding_judge_error"] = str(exc)

    return row


def _build_combined_row(
    answer_row: dict[str, Any],
    grounding_row: dict[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": answer_row.get("question_id"),
        "gold_answer": answer_row.get("gold_answer"),
        "prediction": answer_row.get("prediction"),
        "exact_match": answer_row.get("exact_match"),
        "normalized_string_match": answer_row.get("normalized_string_match"),
        "numeric_match": answer_row.get("numeric_match"),
        "unit_match": answer_row.get("unit_match"),
        "gold_numeric_values": answer_row.get("gold_numeric_values", []),
        "answer_verdict": answer_row.get("answer_verdict"),
        "answer_numeric_error": answer_row.get("answer_numeric_error"),
        "answer_unsupported_claims": answer_row.get("answer_unsupported_claims"),
        "grounding_verdict": grounding_row.get("grounding_verdict"),
        "citation_correct": grounding_row.get("citation_correct"),
        "status": "success",
    }

    row["failure_labels"] = _assign_failure_labels(row)
    row["category"] = _categorize(row)
    return row


def _assign_failure_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []

    answer_verdict = row.get("answer_verdict")
    gold_values = row.get("gold_numeric_values") or []
    numeric_match = row.get("numeric_match")
    answer_numeric_error = row.get("answer_numeric_error")

    # Suppress numeric_error on refusals — over_refusal is the primary label
    if answer_verdict != "not_answered" and (
        answer_numeric_error is True or (gold_values and numeric_match is False)
    ):
        labels.append("numeric_error")

    if row.get("answer_unsupported_claims") is True:
        labels.append("unsupported_claim")

    if row.get("grounding_verdict") == "ungrounded":
        labels.append("ungrounded")

    if answer_verdict == "not_answered":
        labels.append("over_refusal")

    if (
        answer_verdict == "incorrect"
        and "numeric_error" not in labels
        and "unsupported_claim" not in labels
        and "over_refusal" not in labels
    ):
        labels.append("reasoning_error")

    return labels


def _categorize(row: dict[str, Any]) -> str:
    answer_verdict = row.get("answer_verdict")
    numeric_match = row.get("numeric_match")
    unit_match = row.get("unit_match")
    grounding_verdict = row.get("grounding_verdict")

    if answer_verdict is not None:
        answer_correct = answer_verdict in ("correct", "partially_correct")
    else:
        answer_correct = bool(numeric_match) or bool(unit_match)

    grounded = grounding_verdict in ("grounded", "partially_grounded") if grounding_verdict else None

    if grounded is not None:
        if answer_correct:
            return "answer_correct_grounded" if grounded else "answer_correct_ungrounded"
        return "answer_wrong_grounded" if grounded else "answer_wrong_ungrounded"

    return "answer_correct" if answer_correct else "answer_wrong"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks_from_retrieval(raw_chunks: list[dict[str, Any]]) -> list[RAGContextChunk]:
    result = []
    for c in raw_chunks:
        try:
            result.append(RAGContextChunk(
                rank=int(c.get("rank", 0)),
                chunk_id=str(c.get("chunk_id", "")),
                doc_name=str(c.get("doc_name", "")),
                page_num=int(c.get("page_num", 0)),
                text=str(c.get("text", "")),
                score=float(c["score"]) if "score" in c else None,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RAGScoreError",
    "RAGScoreResult",
    "score_rag_run",
]
