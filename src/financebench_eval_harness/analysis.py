from __future__ import annotations

from typing import Any

_CATEGORIES = (
    "retrieval_hit_answer_correct",
    "retrieval_hit_answer_wrong",
    "retrieval_miss_answer_correct",
    "retrieval_miss_answer_wrong",
)

_FAILURE_LABELS = (
    "retrieval_miss",
    "numeric_error",
    "unsupported_claim",
    "over_refusal",
    "reasoning_error",
)


def join_retrieval_and_answer_scores(
    retrieval_scores: list[dict[str, object]],
    answer_scores: list[dict[str, object]],
    *,
    k: int = 5,
) -> list[dict[str, object]]:
    answers_by_id: dict[str, dict[str, object]] = {
        str(row["question_id"]): row  # type: ignore[index]
        for row in answer_scores
        if isinstance(row.get("question_id"), str)
    }

    rows: list[dict[str, object]] = []
    for ret_row in retrieval_scores:
        question_id = str(ret_row.get("question_id", ""))

        retrieval_fields: dict[str, object] = {
            f"doc_hit@{k}": ret_row.get(f"doc_hit@{k}"),
            f"page_hit@{k}": ret_row.get(f"page_hit@{k}"),
            f"evidence_text_hit@{k}": ret_row.get(f"evidence_text_hit@{k}"),
            "best_evidence_overlap": ret_row.get("best_evidence_overlap"),
            "page_first_hit_rank": ret_row.get("page_first_hit_rank"),
        }

        ans_row = answers_by_id.get(question_id)
        answer_fields: dict[str, object] | None = None
        judge_verdict: str | None = None
        judge_numeric_error: bool | None = None
        judge_unsupported_claims: bool | None = None

        if ans_row is not None:
            raw_scores = ans_row.get("scores")
            if isinstance(raw_scores, dict):
                answer_fields = {
                    "exact_match": raw_scores.get("exact_match"),
                    "normalized_string_match": raw_scores.get("normalized_string_match"),
                    "numeric_match": raw_scores.get("numeric_match"),
                    "unit_match": raw_scores.get("unit_match"),
                    "gold_numeric_values": raw_scores.get("gold_numeric_values"),
                }
            judge = ans_row.get("judge")
            if isinstance(judge, dict):
                verdict = judge.get("verdict")
                if isinstance(verdict, str):
                    judge_verdict = verdict
                ne = judge.get("numeric_error")
                if isinstance(ne, bool):
                    judge_numeric_error = ne
                uc = judge.get("unsupported_claims")
                if isinstance(uc, bool):
                    judge_unsupported_claims = uc

        joined: dict[str, object] = {
            "question_id": question_id,
            "retrieval": retrieval_fields,
            "answer": answer_fields,
            "judge_verdict": judge_verdict,
            "judge_numeric_error": judge_numeric_error,
            "judge_unsupported_claims": judge_unsupported_claims,
        }
        joined["category"] = categorize_join_row(joined, k=k)
        joined["failure_labels"] = assign_failure_labels(joined, k=k)
        rows.append(joined)

    return rows


def categorize_join_row(row: dict[str, object], *, k: int = 5) -> str:
    retrieval: Any = row.get("retrieval") or {}
    answer: Any = row.get("answer") or {}
    retrieval_hit = bool(retrieval.get(f"page_hit@{k}"))
    answer_correct = bool(answer.get("numeric_match")) or bool(answer.get("unit_match"))

    if retrieval_hit and answer_correct:
        return "retrieval_hit_answer_correct"
    if retrieval_hit:
        return "retrieval_hit_answer_wrong"
    if answer_correct:
        return "retrieval_miss_answer_correct"
    return "retrieval_miss_answer_wrong"


def assign_failure_labels(row: dict[str, object], *, k: int = 5) -> list[str]:
    labels: list[str] = []
    retrieval: Any = row.get("retrieval") or {}
    answer: Any = row.get("answer") or {}
    page_hit = bool(retrieval.get(f"page_hit@{k}"))
    judge_verdict = row.get("judge_verdict")
    numeric_error_flag = row.get("judge_numeric_error")
    unsupported_flag = row.get("judge_unsupported_claims")

    if not page_hit:
        labels.append("retrieval_miss")

    gold_numeric = answer.get("gold_numeric_values")
    gold_has_numbers = isinstance(gold_numeric, list) and len(gold_numeric) > 0
    numeric_error_heuristic = not bool(answer.get("numeric_match")) and gold_has_numbers
    if numeric_error_flag is True or numeric_error_heuristic:
        labels.append("numeric_error")

    if unsupported_flag is True:
        labels.append("unsupported_claim")

    if judge_verdict == "not_answered" and page_hit:
        labels.append("over_refusal")

    answer_wrong = not (bool(answer.get("numeric_match")) or bool(answer.get("unit_match")))
    verdict_wrong = judge_verdict in ("incorrect", "partially_correct")
    if (
        page_hit
        and answer_wrong
        and verdict_wrong
        and "numeric_error" not in labels
        and "unsupported_claim" not in labels
    ):
        labels.append("reasoning_error")

    return labels


def classify_root_cause(row: dict[str, object], *, k: int = 5) -> str:
    """M7.11: deterministic 6-case root cause hierarchy.

    Uses fields produced by join_all_signals() / failure_analysis rows.
    """
    evidence_hit = row.get(f"evidence_hit@{k}")
    answer_verdict = row.get("answer_verdict")
    citation_quality = row.get("citation_quality")
    context_sufficiency = row.get("context_sufficiency")
    grounding_label = row.get("grounding_label")

    answer_correct = answer_verdict in ("correct", "partially_correct")
    answer_wrong = not answer_correct

    # 5. Context insufficient but model answered anyway → hallucination_under_refusal
    # (checked before rule 1 because under_refusal label is the definitive signal)
    if grounding_label == "under_refusal":
        return "hallucination_under_refusal"

    # 6. Context sufficient but model refused → over_refusal
    if grounding_label == "over_refusal":
        return "over_refusal"
    if answer_verdict == "not_answered" and context_sufficiency == "context_sufficient":
        return "over_refusal"

    # 1. Retrieval miss → retrieval_failure
    if evidence_hit is False:
        return "retrieval_failure"

    # 3. Answer correct but citation bad → citation_failure
    if answer_correct and citation_quality in ("citation_invalid", "does_not_support_answer"):
        return "citation_failure"

    # 2. Evidence present but answer wrong → generation_failure
    if evidence_hit is True and answer_wrong:
        return "generation_failure"

    # 4. No failure
    return "no_failure"


def summarize_joined_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    example_count = len(rows)
    summary: dict[str, object] = {"example_count": example_count}
    for category in _CATEGORIES:
        summary[category] = sum(1 for r in rows if r.get("category") == category)
    summary["retrieval_hit_count"] = sum(
        1
        for r in rows
        if r.get("category") in ("retrieval_hit_answer_correct", "retrieval_hit_answer_wrong")
    )
    summary["answer_correct_count"] = sum(
        1
        for r in rows
        if r.get("category") in ("retrieval_hit_answer_correct", "retrieval_miss_answer_correct")
    )
    for label in _FAILURE_LABELS:
        summary[f"{label}_count"] = sum(
            1
            for r in rows
            if label in (r.get("failure_labels") or [])
        )
    summary["no_failure_label_count"] = sum(
        1 for r in rows if not (r.get("failure_labels") or [])
    )
    return summary


__all__ = [
    "assign_failure_labels",
    "categorize_join_row",
    "classify_root_cause",
    "join_retrieval_and_answer_scores",
    "summarize_joined_metrics",
]
