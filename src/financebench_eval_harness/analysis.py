from __future__ import annotations

from typing import Any

_CATEGORIES = (
    "retrieval_hit_answer_correct",
    "retrieval_hit_answer_wrong",
    "retrieval_miss_answer_correct",
    "retrieval_miss_answer_wrong",
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
        if ans_row is not None:
            raw_scores = ans_row.get("scores")
            if isinstance(raw_scores, dict):
                answer_fields = {
                    "exact_match": raw_scores.get("exact_match"),
                    "normalized_string_match": raw_scores.get("normalized_string_match"),
                    "numeric_match": raw_scores.get("numeric_match"),
                    "unit_match": raw_scores.get("unit_match"),
                }
            judge = ans_row.get("judge")
            if isinstance(judge, dict):
                verdict = judge.get("verdict")
                if isinstance(verdict, str):
                    judge_verdict = verdict

        joined: dict[str, object] = {
            "question_id": question_id,
            "retrieval": retrieval_fields,
            "answer": answer_fields,
            "judge_verdict": judge_verdict,
        }
        joined["category"] = categorize_join_row(joined, k=k)
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
    return summary


__all__ = [
    "categorize_join_row",
    "join_retrieval_and_answer_scores",
    "summarize_joined_metrics",
]
