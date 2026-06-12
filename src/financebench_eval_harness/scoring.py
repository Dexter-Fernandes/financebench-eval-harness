from __future__ import annotations

import re


_NUMERIC_PATTERN = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")
_SCORE_KEYS = (
    "exact_match",
    "normalized_string_match",
    "contains_gold_answer",
    "numeric_match",
)


def extract_numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMERIC_PATTERN.finditer(text):
        raw_value = match.group(0)
        normalized_value = _normalize_numeric_token(raw_value)
        if normalized_value is None:
            continue
        values.append(normalized_value)
    return values


def score_prediction(gold_answer: str, prediction: str) -> dict[str, object]:
    normalized_gold = _normalize_text(gold_answer)
    normalized_prediction = _normalize_text(prediction)
    gold_numeric_values = extract_numeric_values(gold_answer)
    prediction_numeric_values = extract_numeric_values(prediction)
    return {
        "exact_match": prediction == gold_answer,
        "normalized_string_match": normalized_prediction == normalized_gold,
        "contains_gold_answer": bool(normalized_gold)
        and normalized_gold in normalized_prediction,
        "numeric_match": _numeric_values_match(
            gold_numeric_values,
            prediction_numeric_values,
        ),
        "gold_numeric_values": gold_numeric_values,
        "prediction_numeric_values": prediction_numeric_values,
    }


def summarize_scores(scores: list[dict[str, object]]) -> dict[str, object]:
    example_count = len(scores)
    summary: dict[str, object] = {"example_count": example_count}
    for score_key in _SCORE_KEYS:
        count = sum(1 for score in scores if score.get(score_key) is True)
        summary[f"{score_key}_count"] = count
        summary[f"{score_key}_rate"] = count / example_count if example_count else 0.0
    return summary


def _normalize_text(text: str) -> str:
    lowered_text = text.lower()
    alphanumeric_text = re.sub(r"[^a-z0-9]+", " ", lowered_text)
    return " ".join(alphanumeric_text.split())


def _numeric_values_match(
    gold_numeric_values: list[float],
    prediction_numeric_values: list[float],
) -> bool:
    if not gold_numeric_values:
        return False
    return all(
        any(
            abs(gold_value - prediction_value) <= 1e-9
            for prediction_value in prediction_numeric_values
        )
        for gold_value in gold_numeric_values
    )


def _normalize_numeric_token(raw_value: str) -> float | None:
    is_parenthesized_negative = raw_value.startswith("(") and ")" in raw_value
    has_minus_sign = "-" in raw_value
    is_negative = is_parenthesized_negative or has_minus_sign

    normalized_value = (
        raw_value.replace("(", "")
        .replace(")", "")
        .replace("%", "")
        .replace("$", "")
        .replace(",", "")
        .lstrip("-")
    )
    if not normalized_value:
        return None
    if is_negative:
        normalized_value = f"-{normalized_value}"
    try:
        return float(normalized_value)
    except ValueError:
        return None


__all__ = [
    "extract_numeric_values",
    "score_prediction",
    "summarize_scores",
]
