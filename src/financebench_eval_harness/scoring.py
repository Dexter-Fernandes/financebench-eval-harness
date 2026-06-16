from __future__ import annotations

import re


_NUMERIC_PATTERN = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")
_FISCAL_YEAR_PATTERN = re.compile(
    r"\bFY\s?(\d{4})\b|\bfiscal\s+year\s+(\d{4})\b",
    re.IGNORECASE,
)
_QUARTER_PATTERN = re.compile(r"\bQ([1-4])\s+(\d{4})\b", re.IGNORECASE)
_SCORE_KEYS = (
    "exact_match",
    "normalized_string_match",
    "contains_gold_answer",
    "numeric_match",
    "unit_match",
)

_UNIT_SCALE: dict[str, float] = {
    "trillion": 1e12,
    "trillions": 1e12,
    "billion": 1e9,
    "billions": 1e9,
    "million": 1e6,
    "millions": 1e6,
    "thousand": 1e3,
    "thousands": 1e3,
    "percent": 1e-2,
    "%": 1e-2,
    "": 1.0,
}
_UNIT_WORD_ALT = "|".join(
    k for k in sorted(_UNIT_SCALE, key=len, reverse=True) if k not in ("", "%")
)
# Captures: group 1 = numeric token, group 2 = unit suffix ("%" or word like "million")
_UNIT_VALUE_PATTERN = re.compile(
    r"(\(?-?\$?\d[\d,]*(?:\.\d+)?\)?)"
    r"(%|(?:\s+(?:" + _UNIT_WORD_ALT + r")))?",
    re.IGNORECASE,
)


def extract_fiscal_periods(text: str) -> list[str]:
    """Extract fiscal year and quarter strings, normalized, deduped, ordered."""
    seen: set[str] = set()
    results: list[str] = []
    for match in _FISCAL_YEAR_PATTERN.finditer(text):
        year = match.group(1) or match.group(2)
        period = f"FY{year}"
        if period not in seen:
            seen.add(period)
            results.append(period)
    for match in _QUARTER_PATTERN.finditer(text):
        q, year = match.group(1), match.group(2)
        period = f"Q{q}_{year}"
        if period not in seen:
            seen.add(period)
            results.append(period)
    return results


def extract_numeric_values(text: str) -> list[float]:
    values: list[float] = []
    for match in _NUMERIC_PATTERN.finditer(text):
        raw_value = match.group(0)
        normalized_value = _normalize_numeric_token(raw_value)
        if normalized_value is None:
            continue
        values.append(normalized_value)
    return values


def extract_numeric_with_unit(text: str) -> list[tuple[float, str]]:
    """Extract (value, unit) pairs where unit is a scale keyword or '' for dimensionless."""
    results: list[tuple[float, str]] = []
    for match in _UNIT_VALUE_PATTERN.finditer(text):
        raw_num = match.group(1)
        unit_suffix = match.group(2)  # "%" or " million" etc., or None
        value = _normalize_numeric_token(raw_num)
        if value is None:
            continue
        if unit_suffix is None:
            unit = ""
        else:
            unit = unit_suffix.strip().lower()
        results.append((value, unit))
    return results


def score_prediction(gold_answer: str, prediction: str) -> dict[str, object]:
    normalized_gold = _normalize_text(gold_answer)
    normalized_prediction = _normalize_text(prediction)
    gold_numeric_values = extract_numeric_values(gold_answer)
    prediction_numeric_values = extract_numeric_values(prediction)
    gold_unit_pairs = extract_numeric_with_unit(gold_answer)
    prediction_unit_pairs = extract_numeric_with_unit(prediction)
    return {
        "exact_match": prediction == gold_answer,
        "normalized_string_match": normalized_prediction == normalized_gold,
        "contains_gold_answer": bool(normalized_gold)
        and normalized_gold in normalized_prediction,
        "numeric_match": _numeric_values_match(
            gold_numeric_values,
            prediction_numeric_values,
        ),
        "unit_match": _unit_values_match(gold_unit_pairs, prediction_unit_pairs),
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


def _scaled_value(value: float, unit: str) -> float:
    return value * _UNIT_SCALE.get(unit.lower(), 1.0)


def _unit_values_match(
    gold_pairs: list[tuple[float, str]],
    prediction_pairs: list[tuple[float, str]],
) -> bool:
    if not gold_pairs:
        return False
    return all(
        any(
            _values_close(_scaled_value(gv, gu), _scaled_value(pv, pu))
            for pv, pu in prediction_pairs
        )
        for gv, gu in gold_pairs
    )


def _values_close(a: float, b: float) -> bool:
    return abs(a - b) / max(abs(a), 1.0) <= 0.01


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
    "extract_fiscal_periods",
    "extract_numeric_values",
    "extract_numeric_with_unit",
    "score_prediction",
    "summarize_scores",
]
