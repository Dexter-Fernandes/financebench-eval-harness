from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from financebench_eval_harness.run_config import JudgePromptConfig

JUDGE_VERDICTS = ("correct", "partially_correct", "incorrect", "not_answered")
GROUNDING_VERDICTS = ("grounded", "partially_grounded", "ungrounded")


@dataclass(frozen=True)
class RenderedJudgePrompt:
    text: str
    prompt_id: str
    prompt_version: str
    template_path: Path


class JudgeError(ValueError):
    """Raised when judge prompt rendering or response parsing fails."""


def render_judge_prompt_for_processed_example(
    prompt_config: JudgePromptConfig,
    processed_example: dict[str, object],
    *,
    prediction: str,
    retrieved_context: str | None = None,
) -> RenderedJudgePrompt:
    template = _read_prompt_template(prompt_config.template_path)
    question = _required_text(processed_example.get("question"), "question")
    gold_answer = _required_text(processed_example.get("gold_answer"), "gold_answer")
    clean_prediction = _required_text(prediction, "prediction")
    evidence_text = _joined_evidence_context(processed_example.get("evidence"))
    return RenderedJudgePrompt(
        text=template.format(
            question=question,
            gold_answer=gold_answer,
            prediction=clean_prediction,
            evidence_text=evidence_text,
            retrieved_context=retrieved_context or "(no retrieved context)",
        ),
        prompt_id=prompt_config.id,
        prompt_version=prompt_config.version,
        template_path=prompt_config.template_path,
    )


def render_grounding_prompt(
    prompt_config: JudgePromptConfig,
    *,
    question: str,
    prediction: str,
    retrieved_context: str,
) -> RenderedJudgePrompt:
    template = _read_prompt_template(prompt_config.template_path)
    return RenderedJudgePrompt(
        text=template.format(
            question=question,
            prediction=prediction,
            retrieved_context=retrieved_context,
        ),
        prompt_id=prompt_config.id,
        prompt_version=prompt_config.version,
        template_path=prompt_config.template_path,
    )


def parse_grounding_response(raw_response: str) -> dict[str, object]:
    try:
        decoded_response = json.loads(raw_response)
    except JSONDecodeError as exc:
        raise JudgeError("Grounding judge response was not valid JSON") from exc

    if not isinstance(decoded_response, dict):
        raise JudgeError("Grounding judge response must be a JSON object")

    verdict = _string_field(decoded_response, "verdict")
    reason = _string_field(decoded_response, "reason")
    if verdict not in GROUNDING_VERDICTS:
        raise JudgeError(f"Unsupported grounding verdict: {verdict}")

    citation_correct = _optional_bool_field(decoded_response, "citation_correct")
    unsupported_claims_raw = decoded_response.get("unsupported_claims")
    unsupported_claims = str(unsupported_claims_raw) if unsupported_claims_raw is not None else None

    return {
        "verdict": verdict,
        "reason": reason,
        "citation_correct": citation_correct,
        "unsupported_claims": unsupported_claims,
    }


def parse_judge_response(raw_response: str) -> dict[str, object]:
    try:
        decoded_response = json.loads(raw_response)
    except JSONDecodeError as exc:
        raise JudgeError("Judge response was not valid JSON") from exc

    if not isinstance(decoded_response, dict):
        raise JudgeError("Judge response must be a JSON object")

    verdict = _string_field(decoded_response, "verdict")
    reason = _string_field(decoded_response, "reason")
    if verdict not in JUDGE_VERDICTS:
        raise JudgeError(f"Unsupported judge verdict: {verdict}")

    numeric_error = _optional_bool_field(decoded_response, "numeric_error")
    unsupported_claims = _optional_bool_field(decoded_response, "unsupported_claims")

    return {
        "verdict": verdict,
        "reason": reason,
        "numeric_error": numeric_error,
        "unsupported_claims": unsupported_claims,
    }


def summarize_judges(judge_rows: list[dict[str, object]]) -> dict[str, object]:
    attempted_count = len(judge_rows)
    success_count = sum(1 for row in judge_rows if row.get("status") == "success")
    error_count = sum(1 for row in judge_rows if row.get("status") == "error")
    summary: dict[str, object] = {
        "attempted_count": attempted_count,
        "success_count": success_count,
        "error_count": error_count,
    }
    for verdict in JUDGE_VERDICTS:
        count = sum(1 for row in judge_rows if row.get("verdict") == verdict)
        summary[f"{verdict}_count"] = count
        summary[f"{verdict}_rate"] = count / attempted_count if attempted_count else 0.0
    return summary


def _read_prompt_template(template_path: Path) -> str:
    if not template_path.is_file():
        raise JudgeError(f"Judge prompt template file not found: {template_path}")
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JudgeError(
            f"Could not read judge prompt template file: {template_path}"
        ) from exc
    if not template.strip():
        raise JudgeError(
            f"Judge prompt template file must be non-empty: {template_path}"
        )
    return template


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JudgeError(f"Judge prompt field must be non-empty: {field_name}")
    return value


def _joined_evidence_context(evidence: object) -> str:
    evidence_texts: list[str] = []
    if isinstance(evidence, list):
        evidence_texts = [
            item["evidence_text"].strip()
            for item in evidence
            if isinstance(item, dict)
            and isinstance(item.get("evidence_text"), str)
            and item["evidence_text"].strip()
        ]
    if not evidence_texts:
        return "No evidence provided."
    return "\n\n".join(
        f"[Evidence {index}]\n{text}"
        for index, text in enumerate(evidence_texts, start=1)
    )


def _optional_bool_field(
    decoded_response: dict[str, Any], field_name: str
) -> bool | None:
    if field_name not in decoded_response:
        return None
    value = decoded_response[field_name]
    if value is None:
        return None
    if not isinstance(value, bool):
        raise JudgeError(
            f"Judge response field '{field_name}' must be a boolean, got: {type(value).__name__}"
        )
    return value


def _string_field(decoded_response: dict[str, Any], field_name: str) -> str:
    value = decoded_response.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise JudgeError(f"Judge response missing string field: {field_name}")
    return value


__all__ = [
    "GROUNDING_VERDICTS",
    "JUDGE_VERDICTS",
    "JudgeError",
    "RenderedJudgePrompt",
    "parse_grounding_response",
    "parse_judge_response",
    "render_grounding_prompt",
    "render_judge_prompt_for_processed_example",
    "summarize_judges",
]
