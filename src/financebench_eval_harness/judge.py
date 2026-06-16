from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from financebench_eval_harness.run_config import JudgePromptConfig

JUDGE_VERDICTS = ("correct", "partially_correct", "incorrect", "not_answered")
GROUNDING_VERDICTS = ("grounded", "partially_grounded", "ungrounded")

# M7.8: expanded grounding labels for the v2 judge
M7_GROUNDING_LABELS = (
    "grounded",
    "partially_grounded",
    "ungrounded",
    "contradicted",
    "insufficient_evidence",
    "over_refusal",
    "under_refusal",
)
_M7_FAILURE_TYPES = (
    "wrong_number", "wrong_unit", "wrong_period", "wrong_metric",
    "unsupported_claim", "contradicted_by_context", "bad_citation",
    "missing_citation", "retrieval_miss", "generation_error", "format_error",
)
_M7_CITATION_QUALITY_LABELS = (
    "supports_answer", "partially_supports_answer", "does_not_support_answer",
    "citation_missing", "citation_invalid",
)
_M7_CONTEXT_SUFFICIENCY_LABELS = (
    "context_sufficient", "context_partially_sufficient", "context_insufficient",
)


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


def render_grounding_prompt_v2(
    prompt_config: JudgePromptConfig,
    *,
    question: str,
    gold_answer: str,
    prediction: str,
    retrieved_context: str,
    cited_chunks: str,
    gold_evidence: str,
) -> RenderedJudgePrompt:
    """Render the M7.8 grounding judge prompt (answer_grounding_v2.txt)."""
    template = _read_prompt_template(prompt_config.template_path)
    return RenderedJudgePrompt(
        text=template.format(
            question=question,
            gold_answer=gold_answer,
            prediction=prediction,
            retrieved_context=retrieved_context,
            cited_chunks=cited_chunks,
            gold_evidence=gold_evidence,
        ),
        prompt_id=prompt_config.id,
        prompt_version=prompt_config.version,
        template_path=prompt_config.template_path,
    )


def parse_grounding_response_v2(raw_response: str) -> dict[str, Any]:
    """Parse the M7.8 judge response: grounding_label, failure_types, context_sufficiency,
    citation_quality, reason.
    """
    try:
        decoded = json.loads(raw_response)
    except JSONDecodeError as exc:
        raise JudgeError("M7 grounding judge response was not valid JSON") from exc

    if not isinstance(decoded, dict):
        raise JudgeError("M7 grounding judge response must be a JSON object")

    grounding_label = _string_field(decoded, "grounding_label")
    if grounding_label not in M7_GROUNDING_LABELS:
        raise JudgeError(f"Unknown M7 grounding label: {grounding_label!r}")

    raw_failure_types = decoded.get("failure_types", [])
    if not isinstance(raw_failure_types, list):
        raise JudgeError("M7 grounding judge 'failure_types' must be a list")
    for ft in raw_failure_types:
        if ft not in _M7_FAILURE_TYPES:
            raise JudgeError(f"Unknown M7 failure type: {ft!r}")

    context_sufficiency = decoded.get("context_sufficiency")
    if context_sufficiency is not None and context_sufficiency not in _M7_CONTEXT_SUFFICIENCY_LABELS:
        raise JudgeError(f"Unknown context_sufficiency label: {context_sufficiency!r}")

    citation_quality = decoded.get("citation_quality")
    if citation_quality is not None and citation_quality not in _M7_CITATION_QUALITY_LABELS:
        raise JudgeError(f"Unknown citation_quality label: {citation_quality!r}")

    reason = _string_field(decoded, "reason")

    return {
        "grounding_label": grounding_label,
        "failure_types": list(raw_failure_types),
        "context_sufficiency": context_sufficiency,
        "citation_quality": citation_quality,
        "reason": reason,
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
    "M7_GROUNDING_LABELS",
    "RenderedJudgePrompt",
    "parse_grounding_response",
    "parse_grounding_response_v2",
    "parse_judge_response",
    "render_grounding_prompt",
    "render_grounding_prompt_v2",
    "render_judge_prompt_for_processed_example",
    "summarize_judges",
]
