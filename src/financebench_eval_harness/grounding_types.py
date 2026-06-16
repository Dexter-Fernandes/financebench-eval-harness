"""M7 taxonomy constants and dataclasses for hallucination and grounding analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# M7.1: Grounding labels — how well the answer is supported by retrieved context.
GROUNDING_LABELS = (
    "grounded",
    "partially_grounded",
    "ungrounded",
    "contradicted",
    "insufficient_evidence",
    "over_refusal",
    "under_refusal",
)

# M7.2: Hallucination failure types — specific ways an answer can be wrong.
HALLUCINATION_FAILURE_TYPES = (
    "wrong_number",
    "wrong_unit",
    "wrong_period",
    "wrong_metric",
    "unsupported_claim",
    "contradicted_by_context",
    "bad_citation",
    "missing_citation",
    "retrieval_miss",
    "generation_error",
    "format_error",
)

# M7.6: Citation support labels.
CITATION_SUPPORT_LABELS = (
    "supports_answer",
    "partially_supports_answer",
    "does_not_support_answer",
    "citation_missing",
    "citation_invalid",
)

# M7.7: Context sufficiency labels.
CONTEXT_SUFFICIENCY_LABELS = (
    "context_sufficient",
    "context_partially_sufficient",
    "context_insufficient",
)

# M7.11: Root cause labels for failed examples.
ROOT_CAUSE_LABELS = (
    "retrieval_failure",
    "generation_failure",
    "citation_failure",
    "hallucination_under_refusal",
    "over_refusal",
    "no_failure",
)


@dataclass(frozen=True)
class GroundingScoreRow:
    """One row of grounding_scores.jsonl."""

    question_id: str
    grounding_label: str
    context_sufficiency: str
    rule_flags: tuple[str, ...]
    judge_grounding_label: str | None
    judge_failure_types: tuple[str, ...]
    judge_citation_quality: str | None
    judge_reason: str | None
    cited_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "grounding_label": self.grounding_label,
            "context_sufficiency": self.context_sufficiency,
            "rule_flags": list(self.rule_flags),
            "judge_grounding_label": self.judge_grounding_label,
            "judge_failure_types": list(self.judge_failure_types),
            "judge_citation_quality": self.judge_citation_quality,
            "judge_reason": self.judge_reason,
            "cited_chunk_ids": list(self.cited_chunk_ids),
        }


@dataclass(frozen=True)
class CitationScoreRow:
    """One row of citation_scores.jsonl."""

    question_id: str
    citation_quality: str
    cited_chunk_ids: tuple[str, ...]
    invalid_chunk_ids: tuple[str, ...]
    missing_citation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "citation_quality": self.citation_quality,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "invalid_chunk_ids": list(self.invalid_chunk_ids),
            "missing_citation": self.missing_citation,
        }


@dataclass(frozen=True)
class FailureAnalysisRow:
    """One row of failure_analysis.jsonl — the M7.10 joined output."""

    question_id: str
    evidence_hit: bool | None
    answer_verdict: str | None
    grounding_label: str | None
    context_sufficiency: str | None
    citation_quality: str | None
    failure_types: tuple[str, ...]
    rule_flags: tuple[str, ...]
    root_cause: str
    company: str | None
    question_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "evidence_hit": self.evidence_hit,
            "answer_verdict": self.answer_verdict,
            "grounding_label": self.grounding_label,
            "context_sufficiency": self.context_sufficiency,
            "citation_quality": self.citation_quality,
            "failure_types": list(self.failure_types),
            "rule_flags": list(self.rule_flags),
            "root_cause": self.root_cause,
            "company": self.company,
            "question_type": self.question_type,
        }


class GroundingAnalysisError(ValueError):
    """Raised when grounding analysis encounters an unrecoverable error."""


__all__ = [
    "CITATION_SUPPORT_LABELS",
    "CONTEXT_SUFFICIENCY_LABELS",
    "GROUNDING_LABELS",
    "HALLUCINATION_FAILURE_TYPES",
    "ROOT_CAUSE_LABELS",
    "CitationScoreRow",
    "FailureAnalysisRow",
    "GroundingAnalysisError",
    "GroundingScoreRow",
]
