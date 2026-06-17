"""M7.5 / M7.6: Extract cited chunk IDs and check whether they support the answer."""

from __future__ import annotations

from typing import Any

from financebench_eval_harness.claim_extraction import extract_cited_chunk_ids
from financebench_eval_harness.scoring import extract_numeric_values


def score_citations(
    *,
    answer_text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract cited chunk IDs from the answer, validate them, and return a citation quality label.

    Returns a dict with:
    - ``citation_quality``: one of the CITATION_SUPPORT_LABELS values
    - ``cited_chunk_ids``: list of IDs extracted from the answer
    - ``invalid_chunk_ids``: IDs that were cited but not in the retrieved set
    - ``missing_citation``: True when no chunk IDs were found in the answer
    """
    cited_ids = extract_cited_chunk_ids(answer_text)
    retrieved_ids = {str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks}

    if not cited_ids:
        return {
            "citation_quality": "citation_missing",
            "cited_chunk_ids": [],
            "invalid_chunk_ids": [],
            "missing_citation": True,
        }

    invalid_ids = [cid for cid in cited_ids if cid not in retrieved_ids]
    if invalid_ids:
        return {
            "citation_quality": "citation_invalid",
            "cited_chunk_ids": cited_ids,
            "invalid_chunk_ids": invalid_ids,
            "missing_citation": False,
        }

    citation_quality = _assess_support(answer_text, cited_ids, retrieved_chunks)
    return {
        "citation_quality": citation_quality,
        "cited_chunk_ids": cited_ids,
        "invalid_chunk_ids": [],
        "missing_citation": False,
    }


def _assess_support(
    answer_text: str,
    cited_ids: list[str],
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Rule-based check: does any cited chunk contain a number from the answer?"""
    answer_numbers = set(extract_numeric_values(answer_text))
    chunk_by_id = {str(c.get("chunk_id", "")): c for c in retrieved_chunks}

    for cid in cited_ids:
        chunk = chunk_by_id.get(cid)
        if chunk is None:
            continue
        chunk_text = str(chunk.get("text", ""))
        chunk_numbers = set(extract_numeric_values(chunk_text))
        if answer_numbers and answer_numbers & chunk_numbers:
            return "supports_answer"

    return "does_not_support_answer"


__all__ = ["score_citations"]
