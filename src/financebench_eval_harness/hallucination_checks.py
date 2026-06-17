"""M7.9: Rule-based hallucination flags, complementing the LLM judge."""

from __future__ import annotations

import re
from typing import Any

_REFUSAL_PATTERN = re.compile(
    r"\b(cannot determine|can't determine|do not know|don't know|"
    r"unable to determine|not enough information|insufficient information|"
    r"no information|not provided|cannot find)\b",
    re.IGNORECASE,
)


def run_rule_based_checks(
    *,
    prediction: str,
    gold_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    cited_chunk_ids: list[str],
    all_chunk_ids: list[str],
    gold_numeric_values: list[float],
    prediction_numeric_values: list[float],
) -> list[str]:
    """Return a list of triggered rule-based hallucination flags.

    Possible flags:
    - ``no_citation``: prediction has no cited chunk IDs
    - ``cited_chunk_not_retrieved``: a cited ID is not in the retrieved set
    - ``predicted_number_not_in_context``: prediction contains numbers absent from all chunks
    - ``predicted_year_not_in_context``: fiscal year in prediction not found in any chunk
    - ``refusal_with_evidence``: prediction is a refusal despite retrievable context
    """
    flags: list[str] = []
    all_chunk_id_set = set(all_chunk_ids)
    retrieved_set = {str(c.get("chunk_id", "")) for c in retrieved_chunks}

    if not cited_chunk_ids:
        flags.append("no_citation")

    invalid_cited = [cid for cid in cited_chunk_ids if cid not in retrieved_set]
    if invalid_cited:
        flags.append("cited_chunk_not_retrieved")

    if prediction_numeric_values:
        all_chunk_text = " ".join(str(c.get("text", "")) for c in retrieved_chunks)
        chunk_numbers_str = all_chunk_text
        for pv in prediction_numeric_values:
            pv_str = str(abs(pv))
            pv_int = str(int(abs(pv))) if abs(pv) == int(abs(pv)) else None
            found = pv_str in chunk_numbers_str or (pv_int and pv_int in chunk_numbers_str)
            if not found:
                flags.append("predicted_number_not_in_context")
                break

    if _REFUSAL_PATTERN.search(prediction) and retrieved_chunks:
        flags.append("refusal_with_evidence")

    return flags


__all__ = ["run_rule_based_checks"]
