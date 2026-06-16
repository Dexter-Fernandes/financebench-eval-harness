"""M7.7: Check whether retrieved context was sufficient to answer the question."""

from __future__ import annotations

from financebench_eval_harness.scoring import extract_numeric_values


def check_context_sufficiency(
    *,
    gold_answer: str,
    retrieved_chunks: list[dict],
    evidence_hit: bool | None = None,
) -> str:
    """Classify how sufficient the retrieved context was.

    Returns one of: ``context_sufficient``, ``context_partially_sufficient``,
    ``context_insufficient``.

    Logic:
    1. If ``evidence_hit`` is explicitly True, the retriever confirmed a match → sufficient.
    2. If any gold numeric value appears verbatim in a retrieved chunk → partially sufficient.
    3. Otherwise → insufficient.
    """
    if evidence_hit is True:
        return "context_sufficient"

    gold_numbers = extract_numeric_values(gold_answer)
    if gold_numbers:
        all_chunk_text = " ".join(
            str(chunk.get("text", "")) for chunk in retrieved_chunks
        )
        chunk_numbers = set(extract_numeric_values(all_chunk_text))
        if any(abs(g - c) < 1e-6 for g in gold_numbers for c in chunk_numbers):
            return "context_partially_sufficient"

    return "context_insufficient"


__all__ = ["check_context_sufficiency"]
