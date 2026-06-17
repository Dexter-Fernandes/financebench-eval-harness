"""M7.3 / M7.5: Break answers into checkable claims; extract cited chunk IDs."""

from __future__ import annotations

import re

_CHUNK_ID_PATTERN = re.compile(r"\[chunk_id:\s*([^\]\s]+)\]", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand|percent|%))?\b",
                               re.IGNORECASE)


def extract_cited_chunk_ids(answer_text: str) -> list[str]:
    """Extract chunk_id references from model answer text.

    Recognises the pattern ``[chunk_id: <id>]`` produced by the RAG prompt formatter.
    Returns a deduplicated list in order of first appearance.
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in _CHUNK_ID_PATTERN.finditer(answer_text):
        chunk_id = match.group(1).strip()
        if chunk_id not in seen:
            seen.add(chunk_id)
            result.append(chunk_id)
    return result


def extract_claims(answer_text: str) -> list[dict[str, str]]:
    """Split an answer into checkable atomic claims (v1: rule-based).

    Each claim is a sentence-sized fragment of the answer. Claims are classified as:
    - ``numeric``: contains a number or financial figure
    - ``evidence_reference``: contains a chunk_id citation
    - ``textual``: everything else
    """
    sentences = _split_sentences(answer_text)
    claims: list[dict[str, str]] = []
    for index, sentence in enumerate(sentences, start=1):
        sentence = sentence.strip()
        if not sentence:
            continue
        claim_id = f"claim_{index:03d}"
        claim_type = _classify_claim(sentence)
        claims.append({
            "claim_id": claim_id,
            "claim_text": sentence,
            "claim_type": claim_type,
        })
    return claims


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _classify_claim(sentence: str) -> str:
    if _CHUNK_ID_PATTERN.search(sentence):
        return "evidence_reference"
    if _NUMERIC_PATTERN.search(sentence):
        return "numeric"
    return "textual"


__all__ = [
    "extract_cited_chunk_ids",
    "extract_claims",
]
