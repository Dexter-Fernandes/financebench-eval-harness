"""Gold evidence lookup — M4.2.

Loads examples.jsonl and returns a dict keyed by question_id so the
eval-retrieval pipeline can join retrieval results with gold evidence
without re-parsing the file on every question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["GoldLookupError", "build_gold_lookup"]

GoldLookup = dict[str, dict[str, Any]]


class GoldLookupError(ValueError):
    """Raised when the gold lookup cannot be built from the given JSONL."""


def build_gold_lookup(examples_path: str | Path) -> GoldLookup:
    """Load examples.jsonl and return {question_id: evidence_entry}.

    Each entry contains evidence_doc_name, evidence_page_num, evidence_text,
    and page_text extracted from evidence[0] of each record.

    Raises GoldLookupError on empty evidence arrays, missing/invalid required
    fields in evidence[0], or duplicate question_id values.
    """
    path = Path(examples_path)
    rows = _read_jsonl_rows(path)
    lookup: GoldLookup = {}
    for row in rows:
        qid = row["question_id"]
        if qid in lookup:
            raise GoldLookupError(
                f"Duplicate question_id found in '{path}': '{qid}'"
            )
        lookup[qid] = _extract_evidence_entry(row, qid)
    return lookup


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _extract_evidence_entry(record: dict[str, Any], question_id: str) -> dict[str, Any]:
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or len(evidence) == 0:
        raise GoldLookupError(
            f"Record '{question_id}' has empty or missing evidence array"
        )
    ev = evidence[0]
    if not isinstance(ev, dict):
        raise GoldLookupError(
            f"Record '{question_id}': evidence[0] must be an object"
        )

    doc_name = ev.get("doc_name")
    if not isinstance(doc_name, str) or not doc_name.strip():
        raise GoldLookupError(
            f"Record '{question_id}': evidence[0] missing required 'doc_name'"
        )

    gold_page_num = ev.get("gold_page_num")
    if type(gold_page_num) is not int:
        raise GoldLookupError(
            f"Record '{question_id}': evidence[0] 'gold_page_num' must be an int, "
            f"got {type(gold_page_num).__name__!r}"
        )

    evidence_text = ev.get("evidence_text")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise GoldLookupError(
            f"Record '{question_id}': evidence[0] missing required 'evidence_text'"
        )

    if "page_text" not in ev:
        raise GoldLookupError(
            f"Record '{question_id}': evidence[0] missing key 'page_text'"
        )

    if "matched_page_num" not in ev:
        raise GoldLookupError(
            f"Record '{question_id}': evidence[0] missing key 'matched_page_num'"
        )
    matched_page_num = ev["matched_page_num"]

    return {
        "evidence_doc_name": doc_name,
        "evidence_page_num": gold_page_num,
        "matched_page_num": matched_page_num,
        "page_num_mismatch": gold_page_num != matched_page_num,
        "evidence_text": evidence_text,
        "page_text": ev["page_text"],
    }
