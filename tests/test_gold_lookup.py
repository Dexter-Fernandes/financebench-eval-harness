"""Tests for gold_lookup.py — M4.2 gold evidence lookup builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.gold_lookup import GoldLookupError, build_gold_lookup


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_record(
    question_id: str = "q1",
    doc_name: str = "3M_2018_10K",
    gold_page_num: object = 59,
    evidence_text: str = "purchases of property plant and equipment",
    page_text: str = "full page text here",
) -> dict:
    return {
        "question_id": question_id,
        "company": "3M",
        "doc_name": doc_name,
        "question": "What were capex?",
        "gold_answer": "$1577M",
        "evidence": [
            {
                "raw_evidence_index": 0,
                "doc_name": doc_name,
                "gold_page_num": gold_page_num,
                "matched_page_num": gold_page_num,
                "evidence_text": evidence_text,
                "page_text": page_text,
                "evidence_quality": {},
            }
        ],
    }


# ---------------------------------------------------------------------------
# Slice 1 — Tracer bullet: single record, correct shape and field values
# ---------------------------------------------------------------------------


def test_build_gold_lookup_returns_dict_keyed_by_question_id(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    _write_jsonl(f, [_make_record("financebench_id_03029", "3M_2018_10K", 59, "purchases of property", "page text")])
    result = build_gold_lookup(f)
    assert list(result.keys()) == ["financebench_id_03029"]
    entry = result["financebench_id_03029"]
    assert entry["evidence_doc_name"] == "3M_2018_10K"
    assert entry["evidence_page_num"] == 59
    assert entry["evidence_text"] == "purchases of property"
    assert entry["page_text"] == "page text"


# ---------------------------------------------------------------------------
# Slice 2 — Multiple records
# ---------------------------------------------------------------------------


def test_build_gold_lookup_handles_multiple_records(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    _write_jsonl(f, [
        _make_record("q1", "3M_2018_10K", 10, "text A", "page A"),
        _make_record("q2", "AMAZON_2019_10K", 20, "text B", "page B"),
    ])
    result = build_gold_lookup(f)
    assert set(result.keys()) == {"q1", "q2"}
    assert result["q1"]["evidence_page_num"] == 10
    assert result["q2"]["evidence_doc_name"] == "AMAZON_2019_10K"


# ---------------------------------------------------------------------------
# Slice 3 — Empty evidence array raises GoldLookupError
# ---------------------------------------------------------------------------


def test_build_gold_lookup_raises_on_empty_evidence_array(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    record = _make_record("q_no_evidence")
    record["evidence"] = []
    _write_jsonl(f, [record])
    with pytest.raises(GoldLookupError, match="q_no_evidence"):
        build_gold_lookup(f)


def test_build_gold_lookup_raises_on_missing_evidence_key(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    record = _make_record("q_no_key")
    del record["evidence"]
    _write_jsonl(f, [record])
    with pytest.raises(GoldLookupError, match="q_no_key"):
        build_gold_lookup(f)


# ---------------------------------------------------------------------------
# Slice 4 — Missing/invalid required fields in evidence[0]
# ---------------------------------------------------------------------------


def test_build_gold_lookup_raises_on_missing_doc_name(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    record = _make_record("q1")
    del record["evidence"][0]["doc_name"]
    _write_jsonl(f, [record])
    with pytest.raises(GoldLookupError, match="doc_name"):
        build_gold_lookup(f)


def test_build_gold_lookup_raises_on_non_int_gold_page_num(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    _write_jsonl(f, [_make_record("q1", gold_page_num="59")])
    with pytest.raises(GoldLookupError, match="gold_page_num"):
        build_gold_lookup(f)


def test_build_gold_lookup_raises_on_empty_evidence_text(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    _write_jsonl(f, [_make_record("q1", evidence_text="")])
    with pytest.raises(GoldLookupError, match="evidence_text"):
        build_gold_lookup(f)


def test_build_gold_lookup_raises_on_missing_page_text_key(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    record = _make_record("q1")
    del record["evidence"][0]["page_text"]
    _write_jsonl(f, [record])
    with pytest.raises(GoldLookupError, match="page_text"):
        build_gold_lookup(f)


def test_build_gold_lookup_allows_empty_page_text(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    _write_jsonl(f, [_make_record("q1", page_text="")])
    result = build_gold_lookup(f)
    assert result["q1"]["page_text"] == ""


# ---------------------------------------------------------------------------
# Slice 5 — Duplicate question_id raises GoldLookupError
# ---------------------------------------------------------------------------


def test_build_gold_lookup_raises_on_duplicate_question_id(tmp_path: Path) -> None:
    f = tmp_path / "examples.jsonl"
    record = _make_record("q_dup")
    _write_jsonl(f, [record, record])
    with pytest.raises(GoldLookupError, match="q_dup"):
        build_gold_lookup(f)
