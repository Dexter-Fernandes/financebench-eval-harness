"""M5.2 — RAG input format definitions (TDD)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from financebench_eval_harness.rag_types import (
    RAGContextChunk,
    RAGInput,
    RAGInputError,
    build_rag_inputs,
    load_rag_inputs,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CHUNK_DICT: dict[str, Any] = {
    "rank": 1,
    "chunk_id": "ACME_2022_p12_c0",
    "doc_name": "ACME_2022_10K.pdf",
    "page_num": 12,
    "score": 0.92,
    "text": "Revenue was $10 million.",
}

_EXAMPLE: dict[str, Any] = {
    "question_id": "financebench_001",
    "question": "What was FY2022 revenue?",
    "gold_answer": "$10 million",
    "evidence": [
        {
            "evidence_text": "Revenue was $10 million.",
            "doc_name": "ACME_2022_10K.pdf",
            "gold_page_num": 12,
        }
    ],
}

_RETRIEVAL_ROW: dict[str, Any] = {
    "question_id": "financebench_001",
    "query": "What was FY2022 revenue?",
    "retrieved": [_CHUNK_DICT],
}


# ---------------------------------------------------------------------------
# Cycle 1 — RAGContextChunk dataclass
# ---------------------------------------------------------------------------


def test_rag_context_chunk_can_be_constructed() -> None:
    chunk = RAGContextChunk(
        rank=1,
        chunk_id="ACME_2022_p12_c0",
        doc_name="ACME_2022_10K.pdf",
        page_num=12,
        text="Revenue was $10 million.",
        score=0.92,
    )

    assert chunk.rank == 1
    assert chunk.chunk_id == "ACME_2022_p12_c0"
    assert chunk.doc_name == "ACME_2022_10K.pdf"
    assert chunk.page_num == 12
    assert chunk.text == "Revenue was $10 million."
    assert chunk.score == 0.92


def test_rag_context_chunk_score_defaults_to_none() -> None:
    chunk = RAGContextChunk(
        rank=1,
        chunk_id="c0",
        doc_name="doc.pdf",
        page_num=1,
        text="text",
    )
    assert chunk.score is None


def test_rag_context_chunk_to_dict_contains_all_expected_keys() -> None:
    chunk = RAGContextChunk(
        rank=1,
        chunk_id="ACME_2022_p12_c0",
        doc_name="ACME_2022_10K.pdf",
        page_num=12,
        text="Revenue was $10 million.",
        score=0.92,
    )
    d = chunk.to_dict()

    assert d == {
        "rank": 1,
        "chunk_id": "ACME_2022_p12_c0",
        "doc_name": "ACME_2022_10K.pdf",
        "page_num": 12,
        "text": "Revenue was $10 million.",
        "score": 0.92,
    }


def test_rag_context_chunk_to_dict_omits_none_score() -> None:
    chunk = RAGContextChunk(rank=1, chunk_id="c0", doc_name="doc.pdf", page_num=1, text="t")
    d = chunk.to_dict()
    assert "score" not in d


# ---------------------------------------------------------------------------
# Cycle 2 — RAGInput dataclass
# ---------------------------------------------------------------------------


def test_rag_input_can_be_constructed() -> None:
    chunk = RAGContextChunk(rank=1, chunk_id="c0", doc_name="doc.pdf", page_num=1, text="t")
    rag_input = RAGInput(
        question_id="financebench_001",
        question="What was FY2022 revenue?",
        retrieved_context=(chunk,),
        gold_answer="$10 million",
        gold_evidence=({"evidence_text": "Revenue was $10 million."},),
    )

    assert rag_input.question_id == "financebench_001"
    assert rag_input.question == "What was FY2022 revenue?"
    assert len(rag_input.retrieved_context) == 1
    assert rag_input.gold_answer == "$10 million"
    assert len(rag_input.gold_evidence) == 1


def test_rag_input_gold_evidence_defaults_to_empty_tuple() -> None:
    rag_input = RAGInput(
        question_id="q1",
        question="Q?",
        retrieved_context=(),
        gold_answer="A",
    )
    assert rag_input.gold_evidence == ()


def test_rag_input_to_dict_serialises_correctly() -> None:
    chunk = RAGContextChunk(rank=1, chunk_id="c0", doc_name="doc.pdf", page_num=1, text="t", score=0.9)
    evidence = {"evidence_text": "Revenue was $10 million.", "doc_name": "doc.pdf"}
    rag_input = RAGInput(
        question_id="financebench_001",
        question="What was FY2022 revenue?",
        retrieved_context=(chunk,),
        gold_answer="$10 million",
        gold_evidence=(evidence,),
    )

    d = rag_input.to_dict()

    assert d["question_id"] == "financebench_001"
    assert d["question"] == "What was FY2022 revenue?"
    assert d["gold_answer"] == "$10 million"
    assert d["retrieved_context"] == [chunk.to_dict()]
    assert d["gold_evidence"] == [evidence]


# ---------------------------------------------------------------------------
# Cycles 3-5 — build_rag_inputs() joins rows + examples, preserves all fields
# ---------------------------------------------------------------------------


def test_build_rag_inputs_joins_one_retrieval_row_and_one_example() -> None:
    result = build_rag_inputs([_RETRIEVAL_ROW], [_EXAMPLE])

    assert len(result) == 1
    rag_input = result[0]
    assert rag_input.question_id == "financebench_001"
    assert rag_input.question == "What was FY2022 revenue?"


def test_build_rag_inputs_preserves_chunk_fields() -> None:
    result = build_rag_inputs([_RETRIEVAL_ROW], [_EXAMPLE])
    chunk = result[0].retrieved_context[0]

    assert chunk.rank == 1
    assert chunk.chunk_id == "ACME_2022_p12_c0"
    assert chunk.doc_name == "ACME_2022_10K.pdf"
    assert chunk.page_num == 12
    assert chunk.text == "Revenue was $10 million."
    assert chunk.score == 0.92


def test_build_rag_inputs_preserves_gold_fields() -> None:
    result = build_rag_inputs([_RETRIEVAL_ROW], [_EXAMPLE])
    rag_input = result[0]

    assert rag_input.gold_answer == "$10 million"
    assert len(rag_input.gold_evidence) == 1
    assert rag_input.gold_evidence[0]["evidence_text"] == "Revenue was $10 million."


def test_build_rag_inputs_handles_multiple_chunks_in_order() -> None:
    row = {
        "question_id": "financebench_001",
        "query": "Q?",
        "retrieved": [
            {**_CHUNK_DICT, "rank": 1, "chunk_id": "c1", "text": "first"},
            {**_CHUNK_DICT, "rank": 2, "chunk_id": "c2", "text": "second"},
            {**_CHUNK_DICT, "rank": 3, "chunk_id": "c3", "text": "third"},
        ],
    }
    result = build_rag_inputs([row], [_EXAMPLE])
    chunks = result[0].retrieved_context

    assert len(chunks) == 3
    assert chunks[0].rank == 1 and chunks[0].chunk_id == "c1"
    assert chunks[1].rank == 2 and chunks[1].chunk_id == "c2"
    assert chunks[2].rank == 3 and chunks[2].chunk_id == "c3"


def test_build_rag_inputs_handles_multiple_examples() -> None:
    examples = [
        {**_EXAMPLE, "question_id": "q1", "question": "Q1?", "gold_answer": "A1"},
        {**_EXAMPLE, "question_id": "q2", "question": "Q2?", "gold_answer": "A2"},
    ]
    rows = [
        {**_RETRIEVAL_ROW, "question_id": "q1"},
        {**_RETRIEVAL_ROW, "question_id": "q2"},
    ]
    result = build_rag_inputs(rows, examples)

    assert len(result) == 2
    ids = {r.question_id for r in result}
    assert ids == {"q1", "q2"}


def test_build_rag_inputs_ignores_retrieval_rows_with_no_matching_example() -> None:
    orphan_row = {**_RETRIEVAL_ROW, "question_id": "no_match"}
    result = build_rag_inputs([orphan_row], [_EXAMPLE])

    assert len(result) == 1
    assert result[0].question_id == "financebench_001"


# ---------------------------------------------------------------------------
# Cycle 6 — question with no retrieval results gets empty retrieved_context
# ---------------------------------------------------------------------------


def test_build_rag_inputs_question_with_no_retrieval_gets_empty_context() -> None:
    result = build_rag_inputs(retrieval_rows=[], processed_examples=[_EXAMPLE])

    assert len(result) == 1
    assert result[0].retrieved_context == ()


# ---------------------------------------------------------------------------
# Cycles 7-8 — load_rag_inputs() file loader
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows),
        encoding="utf-8",
    )


def test_load_rag_inputs_reads_jsonl_files_and_returns_rag_inputs(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval_results.jsonl"
    examples_path = tmp_path / "examples.jsonl"
    _write_jsonl(retrieval_path, [_RETRIEVAL_ROW])
    _write_jsonl(examples_path, [_EXAMPLE])

    result = load_rag_inputs(retrieval_path, examples_path)

    assert len(result) == 1
    assert result[0].question_id == "financebench_001"
    assert len(result[0].retrieved_context) == 1
    assert result[0].gold_answer == "$10 million"


def test_load_rag_inputs_raises_rag_input_error_when_retrieval_file_missing(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.jsonl"
    _write_jsonl(examples_path, [_EXAMPLE])

    with pytest.raises(RAGInputError, match="retrieval"):
        load_rag_inputs(tmp_path / "missing.jsonl", examples_path)


def test_load_rag_inputs_raises_rag_input_error_when_examples_file_missing(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval_results.jsonl"
    _write_jsonl(retrieval_path, [_RETRIEVAL_ROW])

    with pytest.raises(RAGInputError, match="examples"):
        load_rag_inputs(retrieval_path, tmp_path / "missing.jsonl")
