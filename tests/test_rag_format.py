"""M5.4 — Retrieved context formatting (TDD)."""
from __future__ import annotations

import pytest

from financebench_eval_harness.rag_types import RAGContextChunk, format_retrieved_context

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _chunk(rank: int, chunk_id: str, doc_name: str, page_num: int, text: str) -> RAGContextChunk:
    return RAGContextChunk(rank=rank, chunk_id=chunk_id, doc_name=doc_name, page_num=page_num, text=text)


CHUNK_1 = _chunk(1, "3M_2022_10K_p12_c03", "3M_2022_10K.pdf", 12, "Revenue was $10 million.")
CHUNK_2 = _chunk(2, "3M_2022_10K_p13_c01", "3M_2022_10K.pdf", 13, "Operating expenses rose to $1.8 billion.")


# ---------------------------------------------------------------------------
# Cycles 1-2 — single-chunk format
# ---------------------------------------------------------------------------


def test_format_single_chunk_contains_rank_and_chunk_id() -> None:
    result = format_retrieved_context([CHUNK_1])
    assert "[1] chunk_id=3M_2022_10K_p12_c03" in result


def test_format_single_chunk_contains_document_label() -> None:
    result = format_retrieved_context([CHUNK_1])
    assert "Document: 3M_2022_10K.pdf" in result


def test_format_single_chunk_contains_page_label() -> None:
    result = format_retrieved_context([CHUNK_1])
    assert "Page: 12" in result


def test_format_single_chunk_contains_text_label() -> None:
    result = format_retrieved_context([CHUNK_1])
    assert "Text:" in result


def test_format_single_chunk_contains_chunk_text() -> None:
    result = format_retrieved_context([CHUNK_1])
    assert "Revenue was $10 million." in result


def test_format_single_chunk_exact_block_structure() -> None:
    result = format_retrieved_context([CHUNK_1])
    expected = (
        "[1] chunk_id=3M_2022_10K_p12_c03\n"
        "Document: 3M_2022_10K.pdf\n"
        "Page: 12\n"
        "Text:\n"
        "Revenue was $10 million."
    )
    assert result == expected


# ---------------------------------------------------------------------------
# Cycles 3-4 — multi-chunk output
# ---------------------------------------------------------------------------


def test_format_two_chunks_uses_correct_rank_numbers() -> None:
    result = format_retrieved_context([CHUNK_1, CHUNK_2])
    assert "[1] chunk_id=3M_2022_10K_p12_c03" in result
    assert "[2] chunk_id=3M_2022_10K_p13_c01" in result


def test_format_two_chunks_rank_one_precedes_rank_two() -> None:
    result = format_retrieved_context([CHUNK_1, CHUNK_2])
    assert result.index("[1]") < result.index("[2]")


def test_format_two_chunks_separated_by_blank_line() -> None:
    result = format_retrieved_context([CHUNK_1, CHUNK_2])
    assert "Revenue was $10 million.\n\n[2]" in result


def test_format_two_chunks_exact_structure() -> None:
    result = format_retrieved_context([CHUNK_1, CHUNK_2])
    expected = (
        "[1] chunk_id=3M_2022_10K_p12_c03\n"
        "Document: 3M_2022_10K.pdf\n"
        "Page: 12\n"
        "Text:\n"
        "Revenue was $10 million.\n\n"
        "[2] chunk_id=3M_2022_10K_p13_c01\n"
        "Document: 3M_2022_10K.pdf\n"
        "Page: 13\n"
        "Text:\n"
        "Operating expenses rose to $1.8 billion."
    )
    assert result == expected


# ---------------------------------------------------------------------------
# Cycle 5 — empty chunk list
# ---------------------------------------------------------------------------


def test_format_empty_chunk_list_returns_empty_string() -> None:
    assert format_retrieved_context([]) == ""


# ---------------------------------------------------------------------------
# Cycles 6-10 — max_context_chars truncation
# ---------------------------------------------------------------------------


def test_format_with_no_max_returns_full_output() -> None:
    result = format_retrieved_context([CHUNK_1, CHUNK_2], max_context_chars=None)
    assert "Revenue was $10 million." in result
    assert "Operating expenses rose to $1.8 billion." in result


def test_format_with_max_equal_to_full_length_returns_untruncated() -> None:
    full = format_retrieved_context([CHUNK_1, CHUNK_2])
    result = format_retrieved_context([CHUNK_1, CHUNK_2], max_context_chars=len(full))
    assert result == full


def test_format_with_max_larger_than_full_length_returns_untruncated() -> None:
    full = format_retrieved_context([CHUNK_1, CHUNK_2])
    result = format_retrieved_context([CHUNK_1, CHUNK_2], max_context_chars=len(full) + 1000)
    assert result == full


def test_format_truncated_output_length_does_not_exceed_max() -> None:
    max_chars = 50
    result = format_retrieved_context([CHUNK_1, CHUNK_2], max_context_chars=max_chars)
    assert len(result) <= max_chars


def test_format_truncated_output_ends_with_truncation_marker() -> None:
    max_chars = 50
    result = format_retrieved_context([CHUNK_1, CHUNK_2], max_context_chars=max_chars)
    assert result.endswith("… [truncated]")


def test_format_truncation_preserves_content_up_to_budget() -> None:
    full = format_retrieved_context([CHUNK_1, CHUNK_2])
    max_chars = len(full) // 2
    result = format_retrieved_context([CHUNK_1, CHUNK_2], max_context_chars=max_chars)
    marker = "… [truncated]"
    content_part = result[: -len(marker)]
    assert full.startswith(content_part)


def test_format_truncation_still_contains_first_chunk_rank() -> None:
    # Even with a tight budget, the start of the output should be there
    result = format_retrieved_context([CHUNK_1, CHUNK_2], max_context_chars=80)
    assert "[1]" in result


def test_format_single_chunk_with_tight_max_truncates_safely() -> None:
    max_chars = 30
    result = format_retrieved_context([CHUNK_1], max_context_chars=max_chars)
    assert len(result) <= max_chars
    assert result.endswith("… [truncated]")
