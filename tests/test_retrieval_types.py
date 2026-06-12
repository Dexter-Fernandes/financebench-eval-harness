from __future__ import annotations

import pytest

from financebench_eval_harness.retrieval_types import (
    Chunk,
    DocumentPage,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# DocumentPage
# ---------------------------------------------------------------------------


class TestDocumentPage:
    def test_has_doc_id_separate_from_doc_name(self) -> None:
        page = DocumentPage(
            doc_id="3M_2022_10K",
            doc_name="3M_2022_10K.pdf",
            page_num=12,
            text="Net sales were $35.4 billion.",
        )
        assert page.doc_id == "3M_2022_10K"
        assert page.doc_name == "3M_2022_10K.pdf"

    def test_page_num_and_text_stored(self) -> None:
        page = DocumentPage(
            doc_id="APPLE_2022_10K",
            doc_name="APPLE_2022_10K.pdf",
            page_num=5,
            text="Revenue grew 8% year over year.",
        )
        assert page.page_num == 5
        assert page.text == "Revenue grew 8% year over year."

    def test_is_immutable(self) -> None:
        page = DocumentPage(
            doc_id="3M_2022_10K",
            doc_name="3M_2022_10K.pdf",
            page_num=1,
            text="",
        )
        with pytest.raises(Exception):
            page.doc_id = "changed"  # type: ignore[misc]

    def test_doc_id_is_stable_stem_of_doc_name(self) -> None:
        """doc_id should be derivable from doc_name by stripping the extension."""
        page = DocumentPage(
            doc_id="MSFT_2023_10K",
            doc_name="MSFT_2023_10K.pdf",
            page_num=3,
            text="",
        )
        assert page.doc_id == page.doc_name.removesuffix(".pdf")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_fields_stored(self) -> None:
        chunk = Chunk(
            chunk_id="3M_2022_10K_p12_c03",
            doc_id="3M_2022_10K",
            doc_name="3M_2022_10K.pdf",
            page_num=12,
            text="Net sales were $35.4 billion.",
            start_char=1200,
            end_char=1900,
        )
        assert chunk.chunk_id == "3M_2022_10K_p12_c03"
        assert chunk.doc_id == "3M_2022_10K"
        assert chunk.doc_name == "3M_2022_10K.pdf"
        assert chunk.page_num == 12
        assert chunk.text == "Net sales were $35.4 billion."
        assert chunk.start_char == 1200
        assert chunk.end_char == 1900

    def test_is_immutable(self) -> None:
        chunk = Chunk(
            chunk_id="3M_2022_10K_p01_c00",
            doc_id="3M_2022_10K",
            doc_name="3M_2022_10K.pdf",
            page_num=1,
            text="",
            start_char=0,
            end_char=0,
        )
        with pytest.raises(Exception):
            chunk.chunk_id = "changed"  # type: ignore[misc]

    def test_chunk_id_encodes_doc_page_and_index(self) -> None:
        """chunk_id should embed doc_id, page_num, and chunk index for traceability."""
        chunk = Chunk(
            chunk_id="APPLE_2022_10K_p005_c02",
            doc_id="APPLE_2022_10K",
            doc_name="APPLE_2022_10K.pdf",
            page_num=5,
            text="Services revenue increased to $78.1 billion.",
            start_char=400,
            end_char=900,
        )
        assert chunk.doc_id in chunk.chunk_id
        assert "p005" in chunk.chunk_id
        assert "c02" in chunk.chunk_id

    def test_chunk_preserves_document_provenance(self) -> None:
        """Every chunk must carry enough metadata to trace back to its source document."""
        chunk = Chunk(
            chunk_id="MSFT_2023_10K_p010_c01",
            doc_id="MSFT_2023_10K",
            doc_name="MSFT_2023_10K.pdf",
            page_num=10,
            text="Cloud revenue reached $111 billion.",
            start_char=0,
            end_char=500,
        )
        assert chunk.doc_id is not None
        assert chunk.doc_name is not None
        assert chunk.page_num is not None

    def test_text_extent_is_consistent(self) -> None:
        text = "Cloud revenue reached $111 billion."
        chunk = Chunk(
            chunk_id="MSFT_2023_10K_p010_c00",
            doc_id="MSFT_2023_10K",
            doc_name="MSFT_2023_10K.pdf",
            page_num=10,
            text=text,
            start_char=0,
            end_char=len(text),
        )
        assert chunk.end_char - chunk.start_char == len(chunk.text)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_fields_stored(self) -> None:
        result = RetrievalResult(
            question_id="financebench_001",
            chunk_id="3M_2022_10K_p12_c03",
            score=0.82,
            rank=1,
        )
        assert result.question_id == "financebench_001"
        assert result.chunk_id == "3M_2022_10K_p12_c03"
        assert result.score == pytest.approx(0.82)
        assert result.rank == 1

    def test_is_immutable(self) -> None:
        result = RetrievalResult(
            question_id="financebench_001",
            chunk_id="3M_2022_10K_p12_c03",
            score=0.5,
            rank=1,
        )
        with pytest.raises(Exception):
            result.rank = 99  # type: ignore[misc]

    def test_traceable_to_source_chunk(self) -> None:
        """chunk_id on RetrievalResult must be enough to look up the originating chunk."""
        chunk = Chunk(
            chunk_id="3M_2022_10K_p12_c03",
            doc_id="3M_2022_10K",
            doc_name="3M_2022_10K.pdf",
            page_num=12,
            text="Net sales were $35.4 billion.",
            start_char=1200,
            end_char=1900,
        )
        result = RetrievalResult(
            question_id="financebench_001",
            chunk_id=chunk.chunk_id,
            score=0.82,
            rank=1,
        )
        assert result.chunk_id == chunk.chunk_id

    def test_rank_one_is_top_result(self) -> None:
        top = RetrievalResult(
            question_id="financebench_002",
            chunk_id="APPLE_2022_10K_p005_c00",
            score=0.91,
            rank=1,
        )
        second = RetrievalResult(
            question_id="financebench_002",
            chunk_id="APPLE_2022_10K_p005_c01",
            score=0.78,
            rank=2,
        )
        assert top.rank < second.rank
        assert top.score > second.score
