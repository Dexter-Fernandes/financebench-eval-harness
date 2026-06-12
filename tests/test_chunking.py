from __future__ import annotations

import pytest

from financebench_eval_harness.chunking import ChunkingConfig, chunk_page, chunk_pages
from financebench_eval_harness.retrieval_types import Chunk, DocumentPage


SMALL_CONFIG = ChunkingConfig(chunk_size=100, chunk_overlap=20)
DEFAULT_CONFIG = ChunkingConfig(chunk_size=800, chunk_overlap=150)


def make_page(text: str, doc_id: str = "DOC_A", page_num: int = 1) -> DocumentPage:
    return DocumentPage(
        doc_id=doc_id,
        doc_name=f"{doc_id}.pdf",
        page_num=page_num,
        text=text,
    )


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------


class TestChunkingConfig:
    def test_stores_chunk_size_and_overlap(self) -> None:
        cfg = ChunkingConfig(chunk_size=800, chunk_overlap=150)
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 150

    def test_default_strategy_is_recursive_text(self) -> None:
        cfg = ChunkingConfig(chunk_size=800, chunk_overlap=150)
        assert cfg.strategy == "recursive_text"

    def test_strategy_configurable(self) -> None:
        cfg = ChunkingConfig(chunk_size=400, chunk_overlap=80, strategy="recursive_text")
        assert cfg.strategy == "recursive_text"

    def test_is_immutable(self) -> None:
        cfg = ChunkingConfig(chunk_size=800, chunk_overlap=150)
        with pytest.raises(Exception):
            cfg.chunk_size = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# chunk_page — edge cases
# ---------------------------------------------------------------------------


class TestChunkPageEdgeCases:
    def test_empty_text_yields_no_chunks(self) -> None:
        page = make_page("")
        chunks = chunk_page(page, SMALL_CONFIG)
        assert chunks == []

    def test_whitespace_only_yields_no_chunks(self) -> None:
        page = make_page("   \n\n\t  ")
        chunks = chunk_page(page, SMALL_CONFIG)
        assert chunks == []

    def test_short_text_yields_one_chunk(self) -> None:
        page = make_page("Short text.", SMALL_CONFIG)
        chunks = chunk_page(page, SMALL_CONFIG)
        assert len(chunks) == 1

    def test_single_chunk_preserves_full_text(self) -> None:
        text = "Short text."
        page = make_page(text)
        chunks = chunk_page(page, SMALL_CONFIG)
        assert chunks[0].text == text


# ---------------------------------------------------------------------------
# chunk_page — chunk content and structure
# ---------------------------------------------------------------------------


class TestChunkPageContent:
    def test_each_chunk_carries_page_metadata(self) -> None:
        text = "word " * 200
        page = make_page(text, doc_id="MSFT_2023_10K", page_num=7)
        chunks = chunk_page(page, SMALL_CONFIG)
        for chunk in chunks:
            assert chunk.doc_id == "MSFT_2023_10K"
            assert chunk.doc_name == "MSFT_2023_10K.pdf"
            assert chunk.page_num == 7

    def test_long_text_produces_multiple_chunks(self) -> None:
        text = "word " * 200
        page = make_page(text)
        chunks = chunk_page(page, SMALL_CONFIG)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_chunk_size_by_more_than_one_word(self) -> None:
        text = "word " * 300
        page = make_page(text)
        chunks = chunk_page(page, SMALL_CONFIG)
        # Allow one-word overflow at word boundary
        for chunk in chunks:
            assert len(chunk.text) <= SMALL_CONFIG.chunk_size + 20

    def test_consecutive_chunks_overlap(self) -> None:
        text = "word " * 200
        page = make_page(text)
        chunks = chunk_page(page, SMALL_CONFIG)
        assert len(chunks) >= 2
        tail_of_first = chunks[0].text[-SMALL_CONFIG.chunk_overlap:]
        head_of_second = chunks[1].text[:SMALL_CONFIG.chunk_overlap]
        assert tail_of_first in chunks[1].text or head_of_second in chunks[0].text

    def test_all_text_is_covered(self) -> None:
        """Every character in the page appears in at least one chunk."""
        text = "alpha beta gamma delta epsilon " * 30
        page = make_page(text)
        chunks = chunk_page(page, SMALL_CONFIG)
        covered = set()
        for chunk in chunks:
            start = chunk.start_char
            covered.update(range(start, start + len(chunk.text)))
        assert covered == set(range(len(text)))

    def test_start_and_end_char_reflect_position_in_page_text(self) -> None:
        text = "alpha beta gamma delta epsilon " * 30
        page = make_page(text)
        chunks = chunk_page(page, SMALL_CONFIG)
        for chunk in chunks:
            assert text[chunk.start_char : chunk.start_char + len(chunk.text)] == chunk.text


# ---------------------------------------------------------------------------
# chunk_page — stable IDs
# ---------------------------------------------------------------------------


class TestChunkPageIds:
    def test_chunk_ids_are_unique_within_page(self) -> None:
        text = "word " * 300
        page = make_page(text, doc_id="DOC_X", page_num=5)
        chunks = chunk_page(page, SMALL_CONFIG)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_embeds_doc_id(self) -> None:
        text = "word " * 300
        page = make_page(text, doc_id="AAPL_2022_10K", page_num=3)
        chunks = chunk_page(page, SMALL_CONFIG)
        for chunk in chunks:
            assert "AAPL_2022_10K" in chunk.chunk_id

    def test_chunk_id_embeds_page_num(self) -> None:
        text = "word " * 300
        page = make_page(text, doc_id="DOC_A", page_num=42)
        chunks = chunk_page(page, SMALL_CONFIG)
        for chunk in chunks:
            assert "42" in chunk.chunk_id

    def test_chunking_is_deterministic(self) -> None:
        text = "word " * 300
        page = make_page(text)
        first = chunk_page(page, SMALL_CONFIG)
        second = chunk_page(page, SMALL_CONFIG)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.text for c in first] == [c.text for c in second]

    def test_chunk_offset_shifts_chunk_index(self) -> None:
        text = "word " * 300
        page = make_page(text)
        chunks_zero = chunk_page(page, SMALL_CONFIG, chunk_offset=0)
        chunks_ten = chunk_page(page, SMALL_CONFIG, chunk_offset=10)
        # IDs should differ; chunk at offset 10 has higher index in its id
        assert chunks_zero[0].chunk_id != chunks_ten[0].chunk_id


# ---------------------------------------------------------------------------
# chunk_pages — multi-page aggregation
# ---------------------------------------------------------------------------


class TestChunkPages:
    def test_empty_page_list_yields_no_chunks(self) -> None:
        assert chunk_pages([], SMALL_CONFIG) == []

    def test_returns_chunks_for_all_pages(self) -> None:
        pages = [
            make_page("word " * 200, doc_id="DOC_A", page_num=1),
            make_page("word " * 200, doc_id="DOC_A", page_num=2),
        ]
        chunks = chunk_pages(pages, SMALL_CONFIG)
        page_nums = {c.page_num for c in chunks}
        assert page_nums == {1, 2}

    def test_empty_pages_are_skipped(self) -> None:
        pages = [
            make_page("", doc_id="DOC_A", page_num=1),
            make_page("word " * 200, doc_id="DOC_A", page_num=2),
        ]
        chunks = chunk_pages(pages, SMALL_CONFIG)
        assert all(c.page_num == 2 for c in chunks)

    def test_chunk_ids_are_unique_across_pages(self) -> None:
        pages = [
            make_page("word " * 200, doc_id="DOC_A", page_num=i)
            for i in range(1, 4)
        ]
        chunks = chunk_pages(pages, SMALL_CONFIG)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_returns_list_of_chunk_objects(self) -> None:
        pages = [make_page("hello world", doc_id="DOC_A", page_num=1)]
        chunks = chunk_pages(pages, SMALL_CONFIG)
        assert all(isinstance(c, Chunk) for c in chunks)
