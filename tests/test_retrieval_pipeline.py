from __future__ import annotations

import json
from pathlib import Path

from financebench_eval_harness.chunking import ChunkingConfig, chunk_pages
from financebench_eval_harness.embedding import EmbeddingConfig, MockEmbeddingClient
from financebench_eval_harness.index_builder import IndexMetadata, build_index, load_index
from financebench_eval_harness.retrieval_config import RetrievalConfig
from financebench_eval_harness.retrieval_types import Chunk, DocumentPage
from financebench_eval_harness.retriever import Question, retrieve_for_questions, run_retrieval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAGES = [
    DocumentPage(
        doc_id="ACME_2022",
        doc_name="ACME_2022_10K.pdf",
        page_num=1,
        text="Total revenue for fiscal year 2022 was $4.2 billion.",
    ),
    DocumentPage(
        doc_id="ACME_2022",
        doc_name="ACME_2022_10K.pdf",
        page_num=2,
        text="Operating expenses rose to $1.8 billion. Cost of goods sold increased.",
    ),
    DocumentPage(
        doc_id="BETA_2022",
        doc_name="BETA_2022_10K.pdf",
        page_num=5,
        text="Net income was $540 million. Income tax expense was $120 million.",
    ),
]

CHUNKING_CFG = ChunkingConfig(chunk_size=200, chunk_overlap=20)
EMBED_CFG = EmbeddingConfig(provider="mock", model_name="mock-embed")
EMBED_DIM = 8


def make_client() -> MockEmbeddingClient:
    return MockEmbeddingClient(EMBED_CFG, embedding_dim=EMBED_DIM)


def _build_and_load(
    chunks: list[Chunk],
    client: MockEmbeddingClient,
    idx_dir: Path,
) -> tuple:
    config = RetrievalConfig(chunking=CHUNKING_CFG)
    build_index(chunks, client, config, idx_dir)
    return load_index(idx_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrievalPipeline:
    def test_chunks_are_created_from_pages(self) -> None:
        chunks = chunk_pages(PAGES, CHUNKING_CFG)
        assert len(chunks) >= len(PAGES)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_stable(self) -> None:
        chunks_a = chunk_pages(PAGES, CHUNKING_CFG)
        chunks_b = chunk_pages(PAGES, CHUNKING_CFG)
        assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]

    def test_mock_embeddings_are_generated(self) -> None:
        client = make_client()
        vecs = client.embed_texts(["revenue grew", "expenses fell"])
        assert len(vecs) == 2
        assert all(len(v) == EMBED_DIM for v in vecs)

    def test_vector_index_can_be_built(self, tmp_path: Path) -> None:
        chunks = chunk_pages(PAGES, CHUNKING_CFG)
        config = RetrievalConfig(chunking=CHUNKING_CFG)
        meta = build_index(chunks, make_client(), config, tmp_path / "idx")
        assert isinstance(meta, IndexMetadata)
        assert meta.chunk_count == len(chunks)

    def test_query_returns_top_k_chunks(self, tmp_path: Path) -> None:
        chunks = chunk_pages(PAGES, CHUNKING_CFG)
        client = make_client()
        store, _ = _build_and_load(chunks, client, tmp_path / "idx")
        question = Question(question_id="q001", query=PAGES[0].text)
        rows = retrieve_for_questions([question], store, client, top_k=2)
        assert len(rows) == 1
        assert len(rows[0].retrieved) == 2

    def test_retrieved_chunks_include_doc_and_page_metadata(self, tmp_path: Path) -> None:
        chunks = chunk_pages(PAGES, CHUNKING_CFG)
        client = make_client()
        store, _ = _build_and_load(chunks, client, tmp_path / "idx")
        # query exactly matches first chunk text → guaranteed rank-1 hit
        question = Question(question_id="q001", query=chunks[0].text)
        rows = retrieve_for_questions([question], store, client, top_k=1)
        top = rows[0].retrieved[0]
        assert top.doc_name == chunks[0].doc_name
        assert top.page_num == chunks[0].page_num
        assert top.chunk_id == chunks[0].chunk_id

    def test_retrieval_output_jsonl_is_written(self, tmp_path: Path) -> None:
        chunks = chunk_pages(PAGES, CHUNKING_CFG)
        client = make_client()
        store, _ = _build_and_load(chunks, client, tmp_path / "idx")
        question = Question(question_id="q001", query=chunks[0].text)
        result = run_retrieval(
            [question], store, client, tmp_path / "results.jsonl", top_k=3
        )
        assert result.output_path.is_file()
        lines = result.output_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["question_id"] == "q001"
        assert isinstance(row["retrieved"], list)
        assert len(row["retrieved"]) == 3
