from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from financebench_eval_harness.chunking import ChunkingConfig
from financebench_eval_harness.embedding import EmbeddingConfig, MockEmbeddingClient
from financebench_eval_harness.index_builder import (
    IndexMetadata,
    IndexBuildError,
    build_index,
    load_index,
)
from financebench_eval_harness.retrieval_config import RetrievalConfig
from financebench_eval_harness.retrieval_types import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chunk(chunk_id: str, text: str = "some financial text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="DOC_A",
        doc_name="DOC_A.pdf",
        page_num=1,
        text=text,
        start_char=0,
        end_char=len(text),
    )


CHUNKING = ChunkingConfig(chunk_size=800, chunk_overlap=150, min_chunk_chars=50)
RETRIEVAL_CFG = RetrievalConfig(chunking=CHUNKING, evidence_overlap_threshold=0.5)
EMBED_CFG = EmbeddingConfig(provider="mock", model_name="mock-embed")
EMBED_DIM = 8


def make_client() -> MockEmbeddingClient:
    return MockEmbeddingClient(EMBED_CFG, embedding_dim=EMBED_DIM)


CHUNKS = [
    make_chunk("c001", "Revenue increased by 12% year over year."),
    make_chunk("c002", "Operating expenses rose to $4.2 billion."),
    make_chunk("c003", "Net income was $1.8 billion for the quarter."),
]


# ---------------------------------------------------------------------------
# IndexMetadata
# ---------------------------------------------------------------------------


class TestIndexMetadata:
    def test_stores_required_fields(self) -> None:
        meta = IndexMetadata(
            embedding_provider="mock",
            embedding_model="mock-embed",
            chunk_size=800,
            chunk_overlap=150,
            min_chunk_chars=50,
            corpus_hash="abc123",
            chunk_count=10,
            build_time_utc="2026-06-12T12:00:00+00:00",
        )
        assert meta.embedding_provider == "mock"
        assert meta.embedding_model == "mock-embed"
        assert meta.chunk_size == 800
        assert meta.chunk_overlap == 150
        assert meta.min_chunk_chars == 50
        assert meta.corpus_hash == "abc123"
        assert meta.chunk_count == 10
        assert meta.build_time_utc == "2026-06-12T12:00:00+00:00"

    def test_is_immutable(self) -> None:
        meta = IndexMetadata(
            embedding_provider="mock",
            embedding_model="mock-embed",
            chunk_size=800,
            chunk_overlap=150,
            min_chunk_chars=50,
            corpus_hash="abc",
            chunk_count=1,
            build_time_utc="2026-06-12T12:00:00+00:00",
        )
        with pytest.raises(Exception):
            meta.chunk_count = 99  # type: ignore[misc]

    def test_to_dict_gives_plain_types(self) -> None:
        meta = IndexMetadata(
            embedding_provider="mock",
            embedding_model="mock-embed",
            chunk_size=800,
            chunk_overlap=150,
            min_chunk_chars=50,
            corpus_hash="abc",
            chunk_count=3,
            build_time_utc="2026-06-12T12:00:00+00:00",
        )
        d = meta.to_dict()
        json.dumps(d)  # raises if not JSON-serialisable

    def test_to_dict_includes_all_fields(self) -> None:
        meta = IndexMetadata(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            chunk_size=600,
            chunk_overlap=100,
            min_chunk_chars=80,
            corpus_hash="deadbeef",
            chunk_count=42,
            build_time_utc="2026-06-12T09:00:00+00:00",
        )
        d = meta.to_dict()
        assert d["embedding_provider"] == "ollama"
        assert d["embedding_model"] == "nomic-embed-text"
        assert d["chunk_size"] == 600
        assert d["chunk_overlap"] == 100
        assert d["min_chunk_chars"] == 80
        assert d["corpus_hash"] == "deadbeef"
        assert d["chunk_count"] == 42
        assert d["build_time_utc"] == "2026-06-12T09:00:00+00:00"


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_returns_index_metadata(self, tmp_path: Path) -> None:
        meta = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx")
        assert isinstance(meta, IndexMetadata)

    def test_metadata_records_embedding_model(self, tmp_path: Path) -> None:
        meta = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx")
        assert meta.embedding_provider == "mock"
        assert meta.embedding_model == "mock-embed"

    def test_metadata_records_chunking_settings(self, tmp_path: Path) -> None:
        meta = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx")
        assert meta.chunk_size == CHUNKING.chunk_size
        assert meta.chunk_overlap == CHUNKING.chunk_overlap
        assert meta.min_chunk_chars == CHUNKING.min_chunk_chars

    def test_metadata_records_chunk_count(self, tmp_path: Path) -> None:
        meta = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx")
        assert meta.chunk_count == len(CHUNKS)

    def test_metadata_records_build_time_utc(self, tmp_path: Path) -> None:
        meta = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx")
        # Must be parseable as ISO 8601
        dt = datetime.fromisoformat(meta.build_time_utc)
        assert dt.tzinfo is not None

    def test_corpus_hash_is_deterministic(self, tmp_path: Path) -> None:
        meta1 = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx1")
        meta2 = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx2")
        assert meta1.corpus_hash == meta2.corpus_hash

    def test_corpus_hash_differs_for_different_chunks(self, tmp_path: Path) -> None:
        other_chunks = [make_chunk("x001", "Completely different text.")]
        meta1 = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, tmp_path / "idx1")
        meta2 = build_index(other_chunks, make_client(), RETRIEVAL_CFG, tmp_path / "idx2")
        assert meta1.corpus_hash != meta2.corpus_hash

    def test_writes_index_metadata_json(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        assert (index_dir / "index_metadata.json").is_file()

    def test_writes_faiss_index_file(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        assert (index_dir / "index.faiss").is_file()

    def test_index_metadata_json_is_valid_json(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        content = json.loads((index_dir / "index_metadata.json").read_text())
        assert "embedding_model" in content
        assert "corpus_hash" in content
        assert "build_time_utc" in content

    def test_raises_on_empty_chunk_list(self, tmp_path: Path) -> None:
        with pytest.raises(IndexBuildError, match="empty"):
            build_index([], make_client(), RETRIEVAL_CFG, tmp_path / "idx")

    def test_embeds_all_chunks(self, tmp_path: Path) -> None:
        client = make_client()
        build_index(CHUNKS, client, RETRIEVAL_CFG, tmp_path / "idx")
        total_embedded = sum(len(call) for call in client.calls)
        assert total_embedded == len(CHUNKS)


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------


class TestLoadIndex:
    def test_load_returns_store_and_metadata(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        store, meta = load_index(index_dir)
        assert store is not None
        assert isinstance(meta, IndexMetadata)

    def test_loaded_store_can_search(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        store, _ = load_index(index_dir)
        client = make_client()
        query_vec = client.embed_texts(["Revenue"])[0]
        results = store.search(query_vec, top_k=1)
        assert len(results) == 1

    def test_loaded_metadata_matches_built_metadata(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        built_meta = build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        _, loaded_meta = load_index(index_dir)
        assert loaded_meta.corpus_hash == built_meta.corpus_hash
        assert loaded_meta.chunk_count == built_meta.chunk_count
        assert loaded_meta.embedding_model == built_meta.embedding_model

    def test_load_raises_if_directory_missing(self, tmp_path: Path) -> None:
        with pytest.raises(IndexBuildError, match="not found"):
            load_index(tmp_path / "nonexistent")

    def test_load_raises_if_metadata_missing(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        (index_dir / "index_metadata.json").unlink()
        with pytest.raises(IndexBuildError, match="metadata"):
            load_index(index_dir)

    def test_loaded_store_chunk_count_matches(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        build_index(CHUNKS, make_client(), RETRIEVAL_CFG, index_dir)
        store, meta = load_index(index_dir)
        assert store.count == meta.chunk_count
