"""Tests for hybrid_retriever.py — M9.7 retriever compositions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.hybrid_retriever import (
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    RerankedRetriever,
    run_retrieval_with_retriever,
)
from financebench_eval_harness.index_builder import IndexMetadata
from financebench_eval_harness.reranker import MockRerankerClient, RerankerConfig
from financebench_eval_harness.retrieval_types import Chunk
from financebench_eval_harness.retriever import Question, RetrievedChunk


def _rc(chunk_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank, chunk_id=chunk_id, doc_name="d.pdf", page_num=1,
        score=1.0 / rank, text=f"text for {chunk_id}",
    )


class _FakeRetriever:
    """Stub Retriever returning a pre-baked ranking for a given query."""

    def __init__(self, chunks_by_query: dict[str, list[RetrievedChunk]]) -> None:
        self._chunks_by_query = chunks_by_query

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._chunks_by_query[query][:top_k]


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="DOC_A", doc_name="DOC_A.pdf", page_num=1,
        text=f"Text for {chunk_id}.", start_char=0, end_char=10,
    )


# ---------------------------------------------------------------------------
# HybridRetriever — surfaces chunks found by only one system
# ---------------------------------------------------------------------------


def test_hybrid_retriever_surfaces_chunks_found_by_only_one_system() -> None:
    # s1 is found by both; d1 is dense-only; l1 is lexical-only.
    dense = _FakeRetriever({"q": [_rc("s1", 1), _rc("d1", 2), _rc("d2", 3)]})
    lexical = _FakeRetriever({"q": [_rc("l1", 1), _rc("s1", 2), _rc("l2", 3)]})
    chunk_lookup = {cid: _chunk(cid) for cid in ("s1", "d1", "d2", "l1", "l2")}

    retriever = HybridRetriever(dense, lexical, chunk_lookup, candidate_k=3, fusion_k=60)
    results = retriever.retrieve("q", top_k=3)

    ids = [r.chunk_id for r in results]
    assert ids == ["s1", "l1", "d1"]
    assert "d1" in ids  # dense-only chunk surfaced
    assert "l1" in ids  # lexical-only chunk surfaced


def test_hybrid_retriever_records_provenance_ranks() -> None:
    dense = _FakeRetriever({"q": [_rc("s1", 1), _rc("d1", 2)]})
    lexical = _FakeRetriever({"q": [_rc("l1", 1), _rc("s1", 2)]})
    chunk_lookup = {cid: _chunk(cid) for cid in ("s1", "d1", "l1")}

    retriever = HybridRetriever(dense, lexical, chunk_lookup, candidate_k=2, fusion_k=60)
    results = {r.chunk_id: r for r in retriever.retrieve("q", top_k=3)}

    assert results["s1"].dense_rank == 1
    assert results["s1"].lexical_rank == 2
    assert results["d1"].dense_rank == 2
    assert results["d1"].lexical_rank is None
    assert results["l1"].dense_rank is None
    assert results["l1"].lexical_rank == 1


def test_hybrid_retriever_output_ranks_are_sequential() -> None:
    dense = _FakeRetriever({"q": [_rc("a", 1), _rc("b", 2)]})
    lexical = _FakeRetriever({"q": [_rc("c", 1)]})
    chunk_lookup = {cid: _chunk(cid) for cid in ("a", "b", "c")}

    retriever = HybridRetriever(dense, lexical, chunk_lookup, candidate_k=2)
    results = retriever.retrieve("q", top_k=3)
    assert [r.rank for r in results] == [1, 2, 3]


# ---------------------------------------------------------------------------
# RerankedRetriever — reorders and truncates to top_n
# ---------------------------------------------------------------------------


def test_reranked_retriever_reorders_by_mock_reranker_score_and_truncates() -> None:
    inner = _FakeRetriever({"q": [_rc("a", 1), _rc("b", 2), _rc("c", 3)]})
    reranker_config = RerankerConfig(provider="mock", model_name="mock-reranker")
    reranker = MockRerankerClient(reranker_config, scores=[0.1, 0.9, 0.5])

    retriever = RerankedRetriever(inner, reranker, candidate_k=3, top_n=2)
    results = retriever.retrieve("q", top_k=3)

    assert [r.chunk_id for r in results] == ["b", "c"]
    assert [r.rank for r in results] == [1, 2]
    assert results[0].rerank_score == pytest.approx(0.9)
    assert results[1].rerank_score == pytest.approx(0.5)


def test_reranked_retriever_without_top_n_uses_requested_top_k() -> None:
    inner = _FakeRetriever({"q": [_rc("a", 1), _rc("b", 2), _rc("c", 3)]})
    reranker_config = RerankerConfig(provider="mock", model_name="mock-reranker")
    reranker = MockRerankerClient(reranker_config, scores=[0.3, 0.1, 0.8])

    retriever = RerankedRetriever(inner, reranker, candidate_k=3)
    results = retriever.retrieve("q", top_k=2)

    assert [r.chunk_id for r in results] == ["c", "a"]


def test_reranked_retriever_empty_candidates_returns_empty() -> None:
    inner = _FakeRetriever({"q": []})
    reranker_config = RerankerConfig(provider="mock", model_name="mock-reranker")
    reranker = MockRerankerClient(reranker_config)

    retriever = RerankedRetriever(inner, reranker, candidate_k=5)
    assert retriever.retrieve("q", top_k=3) == []


# ---------------------------------------------------------------------------
# DenseRetriever / LexicalRetriever — thin wrappers, dense-equivalence check
# ---------------------------------------------------------------------------


def test_dense_retriever_matches_existing_retrieve_for_questions_behavior() -> None:
    from financebench_eval_harness.embedding import EmbeddingConfig, MockEmbeddingClient
    from financebench_eval_harness.vector_store import FaissVectorStore

    dim = 4
    store = FaissVectorStore(dim=dim)
    chunks = [_chunk("c1"), _chunk("c2")]
    store.add(chunks, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    client = MockEmbeddingClient(EmbeddingConfig(provider="mock", model_name="m"), embedding_dim=dim)

    retriever = DenseRetriever(store, client)
    results = retriever.retrieve("some query", top_k=2)

    assert len(results) == 2
    assert [r.rank for r in results] == [1, 2]
    assert all(isinstance(r.chunk_id, str) for r in results)


def test_lexical_retriever_ranks_by_bm25_over_store_chunks() -> None:
    from financebench_eval_harness.vector_store import FaissVectorStore

    store = FaissVectorStore(dim=2)
    chunks = [
        Chunk(chunk_id="c1", doc_id="D", doc_name="D.pdf", page_num=1,
              text="the cat sat on the mat", start_char=0, end_char=10),
        Chunk(chunk_id="c2", doc_id="D", doc_name="D.pdf", page_num=2,
              text="the dog sat on the log", start_char=0, end_char=10),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])

    retriever = LexicalRetriever(store)
    results = retriever.retrieve("cat", top_k=2)

    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1


# ---------------------------------------------------------------------------
# run_retrieval_with_retriever
# ---------------------------------------------------------------------------


SAMPLE_INDEX_META = IndexMetadata(
    embedding_provider="mock",
    embedding_model="mock-embed",
    chunk_size=800,
    chunk_overlap=150,
    min_chunk_chars=0,
    corpus_hash="abc123",
    chunk_count=2,
    build_time_utc="2026-06-11T00:00:00+00:00",
)

QUESTIONS = [Question(question_id="q001", query="q")]


def test_run_retrieval_with_retriever_writes_identical_row_schema(tmp_path: Path) -> None:
    retriever = _FakeRetriever({"q": [_rc("c1", 1), _rc("c2", 2)]})
    output_path = tmp_path / "retrieval_results.jsonl"

    run_retrieval_with_retriever(QUESTIONS, retriever, output_path, top_k=2)

    rows = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
    assert rows[0]["question_id"] == "q001"
    retrieved = rows[0]["retrieved"]
    assert set(retrieved[0].keys()) == {"rank", "chunk_id", "doc_name", "page_num", "score", "text"}


def test_run_retrieval_with_retriever_metadata_records_retrieval_mode(tmp_path: Path) -> None:
    retriever = _FakeRetriever({"q": [_rc("c1", 1)]})
    output_path = tmp_path / "run" / "retrieval_results.jsonl"

    run_retrieval_with_retriever(
        QUESTIONS, retriever, output_path, top_k=1,
        index_metadata=SAMPLE_INDEX_META, retrieval_mode="hybrid_reranked",
        fusion={"k": 60, "candidate_k": 50},
        reranker={"provider": "mock", "model_name": "mock-reranker", "adapter_path": None},
    )

    meta = json.loads((tmp_path / "run" / "retrieval_run_metadata.json").read_text())
    assert meta["retrieval_mode"] == "hybrid_reranked"
    assert meta["fusion"] == {"k": 60, "candidate_k": 50}
    assert meta["reranker"] == {"provider": "mock", "model_name": "mock-reranker", "adapter_path": None}


def test_run_retrieval_with_retriever_provenance_keys_appear_when_present(tmp_path: Path) -> None:
    rc_with_provenance = RetrievedChunk(
        rank=1, chunk_id="c1", doc_name="d.pdf", page_num=1, score=0.5, text="t",
        dense_rank=2, lexical_rank=1,
    )
    retriever = _FakeRetriever({"q": [rc_with_provenance]})
    output_path = tmp_path / "retrieval_results.jsonl"

    run_retrieval_with_retriever(QUESTIONS, retriever, output_path, top_k=1)

    rows = [json.loads(l) for l in output_path.read_text().splitlines() if l.strip()]
    retrieved = rows[0]["retrieved"][0]
    assert retrieved["dense_rank"] == 2
    assert retrieved["lexical_rank"] == 1
