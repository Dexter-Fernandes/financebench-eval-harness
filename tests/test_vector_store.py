from __future__ import annotations

from pathlib import Path

import pytest

from financebench_eval_harness.retrieval_types import Chunk, RetrievalResult
from financebench_eval_harness.vector_store import (
    FaissVectorStore,
    VectorStore,
    VectorStoreError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(chunk_id: str, doc_id: str = "DOC_A", page_num: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=f"{doc_id}.pdf",
        page_num=page_num,
        text=f"text for {chunk_id}",
        start_char=0,
        end_char=10,
    )


# Simple 4-d unit vectors pointing in different directions — nearest neighbour
# is unambiguous with L2 distance.
VECS = {
    "north": [0.0, 1.0, 0.0, 0.0],
    "south": [0.0, -1.0, 0.0, 0.0],
    "east":  [1.0, 0.0, 0.0, 0.0],
    "west":  [-1.0, 0.0, 0.0, 0.0],
}
DIM = 4


# ---------------------------------------------------------------------------
# VectorStore Protocol
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    def test_faiss_satisfies_protocol(self) -> None:
        store: VectorStore = FaissVectorStore(dim=DIM)
        assert hasattr(store, "add")
        assert hasattr(store, "search")


# ---------------------------------------------------------------------------
# FaissVectorStore — basic add + search
# ---------------------------------------------------------------------------


class TestFaissVectorStoreBasic:
    def test_search_empty_store_returns_empty_list(self) -> None:
        store = FaissVectorStore(dim=DIM)
        results = store.search(VECS["north"], top_k=5)
        assert results == []

    def test_add_single_chunk_then_search_finds_it(self) -> None:
        store = FaissVectorStore(dim=DIM)
        chunk = make_chunk("c001")
        store.add([chunk], [VECS["north"]])
        results = store.search(VECS["north"], top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == "c001"

    def test_search_returns_nearest_chunk(self) -> None:
        store = FaissVectorStore(dim=DIM)
        chunks = [make_chunk(k) for k in VECS]
        embeddings = list(VECS.values())
        store.add(chunks, embeddings)
        results = store.search(VECS["east"], top_k=1)
        assert results[0].chunk_id == "east"

    def test_top_k_limits_result_count(self) -> None:
        store = FaissVectorStore(dim=DIM)
        chunks = [make_chunk(k) for k in VECS]
        embeddings = list(VECS.values())
        store.add(chunks, embeddings)
        results = store.search(VECS["north"], top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_index_returns_all(self) -> None:
        store = FaissVectorStore(dim=DIM)
        store.add([make_chunk("c001")], [VECS["north"]])
        results = store.search(VECS["north"], top_k=100)
        assert len(results) == 1

    def test_results_are_retrieval_result_objects(self) -> None:
        store = FaissVectorStore(dim=DIM)
        store.add([make_chunk("c001")], [VECS["north"]])
        results = store.search(VECS["north"], top_k=1)
        assert all(isinstance(r, RetrievalResult) for r in results)


# ---------------------------------------------------------------------------
# FaissVectorStore — rank and score ordering
# ---------------------------------------------------------------------------


class TestFaissVectorStoreRanking:
    def test_rank_one_is_assigned_to_best_result(self) -> None:
        store = FaissVectorStore(dim=DIM)
        chunks = [make_chunk(k) for k in VECS]
        store.add(chunks, list(VECS.values()))
        results = store.search(VECS["north"], top_k=len(VECS))
        assert results[0].rank == 1

    def test_ranks_are_sequential_from_one(self) -> None:
        store = FaissVectorStore(dim=DIM)
        chunks = [make_chunk(k) for k in VECS]
        store.add(chunks, list(VECS.values()))
        results = store.search(VECS["north"], top_k=len(VECS))
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_decrease_with_rank(self) -> None:
        store = FaissVectorStore(dim=DIM)
        chunks = [make_chunk(k) for k in VECS]
        store.add(chunks, list(VECS.values()))
        results = store.search(VECS["north"], top_k=len(VECS))
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_match_has_highest_score(self) -> None:
        store = FaissVectorStore(dim=DIM)
        chunks = [make_chunk(k) for k in VECS]
        store.add(chunks, list(VECS.values()))
        results = store.search(VECS["north"], top_k=len(VECS))
        assert results[0].chunk_id == "north"
        assert results[0].score > results[1].score


# ---------------------------------------------------------------------------
# FaissVectorStore — metadata preservation
# ---------------------------------------------------------------------------


class TestFaissVectorStoreMetadata:
    def test_chunk_id_preserved_in_result(self) -> None:
        store = FaissVectorStore(dim=DIM)
        store.add([make_chunk("my_chunk_42")], [VECS["north"]])
        results = store.search(VECS["north"], top_k=1)
        assert results[0].chunk_id == "my_chunk_42"

    def test_question_id_is_empty_string_by_default(self) -> None:
        store = FaissVectorStore(dim=DIM)
        store.add([make_chunk("c001")], [VECS["north"]])
        results = store.search(VECS["north"], top_k=1)
        assert results[0].question_id == ""

    def test_add_multiple_batches(self) -> None:
        store = FaissVectorStore(dim=DIM)
        store.add([make_chunk("c001")], [VECS["north"]])
        store.add([make_chunk("c002")], [VECS["east"]])
        results = store.search(VECS["east"], top_k=1)
        assert results[0].chunk_id == "c002"

    def test_count_reflects_indexed_chunks(self) -> None:
        store = FaissVectorStore(dim=DIM)
        assert store.count == 0
        store.add([make_chunk("c001"), make_chunk("c002")], [VECS["north"], VECS["east"]])
        assert store.count == 2


# ---------------------------------------------------------------------------
# FaissVectorStore — error handling
# ---------------------------------------------------------------------------


class TestFaissVectorStoreErrors:
    def test_mismatched_chunk_and_embedding_counts_raise(self) -> None:
        store = FaissVectorStore(dim=DIM)
        with pytest.raises(VectorStoreError, match="mismatch"):
            store.add([make_chunk("c001")], [VECS["north"], VECS["east"]])

    def test_wrong_dimension_embedding_raises(self) -> None:
        store = FaissVectorStore(dim=DIM)
        with pytest.raises(VectorStoreError):
            store.add([make_chunk("c001")], [[0.1, 0.2]])  # wrong dim


# ---------------------------------------------------------------------------
# FaissVectorStore — configurable index path (save / load)
# ---------------------------------------------------------------------------


class TestFaissVectorStorePersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        store = FaissVectorStore(dim=DIM)
        chunks = [make_chunk(k) for k in VECS]
        store.add(chunks, list(VECS.values()))
        store.save(tmp_path / "index")

        loaded = FaissVectorStore.load(tmp_path / "index")
        results = loaded.search(VECS["east"], top_k=1)
        assert results[0].chunk_id == "east"

    def test_save_creates_index_files(self, tmp_path: Path) -> None:
        store = FaissVectorStore(dim=DIM)
        store.add([make_chunk("c001")], [VECS["north"]])
        index_path = tmp_path / "index"
        store.save(index_path)
        # At minimum the index directory / files must exist
        assert any(tmp_path.iterdir())

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VectorStoreError, match="not found"):
            FaissVectorStore.load(tmp_path / "no_such_index")

    def test_loaded_store_has_correct_count(self, tmp_path: Path) -> None:
        store = FaissVectorStore(dim=DIM)
        store.add([make_chunk("c001"), make_chunk("c002")], [VECS["north"], VECS["east"]])
        store.save(tmp_path / "index")

        loaded = FaissVectorStore.load(tmp_path / "index")
        assert loaded.count == 2
