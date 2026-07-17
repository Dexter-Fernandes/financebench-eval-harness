"""Tests for lexical_index.py — M9.3 BM25 lexical retrieval."""

from __future__ import annotations

import pytest

from financebench_eval_harness.lexical_index import BM25Config, BM25Index

# ---------------------------------------------------------------------------
# Toy corpus — hand-computed against Okapi BM25 with default k1=1.5, b=0.75
# ---------------------------------------------------------------------------

_TOY_CHUNKS = [
    {"chunk_id": "c1", "text": "the cat sat on the mat"},
    {"chunk_id": "c2", "text": "the dog sat on the log"},
    {"chunk_id": "c3", "text": "cats and dogs are pets"},
    {"chunk_id": "c4", "text": "the mat was on the floor"},
]


def test_search_ranks_by_hand_computed_bm25_score() -> None:
    index = BM25Index.from_chunks(_TOY_CHUNKS)
    results = index.search("cat mat", top_k=4)

    ids = [chunk_id for chunk_id, _ in results]
    assert ids == ["c1", "c4", "c2", "c3"]

    scores = dict(results)
    assert scores["c1"] == pytest.approx(1.8607146973294362, rel=1e-6)
    assert scores["c4"] == pytest.approx(0.6798458487368334, rel=1e-6)
    assert scores["c2"] == pytest.approx(0.0)
    assert scores["c3"] == pytest.approx(0.0)


def test_search_respects_top_k() -> None:
    index = BM25Index.from_chunks(_TOY_CHUNKS)
    results = index.search("cat mat", top_k=2)
    assert [chunk_id for chunk_id, _ in results] == ["c1", "c4"]


def test_search_empty_query_returns_empty() -> None:
    index = BM25Index.from_chunks(_TOY_CHUNKS)
    assert index.search("", top_k=4) == []


def test_search_term_present_in_all_documents_still_scores_non_negative() -> None:
    docs = [
        {"chunk_id": "d1", "text": "alpha beta"},
        {"chunk_id": "d2", "text": "alpha gamma"},
    ]
    index = BM25Index.from_chunks(docs)
    results = index.search("alpha", top_k=2)
    assert all(score >= 0.0 for _, score in results)


def test_search_deterministic_tie_break_by_chunk_id() -> None:
    # c2 and c3 both score 0.0 for this query — tie-break must be by chunk_id ascending.
    index = BM25Index.from_chunks(_TOY_CHUNKS)
    results = index.search("cat mat", top_k=4)
    tied = [chunk_id for chunk_id, score in results if score == pytest.approx(0.0)]
    assert tied == sorted(tied)


def test_from_chunks_accepts_custom_config() -> None:
    index = BM25Index.from_chunks(_TOY_CHUNKS, config=BM25Config(k1=1.2, b=0.5))
    results = index.search("cat mat", top_k=1)
    assert results[0][0] == "c1"
