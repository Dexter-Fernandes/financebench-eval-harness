"""Tests for fusion.py — M9.4 reciprocal rank fusion."""

from __future__ import annotations

import pytest

from financebench_eval_harness.fusion import reciprocal_rank_fusion

_DENSE = ["c1", "c2", "c3"]
_LEXICAL = ["c3", "c1", "c4"]


def test_fused_order_matches_hand_computed_rrf_scores() -> None:
    results = reciprocal_rank_fusion([_DENSE, _LEXICAL])
    ids = [chunk_id for chunk_id, _ in results]
    # c3 (found by both, rank 3 dense + rank 1 lexical) edges out c2
    # (dense-only rank 2) because it also shows up in the lexical ranking.
    assert ids == ["c1", "c3", "c2", "c4"]

    scores = dict(results)
    assert scores["c1"] == pytest.approx(0.03252247488101534)
    assert scores["c3"] == pytest.approx(0.032266458495966696)
    assert scores["c2"] == pytest.approx(0.016129032258064516)
    assert scores["c4"] == pytest.approx(0.015873015873015872)


def test_weighted_fusion_changes_order() -> None:
    # Heavily weighting the lexical ranking should push c4 (lexical rank 3,
    # absent from dense) above c2 (dense rank 2, absent from lexical).
    unweighted = reciprocal_rank_fusion([_DENSE, _LEXICAL])
    weighted = reciprocal_rank_fusion([_DENSE, _LEXICAL], weights=[0.1, 10.0])

    unweighted_ids = [chunk_id for chunk_id, _ in unweighted]
    weighted_ids = [chunk_id for chunk_id, _ in weighted]

    assert unweighted_ids.index("c2") < unweighted_ids.index("c4")
    assert weighted_ids.index("c4") < weighted_ids.index("c2")


def test_custom_k_changes_scores() -> None:
    results_default = dict(reciprocal_rank_fusion([_DENSE]))
    results_k1 = dict(reciprocal_rank_fusion([_DENSE], k=1))
    assert results_default["c1"] != results_k1["c1"]
    assert results_k1["c1"] == pytest.approx(1.0 / (1 + 1))


def test_deterministic_tie_break_by_chunk_id() -> None:
    # Both chunks appear only in one ranking each, at the same rank -> tie.
    results = reciprocal_rank_fusion([["b"], ["a"]])
    ids = [chunk_id for chunk_id, _ in results]
    assert ids == ["a", "b"]


def test_empty_rankings_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
