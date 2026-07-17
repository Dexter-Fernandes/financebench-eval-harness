"""Reciprocal rank fusion — M9.4."""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple chunk_id rankings into one, best first.

    score(chunk) = sum over rankings of weight / (k + rank), rank 1-based.
    weights default to 1.0 per ranking. Ties break deterministically by chunk_id.
    """
    effective_weights = weights if weights is not None else [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for weight, ranking in zip(effective_weights, rankings):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


__all__ = ["reciprocal_rank_fusion"]
