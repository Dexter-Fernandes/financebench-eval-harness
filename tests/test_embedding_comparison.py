"""Tests for M6 embedding model comparison.

All tests use MockEmbeddingClient — no real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from financebench_eval_harness.embedding import EmbeddingConfig, MockEmbeddingClient
from financebench_eval_harness.embedding_cache import EmbeddingCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache(tmp_path: Path, *, provider: str = "mock", model_name: str = "model-a",
                dimensions: int | None = None, normalize: bool = True) -> EmbeddingCache:
    return EmbeddingCache(
        cache_dir=tmp_path / "cache",
        provider=provider,
        model_name=model_name,
        dimensions=dimensions,
        normalize=normalize,
    )


# ---------------------------------------------------------------------------
# M6.4 EmbeddingCache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        assert cache.get("some text") is None

    def test_put_then_get_returns_same_vector(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        vector = [0.1, 0.2, 0.3, 0.4]
        cache.put("hello world", vector)
        result = cache.get("hello world")
        assert result is not None
        assert len(result) == len(vector)
        assert result == pytest.approx(vector)

    def test_cache_key_changes_when_model_changes(self, tmp_path: Path) -> None:
        cache_a = _make_cache(tmp_path, model_name="model-a")
        cache_b = _make_cache(tmp_path, model_name="model-b")
        vector = [1.0, 2.0]
        cache_a.put("same text", vector)
        assert cache_b.get("same text") is None

    def test_cache_key_changes_when_text_changes(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.put("text A", [1.0, 0.0])
        assert cache.get("text B") is None

    def test_cache_key_changes_when_dimensions_change(self, tmp_path: Path) -> None:
        cache_512 = _make_cache(tmp_path, dimensions=512)
        cache_1536 = _make_cache(tmp_path, dimensions=1536)
        cache_512.put("hello", [0.5, 0.6])
        assert cache_1536.get("hello") is None

    def test_cache_key_changes_when_normalize_changes(self, tmp_path: Path) -> None:
        cache_norm = _make_cache(tmp_path, normalize=True)
        cache_raw = _make_cache(tmp_path, normalize=False)
        cache_norm.put("hello", [0.5, 0.6])
        assert cache_raw.get("hello") is None

    def test_get_batch_all_misses(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        results, hits, misses = cache.get_batch(["a", "b", "c"])
        assert results == [None, None, None]
        assert hits == 0
        assert misses == 3

    def test_put_batch_then_get_batch(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        texts = ["x", "y", "z"]
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        cache.put_batch(texts, vectors)
        results, hits, misses = cache.get_batch(texts)
        assert hits == 3
        assert misses == 0
        for r, v in zip(results, vectors):
            assert r == pytest.approx(v)

    def test_get_batch_mixed_hit_miss(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.put("cached", [1.0, 0.0])
        results, hits, misses = cache.get_batch(["cached", "not-cached"])
        assert hits == 1
        assert misses == 1
        assert results[0] == pytest.approx([1.0, 0.0])
        assert results[1] is None

    def test_metadata_tracks_hits_and_misses(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.put("text", [0.1, 0.2])
        cache.get("text")       # hit
        cache.get("missing")    # miss
        meta = cache.load_metadata()
        assert meta["hit_count"] >= 1
        assert meta["miss_count"] >= 1

    def test_cache_persists_across_instances(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache1 = EmbeddingCache(cache_dir, provider="mock", model_name="m", dimensions=None, normalize=True)
        cache1.put("persist me", [9.0, 8.0])
        cache2 = EmbeddingCache(cache_dir, provider="mock", model_name="m", dimensions=None, normalize=True)
        result = cache2.get("persist me")
        assert result == pytest.approx([9.0, 8.0])
