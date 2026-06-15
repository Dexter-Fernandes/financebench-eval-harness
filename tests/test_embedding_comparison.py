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
from financebench_eval_harness.embedding_comparison_config import (
    EmbeddingComparisonConfig,
    EmbeddingComparisonConfigError,
    EmbeddingModelSpec,
    load_embedding_comparison_config,
)
from financebench_eval_harness.embedding_comparison import (
    EmbeddingProviderUnavailableError,
    build_client_for_model_spec,
    run_embedding_comparison,
)


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


# ---------------------------------------------------------------------------
# Helpers for comparison runner tests
# ---------------------------------------------------------------------------


def _write_chunks_jsonl(path: Path, chunks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def _write_questions_jsonl(path: Path, questions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")


def _make_minimal_chunks(n: int = 3) -> list[dict]:
    return [
        {
            "chunk_id": f"doc1_p1_c{i}",
            "doc_id": "doc1",
            "doc_name": "AAPL_2022_10K.pdf",
            "page_num": 1,
            "text": f"Revenue was {i * 1000} million dollars in fiscal year 2022.",
            "start_char": i * 100,
            "end_char": i * 100 + 50,
        }
        for i in range(n)
    ]


def _make_minimal_questions(n: int = 2) -> list[dict]:
    return [
        {
            "question_id": f"q{i}",
            "question": f"What was the revenue in year {i}?",
            "answer": f"{i * 1000} million",
            "doc_name": "AAPL_2022_10K.pdf",
            "evidence": [
                {
                    "doc_name": "AAPL_2022_10K.pdf",
                    "gold_page_num": 1,
                    "evidence_text": f"Revenue was {i * 1000} million dollars in fiscal year 2022.",
                }
            ],
        }
        for i in range(n)
    ]


def _make_comparison_config(
    tmp_path: Path,
    *,
    specs: list[EmbeddingModelSpec] | None = None,
    fail_fast: bool = False,
) -> EmbeddingComparisonConfig:
    chunks_path = tmp_path / "chunks" / "chunks.jsonl"
    questions_path = tmp_path / "questions" / "examples.jsonl"
    _write_chunks_jsonl(chunks_path, _make_minimal_chunks())
    _write_questions_jsonl(questions_path, _make_minimal_questions())

    if specs is None:
        specs = [
            EmbeddingModelSpec(name="mock-a", provider="mock", category="open_source"),
            EmbeddingModelSpec(name="mock-b", provider="mock", category="cheap_api"),
        ]

    from financebench_eval_harness.chunking import ChunkingConfig
    from financebench_eval_harness.embedding_comparison_config import ComparisonRetrievalSettings

    return EmbeddingComparisonConfig(
        embedding_models=tuple(specs),
        retrieval=ComparisonRetrievalSettings(
            chunks_path=chunks_path,
            questions_path=questions_path,
            top_k=3,
            evidence_overlap_threshold=0.5,
            chunking=ChunkingConfig(chunk_size=800, chunk_overlap=150),
        ),
        run_id="test_run",
        runs_dir=tmp_path / "runs",
        reports_dir=tmp_path / "reports",
        index_base_dir=tmp_path / "indexes",
        cache_dir=tmp_path / "cache",
        fail_fast=fail_fast,
    )


# ---------------------------------------------------------------------------
# M6.2 EmbeddingModelSpec
# ---------------------------------------------------------------------------


class TestEmbeddingModelSpec:
    def test_slug_replaces_slashes(self) -> None:
        spec = EmbeddingModelSpec(name="BAAI/bge-m3", provider="sentence_transformers", category="open_source")
        assert spec.slug == "BAAI__bge-m3"

    def test_slug_no_slashes(self) -> None:
        spec = EmbeddingModelSpec(name="text-embedding-3-small", provider="openai", category="cheap_api")
        assert spec.slug == "text-embedding-3-small"

    def test_to_embedding_config_maps_fields(self) -> None:
        spec = EmbeddingModelSpec(
            name="text-embedding-3-small",
            provider="openai",
            category="cheap_api",
            dimensions=1536,
            normalize=False,
            batch_size=512,
        )
        cfg = spec.to_embedding_config()
        assert cfg.provider == "openai"
        assert cfg.model_name == "text-embedding-3-small"
        assert cfg.dimensions == 1536
        assert cfg.normalize is False
        assert cfg.batch_size == 512
        assert cfg.category == "cheap_api"

    def test_default_values(self) -> None:
        spec = EmbeddingModelSpec(name="m", provider="mock", category="open_source")
        assert spec.normalize is True
        assert spec.batch_size == 32
        assert spec.dimensions is None


# ---------------------------------------------------------------------------
# M6.2 load_embedding_comparison_config
# ---------------------------------------------------------------------------


class TestLoadEmbeddingComparisonConfig:
    def _write_config(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "embedding_comparison.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        chunks_path.touch()
        questions_path = tmp_path / "examples.jsonl"
        questions_path.touch()
        cfg_path = self._write_config(tmp_path, f"""
comparison:
  runs_dir: {tmp_path}/runs
  reports_dir: {tmp_path}/reports
  index_base_dir: {tmp_path}/indexes
  cache_dir: {tmp_path}/cache
  retrieval:
    chunks_path: {chunks_path}
    questions_path: {questions_path}
    top_k: 10
    evidence_overlap_threshold: 0.5
    chunking:
      strategy: recursive_text
      chunk_size: 800
      chunk_overlap: 150
      min_chunk_chars: 100
  embedding_models:
    - name: mock-model
      provider: mock
      category: open_source
""")
        config = load_embedding_comparison_config(cfg_path)
        assert len(config.embedding_models) == 1
        assert config.embedding_models[0].name == "mock-model"
        assert config.retrieval.top_k == 10

    def test_multiple_models_load_correctly(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        chunks_path.touch()
        questions_path = tmp_path / "examples.jsonl"
        questions_path.touch()
        cfg_path = self._write_config(tmp_path, f"""
comparison:
  runs_dir: {tmp_path}/runs
  reports_dir: {tmp_path}/reports
  index_base_dir: {tmp_path}/indexes
  cache_dir: {tmp_path}/cache
  retrieval:
    chunks_path: {chunks_path}
    questions_path: {questions_path}
    top_k: 10
    evidence_overlap_threshold: 0.5
    chunking:
      chunk_size: 800
      chunk_overlap: 150
  embedding_models:
    - name: model-a
      provider: mock
      category: cheap_api
    - name: model-b
      provider: mock
      category: quality_api
    - name: model-c
      provider: mock
      category: open_source
    - name: model-d
      provider: mock
      category: finance_specialized
""")
        config = load_embedding_comparison_config(cfg_path)
        assert len(config.embedding_models) == 4
        names = [s.name for s in config.embedding_models]
        assert "model-a" in names
        assert "model-d" in names

    def test_missing_embedding_models_raises(self, tmp_path: Path) -> None:
        chunks_path = tmp_path / "chunks.jsonl"
        chunks_path.touch()
        questions_path = tmp_path / "examples.jsonl"
        questions_path.touch()
        cfg_path = self._write_config(tmp_path, f"""
comparison:
  runs_dir: {tmp_path}/runs
  reports_dir: {tmp_path}/reports
  index_base_dir: {tmp_path}/indexes
  cache_dir: {tmp_path}/cache
  retrieval:
    chunks_path: {chunks_path}
    questions_path: {questions_path}
    top_k: 10
    evidence_overlap_threshold: 0.5
    chunking:
      chunk_size: 800
      chunk_overlap: 150
""")
        with pytest.raises(EmbeddingComparisonConfigError):
            load_embedding_comparison_config(cfg_path)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EmbeddingComparisonConfigError):
            load_embedding_comparison_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# M6.6 build_client_for_model_spec
# ---------------------------------------------------------------------------


class TestBuildClientForModelSpec:
    def test_mock_provider_returns_mock_client(self) -> None:
        spec = EmbeddingModelSpec(name="mock-model", provider="mock", category="open_source")
        client = build_client_for_model_spec(spec)
        assert isinstance(client, MockEmbeddingClient)

    def test_mock_client_uses_spec_batch_size(self) -> None:
        spec = EmbeddingModelSpec(name="mock-model", provider="mock", category="open_source", batch_size=16)
        client = build_client_for_model_spec(spec)
        assert client.config.batch_size == 16

    def test_unavailable_provider_raises(self) -> None:
        spec = EmbeddingModelSpec(name="m", provider="nonexistent_provider_xyz", category="open_source")
        with pytest.raises(EmbeddingProviderUnavailableError):
            build_client_for_model_spec(spec)


# ---------------------------------------------------------------------------
# M6.5 + M6.6 + M6.7 run_embedding_comparison
# ---------------------------------------------------------------------------


class TestRunEmbeddingComparison:
    def test_each_model_gets_separate_run_dir(self, tmp_path: Path) -> None:
        config = _make_comparison_config(tmp_path)
        result = run_embedding_comparison(config)
        model_runs_dir = result.run_dir / "model_runs"
        assert (model_runs_dir / "mock-a").is_dir()
        assert (model_runs_dir / "mock-b").is_dir()

    def test_retrieval_results_written_per_model(self, tmp_path: Path) -> None:
        config = _make_comparison_config(tmp_path)
        result = run_embedding_comparison(config)
        model_runs_dir = result.run_dir / "model_runs"
        assert (model_runs_dir / "mock-a" / "retrieval_results.jsonl").is_file()
        assert (model_runs_dir / "mock-b" / "retrieval_results.jsonl").is_file()

    def test_each_model_gets_separate_index_dir(self, tmp_path: Path) -> None:
        config = _make_comparison_config(tmp_path)
        run_embedding_comparison(config)
        assert (config.index_base_dir / "mock-a").is_dir()
        assert (config.index_base_dir / "mock-b").is_dir()

    def test_config_snapshot_written(self, tmp_path: Path) -> None:
        config = _make_comparison_config(tmp_path)
        result = run_embedding_comparison(config)
        assert (result.run_dir / "config.yaml").is_file()

    def test_failed_model_does_not_kill_comparison(self, tmp_path: Path) -> None:
        specs = [
            EmbeddingModelSpec(name="mock-ok", provider="mock", category="open_source"),
            EmbeddingModelSpec(name="bad-model", provider="nonexistent_provider_xyz", category="cheap_api"),
        ]
        config = _make_comparison_config(tmp_path, specs=specs, fail_fast=False)
        result = run_embedding_comparison(config)
        assert result.succeeded_count == 1
        assert result.failed_count == 1
        assert "bad-model" in result.failed_models

    def test_corpus_hash_recorded_in_config_snapshot(self, tmp_path: Path) -> None:
        config = _make_comparison_config(tmp_path)
        result = run_embedding_comparison(config)
        import yaml
        snapshot = yaml.safe_load((result.run_dir / "config.yaml").read_text())
        assert "corpus_hash" in snapshot
