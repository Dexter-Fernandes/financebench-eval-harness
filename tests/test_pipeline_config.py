from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from financebench_eval_harness.pipeline_config import (
    PipelineConfig,
    PipelineConfigError,
    load_pipeline_config,
)
from financebench_eval_harness.chunking import ChunkingConfig
from financebench_eval_harness.embedding import EmbeddingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
    retrieval:
      pages_path: data/processed/financebench/pages.jsonl
      chunks_path: data/processed/financebench/chunks.jsonl
      index_dir: data/indexes/financebench
      questions_path: data/processed/financebench/examples.jsonl
      runs_dir: runs
      top_k: 5
      chunking:
        strategy: recursive_text
        chunk_size: 800
        chunk_overlap: 150
        min_chunk_chars: 100
      embedding:
        provider: ollama
        model_name: nomic-embed-text
        base_url: http://localhost:11434
        timeout_seconds: 30.0
        batch_size: 256
""")

_YAML_WITHOUT_TOP_K = textwrap.dedent("""\
    retrieval:
      pages_path: data/pages.jsonl
      chunks_path: data/chunks.jsonl
      index_dir: data/idx
      questions_path: data/q.jsonl
      runs_dir: runs
      chunking:
        strategy: recursive_text
        chunk_size: 400
        chunk_overlap: 50
      embedding:
        provider: mock
        model_name: mock-embed
""")


def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "retrieval.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# PipelineConfig dataclass
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def _make(self) -> PipelineConfig:
        return PipelineConfig(
            pages_path=Path("data/pages.jsonl"),
            chunks_path=Path("data/chunks.jsonl"),
            index_dir=Path("data/idx"),
            questions_path=Path("data/q.jsonl"),
            runs_dir=Path("runs"),
            top_k=5,
            chunking=ChunkingConfig(chunk_size=800, chunk_overlap=150),
            embedding=EmbeddingConfig(provider="mock", model_name="mock-embed"),
        )

    def test_stores_all_fields(self) -> None:
        cfg = self._make()
        assert cfg.pages_path == Path("data/pages.jsonl")
        assert cfg.chunks_path == Path("data/chunks.jsonl")
        assert cfg.index_dir == Path("data/idx")
        assert cfg.questions_path == Path("data/q.jsonl")
        assert cfg.runs_dir == Path("runs")
        assert cfg.top_k == 5
        assert isinstance(cfg.chunking, ChunkingConfig)
        assert isinstance(cfg.embedding, EmbeddingConfig)

    def test_is_immutable(self) -> None:
        cfg = self._make()
        with pytest.raises(Exception):
            cfg.top_k = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_pipeline_config
# ---------------------------------------------------------------------------


class TestLoadPipelineConfig:
    def test_returns_pipeline_config(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        result = load_pipeline_config(p)
        assert isinstance(result, PipelineConfig)

    def test_paths_are_path_objects(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert isinstance(cfg.pages_path, Path)
        assert isinstance(cfg.chunks_path, Path)
        assert isinstance(cfg.index_dir, Path)
        assert isinstance(cfg.questions_path, Path)
        assert isinstance(cfg.runs_dir, Path)

    def test_pages_path_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.pages_path == Path("data/processed/financebench/pages.jsonl")

    def test_chunks_path_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.chunks_path == Path("data/processed/financebench/chunks.jsonl")

    def test_index_dir_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.index_dir == Path("data/indexes/financebench")

    def test_top_k_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.top_k == 5

    def test_top_k_defaults_to_5_when_omitted(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _YAML_WITHOUT_TOP_K)
        cfg = load_pipeline_config(p)
        assert cfg.top_k == 5

    def test_chunking_chunk_size_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.chunking.chunk_size == 800

    def test_chunking_strategy_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.chunking.strategy == "recursive_text"

    def test_embedding_provider_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.embedding.provider == "ollama"

    def test_embedding_model_name_round_trips(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, _VALID_YAML)
        cfg = load_pipeline_config(p)
        assert cfg.embedding.model_name == "nomic-embed-text"

    def test_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(PipelineConfigError, match="not found"):
            load_pipeline_config(tmp_path / "nonexistent.yaml")

    def test_raises_when_retrieval_key_absent(self, tmp_path: Path) -> None:
        p = _write_config(tmp_path, "other_key: {}\n")
        with pytest.raises(PipelineConfigError, match="retrieval"):
            load_pipeline_config(p)

    def test_raises_when_required_path_missing(self, tmp_path: Path) -> None:
        yaml_missing_chunks = textwrap.dedent("""\
            retrieval:
              pages_path: data/pages.jsonl
              index_dir: data/idx
              questions_path: data/q.jsonl
              runs_dir: runs
              chunking:
                chunk_size: 400
                chunk_overlap: 50
              embedding:
                provider: mock
                model_name: mock-embed
        """)
        p = _write_config(tmp_path, yaml_missing_chunks)
        with pytest.raises(PipelineConfigError, match="chunks_path"):
            load_pipeline_config(p)
