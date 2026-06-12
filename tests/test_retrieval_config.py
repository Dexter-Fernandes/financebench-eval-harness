from __future__ import annotations

from pathlib import Path

import pytest

from financebench_eval_harness.retrieval_config import (
    RetrievalConfig,
    RetrievalConfigError,
    load_retrieval_config,
)
from financebench_eval_harness.chunking import ChunkingConfig


# ---------------------------------------------------------------------------
# RetrievalConfig
# ---------------------------------------------------------------------------


class TestRetrievalConfig:
    def test_holds_chunking_config(self) -> None:
        chunking = ChunkingConfig(chunk_size=800, chunk_overlap=150)
        cfg = RetrievalConfig(chunking=chunking)
        assert cfg.chunking is chunking

    def test_is_immutable(self) -> None:
        cfg = RetrievalConfig(chunking=ChunkingConfig(chunk_size=800, chunk_overlap=150))
        with pytest.raises(Exception):
            cfg.chunking = ChunkingConfig(chunk_size=400, chunk_overlap=80)  # type: ignore[misc]

    def test_to_dict_includes_all_chunking_fields(self) -> None:
        chunking = ChunkingConfig(
            chunk_size=800,
            chunk_overlap=150,
            strategy="recursive_text",
            min_chunk_chars=100,
        )
        cfg = RetrievalConfig(chunking=chunking)
        d = cfg.to_dict()
        assert d["chunking"]["chunk_size"] == 800
        assert d["chunking"]["chunk_overlap"] == 150
        assert d["chunking"]["strategy"] == "recursive_text"
        assert d["chunking"]["min_chunk_chars"] == 100

    def test_to_dict_is_plain_types(self) -> None:
        cfg = RetrievalConfig(chunking=ChunkingConfig(chunk_size=400, chunk_overlap=80))
        d = cfg.to_dict()
        # Must be JSON-serialisable plain types, not dataclasses
        import json
        json.dumps(d)  # raises if not serialisable

    def test_to_dict_top_level_key_is_chunking(self) -> None:
        cfg = RetrievalConfig(chunking=ChunkingConfig(chunk_size=800, chunk_overlap=150))
        assert set(cfg.to_dict().keys()) == {"chunking"}


# ---------------------------------------------------------------------------
# load_retrieval_config — happy path
# ---------------------------------------------------------------------------


class TestLoadRetrievalConfig:
    def test_loads_full_config_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(
            "\n".join([
                "retrieval:",
                "  chunking:",
                "    strategy: recursive_text",
                "    chunk_size: 800",
                "    chunk_overlap: 150",
                "    min_chunk_chars: 100",
            ]),
            encoding="utf-8",
        )

        cfg = load_retrieval_config(config_file)

        assert cfg.chunking.strategy == "recursive_text"
        assert cfg.chunking.chunk_size == 800
        assert cfg.chunking.chunk_overlap == 150
        assert cfg.chunking.min_chunk_chars == 100

    def test_loads_minimal_config_with_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(
            "\n".join([
                "retrieval:",
                "  chunking:",
                "    chunk_size: 400",
                "    chunk_overlap: 80",
            ]),
            encoding="utf-8",
        )

        cfg = load_retrieval_config(config_file)

        assert cfg.chunking.chunk_size == 400
        assert cfg.chunking.chunk_overlap == 80
        assert cfg.chunking.strategy == "recursive_text"
        assert cfg.chunking.min_chunk_chars == 0

    def test_roundtrip_to_dict_preserves_loaded_values(self, tmp_path: Path) -> None:
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(
            "\n".join([
                "retrieval:",
                "  chunking:",
                "    strategy: recursive_text",
                "    chunk_size: 600",
                "    chunk_overlap: 120",
                "    min_chunk_chars: 50",
            ]),
            encoding="utf-8",
        )

        cfg = load_retrieval_config(config_file)
        d = cfg.to_dict()

        assert d["chunking"]["chunk_size"] == 600
        assert d["chunking"]["chunk_overlap"] == 120
        assert d["chunking"]["min_chunk_chars"] == 50


# ---------------------------------------------------------------------------
# load_retrieval_config — error cases
# ---------------------------------------------------------------------------


class TestLoadRetrievalConfigErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RetrievalConfigError, match="not found"):
            load_retrieval_config(tmp_path / "nonexistent.yaml")

    def test_missing_retrieval_key_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text("other: stuff\n", encoding="utf-8")

        with pytest.raises(RetrievalConfigError, match="retrieval"):
            load_retrieval_config(config_file)

    def test_missing_chunking_key_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(
            "retrieval:\n  embedding: stuff\n", encoding="utf-8"
        )

        with pytest.raises(RetrievalConfigError, match="chunking"):
            load_retrieval_config(config_file)

    def test_missing_chunk_size_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(
            "\n".join([
                "retrieval:",
                "  chunking:",
                "    chunk_overlap: 150",
            ]),
            encoding="utf-8",
        )

        with pytest.raises(RetrievalConfigError, match="chunk_size"):
            load_retrieval_config(config_file)

    def test_missing_chunk_overlap_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(
            "\n".join([
                "retrieval:",
                "  chunking:",
                "    chunk_size: 800",
            ]),
            encoding="utf-8",
        )

        with pytest.raises(RetrievalConfigError, match="chunk_overlap"):
            load_retrieval_config(config_file)
