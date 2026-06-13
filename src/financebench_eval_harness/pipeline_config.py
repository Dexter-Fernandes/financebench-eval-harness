from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from financebench_eval_harness.chunking import ChunkingConfig
from financebench_eval_harness.embedding import EmbeddingConfig


_REQUIRED_PATH_KEYS = ("pages_path", "chunks_path", "index_dir", "questions_path", "runs_dir", "evidence_overlap_threshold")


@dataclass(frozen=True)
class PipelineConfig:
    """Unified configuration for the chunk-documents → build-index → retrieve pipeline."""

    pages_path: Path
    chunks_path: Path
    index_dir: Path
    questions_path: Path
    runs_dir: Path
    top_k: int
    evidence_overlap_threshold: float
    chunking: ChunkingConfig
    embedding: EmbeddingConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "evidence_overlap_threshold": self.evidence_overlap_threshold,
            "questions_path": str(self.questions_path),
            "chunks_path": str(self.chunks_path),
            "index_dir": str(self.index_dir),
            "runs_dir": str(self.runs_dir),
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
                "min_chunk_chars": self.chunking.min_chunk_chars,
            },
            "embedding": {
                "provider": self.embedding.provider,
                "model_name": self.embedding.model_name,
                "base_url": self.embedding.base_url,
                "timeout_seconds": self.embedding.timeout_seconds,
                "batch_size": self.embedding.batch_size,
            },
        }


class PipelineConfigError(ValueError):
    """Raised when a pipeline config cannot be loaded or is invalid."""


def load_pipeline_config(path: Path) -> PipelineConfig:
    path = Path(path)
    if not path.is_file():
        raise PipelineConfigError(f"Pipeline config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PipelineConfigError(f"Pipeline config is not valid YAML: {path}") from exc

    if not isinstance(raw, dict) or "retrieval" not in raw:
        raise PipelineConfigError(
            f"Pipeline config must contain a 'retrieval' mapping: {path}"
        )

    r = raw["retrieval"]
    if not isinstance(r, dict):
        raise PipelineConfigError(
            f"Pipeline config 'retrieval' must be a mapping: {path}"
        )

    missing = [k for k in _REQUIRED_PATH_KEYS if k not in r]
    if missing:
        raise PipelineConfigError(
            f"Pipeline config missing required key(s): {', '.join(missing)}"
        )

    chunking_raw = r.get("chunking", {})
    chunking = ChunkingConfig(
        chunk_size=chunking_raw["chunk_size"],
        chunk_overlap=chunking_raw["chunk_overlap"],
        strategy=chunking_raw.get("strategy", "recursive_text"),
        min_chunk_chars=chunking_raw.get("min_chunk_chars", 0),
    )

    emb_raw = r.get("embedding", {})
    embedding = EmbeddingConfig(
        provider=emb_raw["provider"],
        model_name=emb_raw["model_name"],
        base_url=emb_raw.get("base_url"),
        timeout_seconds=float(emb_raw.get("timeout_seconds", 30.0)),
        batch_size=int(emb_raw.get("batch_size", 32)),
    )

    return PipelineConfig(
        pages_path=Path(r["pages_path"]),
        chunks_path=Path(r["chunks_path"]),
        index_dir=Path(r["index_dir"]),
        questions_path=Path(r["questions_path"]),
        runs_dir=Path(r["runs_dir"]),
        top_k=int(r.get("top_k", 5)),
        evidence_overlap_threshold=float(r["evidence_overlap_threshold"]),
        chunking=chunking,
        embedding=embedding,
    )
