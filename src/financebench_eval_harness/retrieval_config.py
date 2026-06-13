from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from financebench_eval_harness.chunking import ChunkingConfig


REQUIRED_CHUNKING_KEYS = {"chunk_size", "chunk_overlap"}
REQUIRED_RETRIEVAL_KEYS = {"chunking", "evidence_overlap_threshold"}


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval pipeline configuration for one experiment."""

    chunking: ChunkingConfig
    evidence_overlap_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_overlap_threshold": self.evidence_overlap_threshold,
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
                "min_chunk_chars": self.chunking.min_chunk_chars,
            },
        }


class RetrievalConfigError(ValueError):
    """Raised when a retrieval config cannot be loaded or validated."""


def load_retrieval_config(config_path: str | Path) -> RetrievalConfig:
    path = Path(config_path)

    if not path.is_file():
        raise RetrievalConfigError(f"Retrieval config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RetrievalConfigError(f"Retrieval config is not valid YAML: {path}") from exc

    if not isinstance(raw, dict) or "retrieval" not in raw:
        raise RetrievalConfigError(
            f"Retrieval config must contain a 'retrieval' mapping: {path}"
        )

    retrieval = raw["retrieval"]
    if not isinstance(retrieval, dict):
        raise RetrievalConfigError(
            f"Retrieval config 'retrieval' must be a mapping: {path}"
        )

    missing_retrieval = sorted(REQUIRED_RETRIEVAL_KEYS - retrieval.keys())
    if missing_retrieval:
        raise RetrievalConfigError(
            f"Retrieval config missing required key(s): {', '.join(missing_retrieval)}"
        )

    chunking_raw = retrieval["chunking"]
    if not isinstance(chunking_raw, dict):
        raise RetrievalConfigError(
            f"Retrieval config 'chunking' must be a mapping: {path}"
        )

    missing = sorted(REQUIRED_CHUNKING_KEYS - chunking_raw.keys())
    if missing:
        raise RetrievalConfigError(
            f"Retrieval config chunking missing required key(s): {', '.join(missing)}"
        )

    return RetrievalConfig(
        chunking=ChunkingConfig(
            chunk_size=chunking_raw["chunk_size"],
            chunk_overlap=chunking_raw["chunk_overlap"],
            strategy=chunking_raw.get("strategy", "recursive_text"),
            min_chunk_chars=chunking_raw.get("min_chunk_chars", 0),
        ),
        evidence_overlap_threshold=float(retrieval["evidence_overlap_threshold"]),
    )
