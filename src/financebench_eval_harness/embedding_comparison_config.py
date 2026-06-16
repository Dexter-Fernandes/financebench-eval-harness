"""Configuration types and loader for M6 embedding model comparison.

Loads configs/embedding_comparison.yaml into typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from financebench_eval_harness.chunking import ChunkingConfig
from financebench_eval_harness.embedding import EmbeddingConfig


class EmbeddingComparisonConfigError(ValueError):
    """Raised when the embedding comparison config is missing or invalid."""


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """One candidate embedding model in the comparison."""

    name: str
    provider: str
    category: str
    dimensions: int | None = None
    normalize: bool = True
    batch_size: int = 32
    base_url: str | None = None
    timeout_seconds: float = 60.0

    @property
    def slug(self) -> str:
        """Filesystem-safe unique identifier (slashes replaced with __)."""
        return self.name.replace("/", "__")

    def to_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=self.provider,
            model_name=self.name,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
            batch_size=self.batch_size,
            dimensions=self.dimensions,
            normalize=self.normalize,
            category=self.category,
        )


@dataclass(frozen=True)
class ComparisonRetrievalSettings:
    """Fixed retrieval settings shared across all model runs."""

    chunks_path: Path
    questions_path: Path
    top_k: int
    evidence_overlap_threshold: float
    chunking: ChunkingConfig


@dataclass(frozen=True)
class EmbeddingComparisonConfig:
    """Full configuration for a multi-model embedding comparison run."""

    embedding_models: tuple[EmbeddingModelSpec, ...]
    retrieval: ComparisonRetrievalSettings
    run_id: str
    runs_dir: Path
    reports_dir: Path
    index_base_dir: Path
    cache_dir: Path
    fail_fast: bool = False
    run_rag: bool = False


def load_embedding_comparison_config(path: str | Path) -> EmbeddingComparisonConfig:
    """Parse an embedding_comparison.yaml into an EmbeddingComparisonConfig."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EmbeddingComparisonConfigError(
            f"Embedding comparison config not found: {path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise EmbeddingComparisonConfigError(
            f"Invalid YAML in embedding comparison config: {exc}"
        ) from exc

    if not isinstance(raw, dict) or "comparison" not in raw:
        raise EmbeddingComparisonConfigError(
            f"Config must have a top-level 'comparison' key: {path}"
        )

    cmp = raw["comparison"]

    # --- retrieval ---
    ret_raw = cmp.get("retrieval", {})
    for key in ("chunks_path", "questions_path", "top_k", "evidence_overlap_threshold", "chunking"):
        if key not in ret_raw:
            raise EmbeddingComparisonConfigError(
                f"comparison.retrieval missing required key '{key}' in: {path}"
            )
    ck = ret_raw["chunking"]
    chunking = ChunkingConfig(
        chunk_size=ck["chunk_size"],
        chunk_overlap=ck["chunk_overlap"],
        strategy=ck.get("strategy", "recursive_text"),
        min_chunk_chars=ck.get("min_chunk_chars", 0),
    )
    retrieval = ComparisonRetrievalSettings(
        chunks_path=Path(ret_raw["chunks_path"]),
        questions_path=Path(ret_raw["questions_path"]),
        top_k=int(ret_raw["top_k"]),
        evidence_overlap_threshold=float(ret_raw["evidence_overlap_threshold"]),
        chunking=chunking,
    )

    # --- embedding models ---
    models_raw = cmp.get("embedding_models")
    if not models_raw:
        raise EmbeddingComparisonConfigError(
            f"comparison.embedding_models must be a non-empty list in: {path}"
        )
    specs: list[EmbeddingModelSpec] = []
    for m in models_raw:
        for key in ("name", "provider", "category"):
            if key not in m:
                raise EmbeddingComparisonConfigError(
                    f"embedding_models entry missing required key '{key}' in: {path}"
                )
        specs.append(EmbeddingModelSpec(
            name=m["name"],
            provider=m["provider"],
            category=m["category"],
            dimensions=int(m["dimensions"]) if "dimensions" in m else None,
            normalize=bool(m.get("normalize", True)),
            batch_size=int(m.get("batch_size", 32)),
            base_url=m.get("base_url"),
            timeout_seconds=float(m.get("timeout_seconds", 60.0)),
        ))

    import datetime as _dt
    run_id = cmp.get("run_id") or f"emb_cmp_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    return EmbeddingComparisonConfig(
        embedding_models=tuple(specs),
        retrieval=retrieval,
        run_id=run_id,
        runs_dir=Path(cmp.get("runs_dir", "runs/embedding_comparison")),
        reports_dir=Path(cmp.get("reports_dir", "reports")),
        index_base_dir=Path(cmp.get("index_base_dir", "data/indexes/financebench")),
        cache_dir=Path(cmp.get("cache_dir", "data/cache/embeddings")),
        fail_fast=bool(cmp.get("fail_fast", False)),
        run_rag=bool(cmp.get("run_rag", False)),
    )


__all__ = [
    "ComparisonRetrievalSettings",
    "EmbeddingComparisonConfig",
    "EmbeddingComparisonConfigError",
    "EmbeddingModelSpec",
    "load_embedding_comparison_config",
]
