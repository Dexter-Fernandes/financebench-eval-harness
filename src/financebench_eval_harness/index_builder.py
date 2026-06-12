from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from financebench_eval_harness.embedding import EmbeddingClient
from financebench_eval_harness.retrieval_config import RetrievalConfig
from financebench_eval_harness.retrieval_types import Chunk
from financebench_eval_harness.vector_store import FaissVectorStore


_METADATA_FILE = "index_metadata.json"


@dataclass(frozen=True)
class IndexMetadata:
    """Provenance record written alongside every vector index."""

    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int
    corpus_hash: str
    chunk_count: int
    build_time_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_chars": self.min_chunk_chars,
            "corpus_hash": self.corpus_hash,
            "chunk_count": self.chunk_count,
            "build_time_utc": self.build_time_utc,
        }


class IndexBuildError(RuntimeError):
    """Raised when an index build or load operation fails."""


def build_index(
    chunks: list[Chunk],
    embedding_client: EmbeddingClient,
    config: RetrievalConfig,
    index_dir: Path,
    *,
    on_batch: Callable[[int, int], None] | None = None,
) -> IndexMetadata:
    """Embed chunks, build a FAISS index, and write all artefacts to index_dir.

    on_batch, if provided, is called after each embedding batch as
    on_batch(completed_chunks, total_chunks).
    """
    if not chunks:
        raise IndexBuildError("Cannot build index from empty chunk list")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    batch_size = embedding_client.config.batch_size
    total = len(chunks)
    embeddings: list[list[float]] = []
    for i in range(0, total, batch_size):
        batch_texts = [c.text for c in chunks[i : i + batch_size]]
        embeddings.extend(embedding_client.embed_texts(batch_texts))
        if on_batch is not None:
            on_batch(min(i + batch_size, total), total)

    dim = len(embeddings[0])
    store = FaissVectorStore(dim=dim)
    store.add(chunks, embeddings)
    store.save(index_dir)

    meta = IndexMetadata(
        embedding_provider=embedding_client.config.provider,
        embedding_model=embedding_client.config.model_name,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        min_chunk_chars=config.chunking.min_chunk_chars,
        corpus_hash=_corpus_hash(chunks),
        chunk_count=len(chunks),
        build_time_utc=datetime.now(timezone.utc).isoformat(),
    )
    (index_dir / _METADATA_FILE).write_text(
        json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def load_index(index_dir: Path) -> tuple[FaissVectorStore, IndexMetadata]:
    """Load a saved FAISS index and its metadata from index_dir."""
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        raise IndexBuildError(f"Index directory not found: {index_dir}")

    metadata_path = index_dir / _METADATA_FILE
    if not metadata_path.is_file():
        raise IndexBuildError(f"Index metadata file not found: {metadata_path}")

    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta = IndexMetadata(
        embedding_provider=raw["embedding_provider"],
        embedding_model=raw["embedding_model"],
        chunk_size=raw["chunk_size"],
        chunk_overlap=raw["chunk_overlap"],
        min_chunk_chars=raw["min_chunk_chars"],
        corpus_hash=raw["corpus_hash"],
        chunk_count=raw["chunk_count"],
        build_time_utc=raw["build_time_utc"],
    )
    store = FaissVectorStore.load(index_dir)
    return store, meta


def _corpus_hash(chunks: list[Chunk]) -> str:
    """Stable SHA-256 fingerprint of the corpus, independent of chunk order."""
    entries = sorted(f"{c.chunk_id}\t{c.text}" for c in chunks)
    content = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(content).hexdigest()
