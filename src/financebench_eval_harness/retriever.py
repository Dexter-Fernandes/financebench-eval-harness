from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from financebench_eval_harness.embedding import EmbeddingClient
from financebench_eval_harness.vector_store import FaissVectorStore

if TYPE_CHECKING:
    from financebench_eval_harness.index_builder import IndexMetadata

_METADATA_FILE = "retrieval_run_metadata.json"


@dataclass(frozen=True)
class Question:
    question_id: str
    query: str


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_id: str
    doc_name: str
    page_num: int
    score: float
    text: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_num": self.page_num,
            "score": self.score,
            "text": self.text,
        }
        if self.dense_rank is not None:
            d["dense_rank"] = self.dense_rank
        if self.lexical_rank is not None:
            d["lexical_rank"] = self.lexical_rank
        if self.rerank_score is not None:
            d["rerank_score"] = self.rerank_score
        return d


@dataclass(frozen=True)
class RetrievalRow:
    question_id: str
    query: str
    retrieved: list[RetrievedChunk]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "query": self.query,
            "retrieved": [rc.to_dict() for rc in self.retrieved],
        }


@dataclass(frozen=True)
class RetrievalRunMetadata:
    run_id: str
    dataset_path: str
    chunks_path: str
    embedding_model: str
    vector_store: str
    chunk_size: int | None
    chunk_overlap: int | None
    top_k: int
    retrieval_mode: str = "dense"
    fusion: dict[str, Any] | None = None
    reranker: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "chunks_path": self.chunks_path,
            "embedding_model": self.embedding_model,
            "vector_store": self.vector_store,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "retrieval_mode": self.retrieval_mode,
            "fusion": self.fusion,
            "reranker": self.reranker,
        }


@dataclass(frozen=True)
class RetrievalRunResult:
    question_count: int
    output_path: Path
    metadata_path: Path | None = None


def run_retrieval(
    questions: list[Question],
    store: FaissVectorStore,
    embedding_client: EmbeddingClient,
    output_path: Path,
    *,
    top_k: int = 5,
    run_id: str | None = None,
    dataset_path: str | Path | None = None,
    chunks_path: str | Path | None = None,
    index_metadata: IndexMetadata | None = None,
) -> RetrievalRunResult:
    """Run dense retrieval for all questions and write results to a JSONL file.

    Thin wrapper delegating to DenseRetriever + run_retrieval_with_retriever (M9.7).
    """
    # Local import: hybrid_retriever imports dataclasses from this module, so a
    # module-level import here would create a circular import.
    from financebench_eval_harness.hybrid_retriever import (
        DenseRetriever,
        run_retrieval_with_retriever,
    )

    retriever = DenseRetriever(store, embedding_client)
    return run_retrieval_with_retriever(
        questions,
        retriever,
        output_path,
        top_k=top_k,
        run_id=run_id,
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        index_metadata=index_metadata,
        retrieval_mode="dense",
    )


def next_run_dir(runs_dir: Path) -> Path:
    """Return the next sequential run directory under runs_dir (e.g. run_003)."""
    runs_dir = Path(runs_dir)
    max_num = 0
    if runs_dir.is_dir():
        import re
        pattern = re.compile(r"^run_(\d+)$")
        for child in runs_dir.iterdir():
            m = pattern.match(child.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return runs_dir / f"run_{max_num + 1:03d}"


def _auto_run_id(embedding_model: str) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    model_slug = embedding_model.replace("/", "-")
    return f"{date_str}_dense_{model_slug}"
