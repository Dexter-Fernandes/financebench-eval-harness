from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from financebench_eval_harness.embedding import EmbeddingClient
from financebench_eval_harness.query_embedder import embed_question
from financebench_eval_harness.retrieval_types import Chunk
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_num": self.page_num,
            "score": self.score,
            "text": self.text,
        }


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
        }


@dataclass(frozen=True)
class RetrievalRunResult:
    question_count: int
    output_path: Path
    metadata_path: Path | None = None


def retrieve_for_questions(
    questions: list[Question],
    store: FaissVectorStore,
    embedding_client: EmbeddingClient,
    *,
    top_k: int = 5,
) -> list[RetrievalRow]:
    """Embed each question and return the top-k nearest chunks from the store."""
    if not questions:
        return []

    chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in store.chunks}
    rows: list[RetrievalRow] = []

    for question in questions:
        query_vec = embed_question(question.query, embedding_client)
        results = store.search(query_vec, top_k=top_k)
        retrieved = [
            RetrievedChunk(
                rank=r.rank,
                chunk_id=r.chunk_id,
                doc_name=chunk_by_id[r.chunk_id].doc_name,
                page_num=chunk_by_id[r.chunk_id].page_num,
                score=r.score,
                text=chunk_by_id[r.chunk_id].text,
            )
            for r in results
        ]
        rows.append(RetrievalRow(
            question_id=question.question_id,
            query=question.query,
            retrieved=retrieved,
        ))

    return rows


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
    """Run retrieval for all questions and write results to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = retrieve_for_questions(questions, store, embedding_client, top_k=top_k)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    metadata_path: Path | None = None
    if index_metadata is not None:
        effective_run_id = run_id or _auto_run_id(index_metadata.embedding_model)
        run_meta = RetrievalRunMetadata(
            run_id=effective_run_id,
            dataset_path=str(dataset_path) if dataset_path is not None else "",
            chunks_path=str(chunks_path) if chunks_path is not None else "",
            embedding_model=index_metadata.embedding_model,
            vector_store="faiss",
            chunk_size=index_metadata.chunk_size,
            chunk_overlap=index_metadata.chunk_overlap,
            top_k=top_k,
        )
        metadata_path = output_path.parent / _METADATA_FILE
        metadata_path.write_text(
            json.dumps(run_meta.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return RetrievalRunResult(
        question_count=len(rows),
        output_path=output_path,
        metadata_path=metadata_path,
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
