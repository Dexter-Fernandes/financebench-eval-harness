from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from financebench_eval_harness.retrieval_types import Chunk, RetrievalResult


_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunk_metadata.jsonl"


class VectorStore(Protocol):
    """Common interface for vector index backends."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Index chunks with their corresponding embedding vectors."""

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievalResult]:
        """Return the top-k nearest chunks for a query embedding."""


class VectorStoreError(RuntimeError):
    """Raised when a vector store operation fails."""


class FaissVectorStore:
    """Flat L2 FAISS index with chunk metadata sidecar."""

    def __init__(self, *, dim: int) -> None:
        self._dim = dim
        self._index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dim)
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings"
            )
        if not chunks:
            return
        matrix = _to_float32(embeddings)
        if matrix.shape[1] != self._dim:
            raise VectorStoreError(
                f"embedding dimension {matrix.shape[1]} does not match "
                f"index dimension {self._dim}"
            )
        self._index.add(matrix)
        self._chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievalResult]:
        if self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        query = _to_float32([query_embedding])
        distances, indices = self._index.search(query, k)
        results: list[RetrievalResult] = []
        for rank, (dist, idx) in enumerate(
            zip(distances[0].tolist(), indices[0].tolist()), start=1
        ):
            score = 1.0 / (1.0 + float(dist))
            results.append(
                RetrievalResult(
                    question_id="",
                    chunk_id=self._chunks[idx].chunk_id,
                    score=score,
                    rank=rank,
                )
            )
        return results

    @property
    def count(self) -> int:
        return self._index.ntotal

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))
        with (path / _CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "doc_name": chunk.doc_name,
                            "page_num": chunk.page_num,
                            "text": chunk.text,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        path = Path(path)
        index_file = path / _INDEX_FILE
        chunks_file = path / _CHUNKS_FILE
        if not index_file.is_file() or not chunks_file.is_file():
            raise VectorStoreError(f"Index not found at: {path}")
        index = faiss.read_index(str(index_file))
        chunks: list[Chunk] = []
        with chunks_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=d["chunk_id"],
                        doc_id=d["doc_id"],
                        doc_name=d["doc_name"],
                        page_num=d["page_num"],
                        text=d["text"],
                        start_char=d["start_char"],
                        end_char=d["end_char"],
                    )
                )
        store = cls.__new__(cls)
        store._dim = index.d
        store._index = index
        store._chunks = chunks
        return store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float32(embeddings: list[list[float]]) -> np.ndarray:
    return np.array(embeddings, dtype=np.float32)
