"""Retriever compositions — M9.7.

Composes dense, lexical, hybrid, and reranked retrieval behind one `Retriever`
seam. Writes the same retrieval_results.jsonl schema as the original
retriever.run_retrieval (plus optional provenance keys), and an extended
retrieval_run_metadata.json recording retrieval_mode/fusion/reranker settings.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from financebench_eval_harness.embedding import EmbeddingClient
from financebench_eval_harness.fusion import reciprocal_rank_fusion
from financebench_eval_harness.lexical_index import BM25Index
from financebench_eval_harness.query_embedder import embed_question
from financebench_eval_harness.reranker import RerankerClient
from financebench_eval_harness.retrieval_types import Chunk
from financebench_eval_harness.retriever import (
    Question,
    RetrievalRow,
    RetrievalRunMetadata,
    RetrievalRunResult,
    RetrievedChunk,
    _METADATA_FILE,
    _auto_run_id,
)
from financebench_eval_harness.vector_store import FaissVectorStore

if TYPE_CHECKING:
    from financebench_eval_harness.index_builder import IndexMetadata


class Retriever(Protocol):
    """Common interface for all retrieval strategies (dense, lexical, hybrid, reranked)."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top-k retrieved chunks for a query, best first."""


class DenseRetriever:
    """Wraps FaissVectorStore + EmbeddingClient — the original dense retrieval behavior."""

    def __init__(self, store: FaissVectorStore, embedding_client: EmbeddingClient) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in store.chunks}

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_vec = embed_question(query, self._embedding_client)
        results = self._store.search(query_vec, top_k=top_k)
        return [
            RetrievedChunk(
                rank=r.rank,
                chunk_id=r.chunk_id,
                doc_name=self._chunk_by_id[r.chunk_id].doc_name,
                page_num=self._chunk_by_id[r.chunk_id].page_num,
                score=r.score,
                text=self._chunk_by_id[r.chunk_id].text,
            )
            for r in results
        ]


class LexicalRetriever:
    """BM25 retrieval over the same chunks the FAISS index is built from."""

    def __init__(self, store: FaissVectorStore) -> None:
        chunks = store.chunks
        self._chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self._index = BM25Index.from_chunks(
            [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results = self._index.search(query, top_k=top_k)
        return [
            RetrievedChunk(
                rank=rank,
                chunk_id=chunk_id,
                doc_name=self._chunk_by_id[chunk_id].doc_name,
                page_num=self._chunk_by_id[chunk_id].page_num,
                score=score,
                text=self._chunk_by_id[chunk_id].text,
            )
            for rank, (chunk_id, score) in enumerate(results, start=1)
        ]


class HybridRetriever:
    """Fuses dense + lexical candidates (candidate_k each) via reciprocal rank fusion."""

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        chunk_lookup: Mapping[str, Chunk],
        *,
        candidate_k: int = 50,
        fusion_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._chunk_lookup = chunk_lookup
        self._candidate_k = candidate_k
        self._fusion_k = fusion_k
        self._dense_weight = dense_weight
        self._lexical_weight = lexical_weight

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        dense_chunks = self._dense.retrieve(query, self._candidate_k)
        lexical_chunks = self._lexical.retrieve(query, self._candidate_k)

        dense_rank_by_id = {c.chunk_id: c.rank for c in dense_chunks}
        lexical_rank_by_id = {c.chunk_id: c.rank for c in lexical_chunks}

        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c in dense_chunks], [c.chunk_id for c in lexical_chunks]],
            k=self._fusion_k,
            weights=[self._dense_weight, self._lexical_weight],
        )

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(fused[:top_k], start=1):
            chunk = self._chunk_lookup[chunk_id]
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk_id,
                    doc_name=chunk.doc_name,
                    page_num=chunk.page_num,
                    score=score,
                    text=chunk.text,
                    dense_rank=dense_rank_by_id.get(chunk_id),
                    lexical_rank=lexical_rank_by_id.get(chunk_id),
                )
            )
        return results


class RerankedRetriever:
    """Reranks an inner retriever's candidate_k results with a cross-encoder, truncated to top_n."""

    def __init__(
        self,
        inner: Retriever,
        reranker: RerankerClient,
        *,
        candidate_k: int = 50,
        top_n: int | None = None,
    ) -> None:
        self._inner = inner
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._top_n = top_n

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        candidates = self._inner.retrieve(query, self._candidate_k)
        if not candidates:
            return []

        scores = self._reranker.rerank(query, [c.text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        effective_top_n = self._top_n or top_k

        results: list[RetrievedChunk] = []
        for rank, i in enumerate(order[:effective_top_n], start=1):
            candidate = candidates[i]
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=candidate.chunk_id,
                    doc_name=candidate.doc_name,
                    page_num=candidate.page_num,
                    score=scores[i],
                    text=candidate.text,
                    dense_rank=candidate.dense_rank,
                    lexical_rank=candidate.lexical_rank,
                    rerank_score=scores[i],
                )
            )
        return results


def run_retrieval_with_retriever(
    questions: list[Question],
    retriever: Retriever,
    output_path: Path,
    *,
    top_k: int = 5,
    run_id: str | None = None,
    dataset_path: str | Path | None = None,
    chunks_path: str | Path | None = None,
    index_metadata: "IndexMetadata | None" = None,
    retrieval_mode: str = "dense",
    fusion: dict[str, Any] | None = None,
    reranker: dict[str, Any] | None = None,
) -> RetrievalRunResult:
    """Run retrieval for all questions through `retriever` and write results to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        RetrievalRow(
            question_id=q.question_id,
            query=q.query,
            retrieved=retriever.retrieve(q.query, top_k),
        )
        for q in questions
    ]

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
            retrieval_mode=retrieval_mode,
            fusion=fusion,
            reranker=reranker,
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


__all__ = [
    "Retriever",
    "DenseRetriever",
    "LexicalRetriever",
    "HybridRetriever",
    "RerankedRetriever",
    "run_retrieval_with_retriever",
]
