from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class RAGInputError(ValueError):
    """Raised when a RAG input cannot be built or loaded."""


@dataclass(frozen=True)
class RAGContextChunk:
    """One retrieved chunk as it enters the answer generator."""

    rank: int
    chunk_id: str
    doc_name: str
    page_num: int
    text: str
    score: float | None = None

    def format_for_prompt(self) -> str:
        """Return a metadata header + text string suitable for passing to render_prompt()."""
        header = f"[chunk_id: {self.chunk_id} | doc: {self.doc_name} | page: {self.page_num}]"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "page_num": self.page_num,
            "text": self.text,
        }
        if self.score is not None:
            d["score"] = self.score
        return d


@dataclass(frozen=True)
class RAGInput:
    """Standardised input to RAG answer generation for one question."""

    question_id: str
    question: str
    retrieved_context: tuple[RAGContextChunk, ...]
    gold_answer: str
    gold_evidence: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "retrieved_context": [c.to_dict() for c in self.retrieved_context],
            "gold_answer": self.gold_answer,
            "gold_evidence": list(self.gold_evidence),
        }


def build_rag_inputs(
    retrieval_rows: Sequence[dict[str, Any]],
    processed_examples: Sequence[dict[str, Any]],
) -> list[RAGInput]:
    """Join retrieval rows and processed examples on question_id."""
    chunks_by_qid: dict[str, tuple[RAGContextChunk, ...]] = defaultdict(tuple)
    for row in retrieval_rows:
        qid = row.get("question_id", "")
        chunks_by_qid[qid] = tuple(
            _chunk_from_dict(c) for c in row.get("retrieved", [])
        )

    results: list[RAGInput] = []
    for example in processed_examples:
        qid = example.get("question_id", "")
        question = example.get("question", "")
        gold_answer = example.get("gold_answer", "")
        raw_evidence = example.get("evidence") or []
        gold_evidence = tuple(e for e in raw_evidence if isinstance(e, dict))

        results.append(
            RAGInput(
                question_id=qid,
                question=question,
                retrieved_context=chunks_by_qid.get(qid, ()),
                gold_answer=gold_answer,
                gold_evidence=gold_evidence,
            )
        )
    return results


def load_rag_inputs(
    retrieval_path: str | Path,
    examples_path: str | Path,
) -> list[RAGInput]:
    """Load retrieval results and processed examples from JSONL files and join them."""
    retrieval_path = Path(retrieval_path)
    examples_path = Path(examples_path)

    if not retrieval_path.is_file():
        raise RAGInputError(f"retrieval file not found: {retrieval_path}")
    if not examples_path.is_file():
        raise RAGInputError(f"examples file not found: {examples_path}")

    try:
        retrieval_rows = [
            json.loads(line)
            for line in retrieval_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (json.JSONDecodeError, OSError) as exc:
        raise RAGInputError(f"Could not read retrieval file: {retrieval_path}") from exc

    try:
        processed_examples = [
            json.loads(line)
            for line in examples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (json.JSONDecodeError, OSError) as exc:
        raise RAGInputError(f"Could not read examples file: {examples_path}") from exc

    return build_rag_inputs(retrieval_rows, processed_examples)


def _chunk_from_dict(d: dict[str, Any]) -> RAGContextChunk:
    return RAGContextChunk(
        rank=d["rank"],
        chunk_id=d["chunk_id"],
        doc_name=d["doc_name"],
        page_num=d["page_num"],
        text=d["text"],
        score=d.get("score"),
    )


__all__ = [
    "RAGContextChunk",
    "RAGInput",
    "RAGInputError",
    "build_rag_inputs",
    "load_rag_inputs",
]
