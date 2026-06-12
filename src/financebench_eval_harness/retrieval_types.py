from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentPage:
    """One extracted page from a source document, with a stable document ID."""

    doc_id: str
    doc_name: str
    page_num: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A text chunk derived from one document page, with full provenance metadata."""

    chunk_id: str
    doc_id: str
    doc_name: str
    page_num: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked retrieval hit linking a question to a source chunk."""

    question_id: str
    chunk_id: str
    score: float
    rank: int
