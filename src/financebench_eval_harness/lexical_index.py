"""BM25 lexical retrieval index — M9.3.

Built in-memory from the chunk metadata already persisted alongside the FAISS
index (chunk_metadata.jsonl); no separate on-disk artifact, so it can never go
stale relative to the dense index.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from financebench_eval_harness.scoring import _normalize_text


@dataclass(frozen=True)
class BM25Config:
    """Okapi BM25 hyperparameters."""

    k1: float = 1.5
    b: float = 0.75


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return normalized.split() if normalized else []


class BM25Index:
    """In-memory Okapi BM25 index with a Lucene-style non-negative IDF floor."""

    def __init__(
        self,
        *,
        config: BM25Config,
        chunk_ids: list[str],
        doc_term_counts: list[Counter[str]],
        doc_lengths: list[int],
        avg_doc_length: float,
        idf: dict[str, float],
    ) -> None:
        self._config = config
        self._chunk_ids = chunk_ids
        self._doc_term_counts = doc_term_counts
        self._doc_lengths = doc_lengths
        self._avg_doc_length = avg_doc_length
        self._idf = idf

    @classmethod
    def from_chunks(cls, chunks: list[dict], config: BM25Config = BM25Config()) -> "BM25Index":
        chunk_ids = [c["chunk_id"] for c in chunks]
        doc_term_counts = [Counter(_tokenize(c["text"])) for c in chunks]
        doc_lengths = [sum(counts.values()) for counts in doc_term_counts]
        n = len(chunks)
        avg_doc_length = sum(doc_lengths) / n if n else 0.0

        doc_freq: Counter[str] = Counter()
        for counts in doc_term_counts:
            doc_freq.update(counts.keys())

        idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

        return cls(
            config=config,
            chunk_ids=chunk_ids,
            doc_term_counts=doc_term_counts,
            doc_lengths=doc_lengths,
            avg_doc_length=avg_doc_length,
            idf=idf,
        )

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        k1, b = self._config.k1, self._config.b
        avgdl = self._avg_doc_length
        scores: list[tuple[str, float]] = []
        for chunk_id, counts, doc_len in zip(
            self._chunk_ids, self._doc_term_counts, self._doc_lengths
        ):
            score = 0.0
            length_norm = k1 * (1 - b + b * (doc_len / avgdl if avgdl else 0.0))
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                term_idf = self._idf.get(term, 0.0)
                score += term_idf * tf * (k1 + 1) / (tf + length_norm)
            scores.append((chunk_id, score))

        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[:top_k]


__all__ = ["BM25Config", "BM25Index"]
