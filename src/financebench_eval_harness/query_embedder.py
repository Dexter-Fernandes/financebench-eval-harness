from __future__ import annotations

from financebench_eval_harness.embedding import EmbeddingClient, EmbeddingProviderError


class QueryEmbeddingError(RuntimeError):
    """Raised when a question cannot be embedded."""


def embed_question(text: str, client: EmbeddingClient) -> list[float]:
    """Embed a single question text into a query vector."""
    if not text.strip():
        raise QueryEmbeddingError("Empty question text cannot be embedded")
    try:
        return client.embed_texts([text])[0]
    except EmbeddingProviderError as exc:
        raise QueryEmbeddingError(f"Failed to embed question: {exc}") from exc


def embed_questions(texts: list[str], client: EmbeddingClient) -> list[list[float]]:
    """Embed a list of question texts into query vectors, preserving order."""
    if not texts:
        raise QueryEmbeddingError("Empty question list cannot be embedded")
    for text in texts:
        if not text.strip():
            raise QueryEmbeddingError("Empty question text cannot be embedded")
    try:
        return client.embed_texts(texts)
    except EmbeddingProviderError as exc:
        raise QueryEmbeddingError(f"Failed to embed questions: {exc}") from exc
