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

