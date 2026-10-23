from __future__ import annotations

import pytest

from financebench_eval_harness.embedding import EmbeddingConfig, EmbeddingProviderError, MockEmbeddingClient
from financebench_eval_harness.query_embedder import (
    QueryEmbeddingError,
    embed_question,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

EMBED_CFG = EmbeddingConfig(provider="mock", model_name="mock-embed")
EMBED_DIM = 8


def make_client() -> MockEmbeddingClient:
    return MockEmbeddingClient(EMBED_CFG, embedding_dim=EMBED_DIM)


class _FailingEmbeddingClient:
    """Minimal EmbeddingClient that always raises EmbeddingProviderError."""

    config = EMBED_CFG

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingProviderError("connection refused")


# ---------------------------------------------------------------------------
# embed_question
# ---------------------------------------------------------------------------


class TestEmbedQuestion:
    def test_returns_list_of_floats(self) -> None:
        vec = embed_question("What was the revenue?", make_client())
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_vector_has_correct_dimension(self) -> None:
        vec = embed_question("What was the revenue?", make_client())
        assert len(vec) == EMBED_DIM

    def test_same_text_returns_same_vector(self) -> None:
        text = "What was the net income for Q3?"
        vec1 = embed_question(text, make_client())
        vec2 = embed_question(text, make_client())
        assert vec1 == vec2

    def test_different_texts_produce_different_vectors(self) -> None:
        vec1 = embed_question("What was revenue?", make_client())
        vec2 = embed_question("What was net income?", make_client())
        assert vec1 != vec2

    def test_delegates_to_client_embed_texts(self) -> None:
        client = make_client()
        embed_question("What was the revenue?", client)
        assert len(client.calls) == 1
        assert client.calls[0] == ["What was the revenue?"]

    def test_raises_on_empty_string(self) -> None:
        with pytest.raises(QueryEmbeddingError, match="[Ee]mpty"):
            embed_question("", make_client())

    def test_raises_on_whitespace_only_string(self) -> None:
        with pytest.raises(QueryEmbeddingError, match="[Ee]mpty"):
            embed_question("   \t\n", make_client())

    def test_provider_error_wrapped_as_query_embedding_error(self) -> None:
        with pytest.raises(QueryEmbeddingError):
            embed_question("What was the revenue?", _FailingEmbeddingClient())

    def test_provider_error_preserves_cause(self) -> None:
        with pytest.raises(QueryEmbeddingError) as exc_info:
            embed_question("What was the revenue?", _FailingEmbeddingClient())
        assert exc_info.value.__cause__ is not None
