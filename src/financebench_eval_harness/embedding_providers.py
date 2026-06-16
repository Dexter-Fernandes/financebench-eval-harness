"""Optional embedding provider wrappers for OpenAI and Voyage.

These wrap third-party SDKs that may not be installed. Import errors are
surfaced as EmbeddingProviderError so the comparison runner can skip the
model gracefully instead of crashing.
"""

from __future__ import annotations

from typing import Any

from financebench_eval_harness.embedding import EmbeddingConfig, EmbeddingProviderError


class OpenAIEmbeddingClient:
    """Embedding client backed by the OpenAI embeddings API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            import openai as _openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openai package is required for provider 'openai'. "
                "Install it with: pip install openai"
            ) from exc
        self.config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self.config.model_name, "input": texts}
        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions
        try:
            response = client.embeddings.create(**kwargs)
        except Exception as exc:
            raise EmbeddingProviderError(f"OpenAI embedding call failed: {exc}") from exc
        return [item.embedding for item in response.data]


class VoyageEmbeddingClient:
    """Embedding client backed by the Voyage AI embeddings API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            import voyageai as _voyageai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "voyageai package is required for provider 'voyage'. "
                "Install it with: pip install voyageai"
            ) from exc
        self.config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import voyageai
            self._client = voyageai.Client()
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            result = client.embed(texts, model=self.config.model_name)
        except Exception as exc:
            raise EmbeddingProviderError(f"Voyage embedding call failed: {exc}") from exc
        return result.embeddings


__all__ = ["OpenAIEmbeddingClient", "VoyageEmbeddingClient"]
