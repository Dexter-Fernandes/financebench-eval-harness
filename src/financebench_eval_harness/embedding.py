from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib import request
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configured embedding provider and model."""

    provider: str
    model_name: str
    base_url: str | None = None
    timeout_seconds: float = 30.0
    batch_size: int = 32


class EmbeddingClient(Protocol):
    """Common interface for embedding providers."""

    config: EmbeddingConfig

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider call fails."""


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


class MockEmbeddingClient:
    """Deterministic embedding client for tests — never calls an external API."""

    def __init__(self, config: EmbeddingConfig, *, embedding_dim: int = 8) -> None:
        self.config = config
        self._dim = embedding_dim
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_deterministic_vector(text, self._dim) for text in texts]


def _deterministic_vector(text: str, dim: int) -> list[float]:
    """Derive a stable float vector from text using SHA-256."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat digest to fill at least dim * 4 bytes, then unpack as uint32s.
    needed = dim * 4
    repeated = (digest * (needed // len(digest) + 1))[:needed]
    ints = struct.unpack(f">{dim}I", repeated)
    max_u32 = 2**32 - 1
    return [v / max_u32 for v in ints]


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


OllamaEmbedTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class OllamaEmbeddingClient:
    """Embedding client for a local Ollama /api/embed endpoint."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        transport: OllamaEmbedTransport | None = None,
    ) -> None:
        if config.provider != "ollama":
            raise EmbeddingProviderError(
                f"OllamaEmbeddingClient requires provider 'ollama', got: {config.provider}"
            )
        self.config = config
        self._transport = transport or _ollama_http_transport

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "input": texts,
        }
        response = self._transport(
            f"{base_url}/api/embed",
            payload,
            self.config.timeout_seconds,
        )
        error = response.get("error")
        if isinstance(error, str) and error.strip():
            raise EmbeddingProviderError(f"Ollama embedding request failed: {error}")
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingProviderError(
                "Ollama embed response missing 'embeddings' list"
            )
        return embeddings


def _ollama_http_transport(
    request_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        request_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as resp:
            response_body = resp.read().decode("utf-8")
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        try:
            decoded_error = json.loads(response_body)
        except json.JSONDecodeError:
            decoded_error = None
        if isinstance(decoded_error, dict):
            error = decoded_error.get("error")
            if isinstance(error, str) and error.strip():
                return decoded_error
        raise EmbeddingProviderError(
            f"Ollama HTTP error from {request_url}: {exc.code} {exc.reason}"
        ) from exc
    except URLError as exc:
        base_url = request_url.removesuffix("/api/embed")
        raise EmbeddingProviderError(
            f"Could not reach Ollama server at {base_url}. "
            "Start Ollama or update the configured base_url."
        ) from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise EmbeddingProviderError("Ollama response was not valid JSON") from exc

    if not isinstance(decoded, dict):
        raise EmbeddingProviderError("Ollama response must be a JSON object")
    return decoded


# ---------------------------------------------------------------------------
# SentenceTransformers client
# ---------------------------------------------------------------------------


SentenceTransformersFactory = Callable[[str], Any]


class SentenceTransformersEmbeddingClient:
    """Embedding client backed by the sentence-transformers library."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        model_factory: SentenceTransformersFactory | None = None,
    ) -> None:
        self.config = config
        self._model_factory = model_factory or _default_st_factory
        self._model: Any = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            try:
                self._model = self._model_factory(self.config.model_name)
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "sentence_transformers is required for this embedding provider. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        raw = self._model.encode(texts)
        return [[float(v) for v in vec] for vec in raw]


def _default_st_factory(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("No module named 'sentence_transformers'") from exc
    return SentenceTransformer(model_name)


__all__ = [
    "EmbeddingClient",
    "EmbeddingConfig",
    "EmbeddingProviderError",
    "MockEmbeddingClient",
    "OllamaEmbeddingClient",
    "SentenceTransformersEmbeddingClient",
]
