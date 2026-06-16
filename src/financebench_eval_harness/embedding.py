from __future__ import annotations

import hashlib
import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import request
from urllib.error import HTTPError, URLError

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configured embedding provider and model."""

    provider: str
    model_name: str
    base_url: str | None = None
    timeout_seconds: float = 30.0
    batch_size: int = 32
    dimensions: int | None = None
    normalize: bool = True
    category: str | None = None


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

        def _do_embed(batch: list[str]) -> list[list[float]]:
            payload: dict[str, Any] = {"model": self.config.model_name, "input": batch}
            response = self._transport(
                f"{base_url}/api/embed", payload, self.config.timeout_seconds
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

        try:
            return _do_embed(texts)
        except EmbeddingProviderError as exc:
            # Some GGUF-quantized models (e.g. bge-m3) produce NaN in their
            # output vectors for certain inputs. Ollama's Go JSON encoder then
            # refuses to serialise the response. Fall back to one-at-a-time so
            # only the bad chunk is affected, not the entire batch.
            if "NaN" not in str(exc):
                raise
            logger.warning(
                "[ollama/%s] Batch of %d failed (NaN in embeddings) — retrying one at a time",
                self.config.model_name, len(texts),
            )
            dim = self.config.dimensions
            results: list[list[float]] = []
            for text in texts:
                try:
                    vec = _do_embed([text])[0]
                    if dim is None:
                        dim = len(vec)
                    results.append(vec)
                except EmbeddingProviderError as inner:
                    if "NaN" not in str(inner):
                        raise
                    fallback_dim = dim or 1024
                    logger.warning(
                        "[ollama/%s] Single chunk still NaN — substituting zero vector (dim=%d)",
                        self.config.model_name, fallback_dim,
                    )
                    results.append([0.0] * fallback_dim)
            return results


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


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class EmbeddingConfigError(ValueError):
    """Raised for invalid or missing embedding configuration."""


def load_embedding_config(config_path: str | Path) -> EmbeddingConfig:
    """Load EmbeddingConfig from a YAML file.

    Expected structure::

        embedding:
          provider: ollama
          model_name: nomic-embed-text
          base_url: http://localhost:11434   # optional
          timeout_seconds: 30.0             # optional
          batch_size: 32                    # optional
    """
    path = Path(config_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EmbeddingConfigError(f"Embedding config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise EmbeddingConfigError(f"Invalid YAML in embedding config: {exc}") from exc

    if not isinstance(raw, dict) or "embedding" not in raw:
        raise EmbeddingConfigError(
            f"Embedding config must contain a top-level 'embedding' key: {path}"
        )

    section = raw["embedding"]
    if not isinstance(section, dict):
        raise EmbeddingConfigError(f"'embedding' must be a mapping in: {path}")

    for required in ("provider", "model_name"):
        if required not in section:
            raise EmbeddingConfigError(
                f"Embedding config missing required field '{required}' in: {path}"
            )

    return EmbeddingConfig(
        provider=section["provider"],
        model_name=section["model_name"],
        base_url=section.get("base_url"),
        timeout_seconds=float(section.get("timeout_seconds", 30.0)),
        batch_size=int(section.get("batch_size", 32)),
        dimensions=int(section["dimensions"]) if "dimensions" in section else None,
        normalize=bool(section.get("normalize", True)),
        category=section.get("category"),
    )


__all__ = [
    "EmbeddingClient",
    "EmbeddingConfig",
    "EmbeddingConfigError",
    "EmbeddingProviderError",
    "MockEmbeddingClient",
    "OllamaEmbeddingClient",
    "SentenceTransformersEmbeddingClient",
    "load_embedding_config",
]
