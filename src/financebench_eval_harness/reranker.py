"""Reranker seam — M9.5/M9.6.

Provider-agnostic cross-encoder reranking interface mirroring the
EmbeddingClient pattern in embedding.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import request
from urllib.error import HTTPError, URLError

import yaml

from financebench_eval_harness.retrieval_scoring import _jaccard
from financebench_eval_harness.scoring import _normalize_text

_VALID_PROVIDERS = frozenset({"mock", "local", "http"})


@dataclass(frozen=True)
class RerankerConfig:
    """Configured reranker provider and model."""

    provider: str
    model_name: str
    base_url: str | None = None
    timeout_seconds: float = 30.0
    batch_size: int = 16
    top_n: int | None = None
    adapter_path: str | None = None


class RerankerClient(Protocol):
    """Common interface for reranker providers."""

    config: RerankerConfig

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage, same order as input."""


class RerankerConfigError(ValueError):
    """Raised when a reranker config cannot be loaded or validated."""


class RerankerProviderError(RuntimeError):
    """Raised when a reranker provider call fails."""


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


class MockRerankerClient:
    """Deterministic reranker client for tests — never calls an external API.

    Scores by word-overlap (Jaccard) between query and passage unless
    explicit scores are injected via the `scores` argument.
    """

    def __init__(self, config: RerankerConfig, *, scores: list[float] | None = None) -> None:
        self.config = config
        self._injected_scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        if not passages:
            return []
        if self._injected_scores is not None:
            return list(self._injected_scores)
        query_tokens = set(_normalize_text(query).split())
        return [_jaccard(query_tokens, set(_normalize_text(p).split())) for p in passages]


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


RerankerTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class HTTPRerankerClient:
    """Reranker client for an HTTP /rerank endpoint (shared contract with M10's inference server)."""

    def __init__(
        self,
        config: RerankerConfig,
        *,
        transport: RerankerTransport | None = None,
    ) -> None:
        if config.provider != "http":
            raise RerankerConfigError(
                f"HTTPRerankerClient requires provider 'http', got: {config.provider}"
            )
        self.config = config
        self._transport = transport or _http_transport

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        base_url = (self.config.base_url or "").rstrip("/")
        if not base_url:
            raise RerankerProviderError("HTTPRerankerClient requires a base_url")

        payload: dict[str, Any] = {"query": query, "passages": passages}
        response = self._transport(f"{base_url}/rerank", payload, self.config.timeout_seconds)

        scores = response.get("scores")
        if not isinstance(scores, list):
            raise RerankerProviderError("Reranker response missing 'scores' list")
        return [float(s) for s in scores]


LocalRerankerModelFactory = Callable[[str, "str | None"], Any]


class LocalRerankerClient:
    """Reranker client running a local cross-encoder (e.g. BAAI/bge-reranker-base).

    Lazily imports torch/transformers on first use so the base install stays
    dependency-free; the `rerank` optional dependency group is only needed here.
    """

    def __init__(
        self,
        config: RerankerConfig,
        *,
        model_factory: LocalRerankerModelFactory | None = None,
    ) -> None:
        if config.provider != "local":
            raise RerankerConfigError(
                f"LocalRerankerClient requires provider 'local', got: {config.provider}"
            )
        if config.adapter_path is not None and not Path(config.adapter_path).exists():
            raise RerankerConfigError(
                f"Reranker adapter path not found: {config.adapter_path}"
            )
        self.config = config
        self._model_factory = model_factory or _default_local_reranker_factory
        self._model: Any = None

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is None:
            self._model = self._model_factory(self.config.model_name, self.config.adapter_path)

        scores: list[float] = []
        batch_size = self.config.batch_size
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            scores.extend(self._model.score_batch(query, batch))
        return scores


def _select_device(torch_module: Any) -> str:
    """Return "cuda" if a GPU is available to torch, else "cpu"."""
    return "cuda" if torch_module.cuda.is_available() else "cpu"


class _TransformersRerankerModel:
    """Wraps a loaded tokenizer + sequence-classification model for scoring.

    Moves the model to `device` once at construction, and moves each batch's
    input tensors to the same device before inference.
    """

    def __init__(self, *, model: Any, tokenizer: Any, torch_module: Any, device: str) -> None:
        self._model = model.to(device)
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._device = device

    def score_batch(self, query: str, passages: list[str]) -> list[float]:
        inputs = self._tokenizer(
            [query] * len(passages),
            passages,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = inputs.to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)
        return [float(v) for v in logits.tolist()]


def _default_local_reranker_factory(
    model_name: str, adapter_path: str | None
) -> "_TransformersRerankerModel":
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RerankerProviderError(
            "torch and transformers are required for the local reranker provider. "
            "Install them with: pip install .[rerank]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RerankerProviderError(
                "peft is required to load a reranker adapter. "
                "Install it with: pip install .[rerank]"
            ) from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    device = _select_device(torch)
    return _TransformersRerankerModel(model=model, tokenizer=tokenizer, torch_module=torch, device=device)


def _http_transport(
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
        raise RerankerProviderError(
            f"Reranker HTTP error from {request_url}: {exc.code} {exc.reason}"
        ) from exc
    except URLError as exc:
        base_url = request_url.removesuffix("/rerank")
        raise RerankerProviderError(
            f"Could not reach reranker server at {base_url}: {exc.reason}"
        ) from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RerankerProviderError("Reranker response was not valid JSON") from exc

    if not isinstance(decoded, dict):
        raise RerankerProviderError("Reranker response must be a JSON object")
    return decoded


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_reranker_config(config_path: str | Path) -> RerankerConfig:
    """Load RerankerConfig from a YAML file.

    Expected structure::

        reranker:
          provider: mock | local | http
          model_name: <model name>
          base_url: http://localhost:8000   # optional, http provider
          timeout_seconds: 30.0             # optional
          batch_size: 16                    # optional
          top_n: 20                         # optional
          adapter_path: /path/to/adapter    # optional, local provider
    """
    path = Path(config_path)
    if not path.is_file():
        raise RerankerConfigError(f"Reranker config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RerankerConfigError(f"Invalid YAML in reranker config: {exc}") from exc

    if not isinstance(raw, dict) or "reranker" not in raw:
        raise RerankerConfigError(
            f"Reranker config must contain a top-level 'reranker' key: {path}"
        )

    section = raw["reranker"]
    if not isinstance(section, dict):
        raise RerankerConfigError(f"'reranker' must be a mapping in: {path}")

    for required in ("provider", "model_name"):
        if required not in section:
            raise RerankerConfigError(
                f"Reranker config missing required field '{required}' in: {path}"
            )

    provider = section["provider"]
    if provider not in _VALID_PROVIDERS:
        raise RerankerConfigError(
            f"Reranker config 'provider' must be one of {sorted(_VALID_PROVIDERS)}, got: {provider!r}"
        )

    return RerankerConfig(
        provider=provider,
        model_name=section["model_name"],
        base_url=section.get("base_url"),
        timeout_seconds=float(section.get("timeout_seconds", 30.0)),
        batch_size=int(section.get("batch_size", 16)),
        top_n=int(section["top_n"]) if "top_n" in section else None,
        adapter_path=section.get("adapter_path"),
    )


__all__ = [
    "RerankerClient",
    "RerankerConfig",
    "RerankerConfigError",
    "RerankerProviderError",
    "MockRerankerClient",
    "HTTPRerankerClient",
    "LocalRerankerClient",
    "load_reranker_config",
]
