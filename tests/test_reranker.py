"""Tests for reranker.py — M9.5 reranker seam (config/Protocol/mock/HTTP)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import financebench_eval_harness.reranker as reranker_module
from financebench_eval_harness.reranker import (
    HTTPRerankerClient,
    LocalRerankerClient,
    MockRerankerClient,
    RerankerConfig,
    RerankerConfigError,
    RerankerProviderError,
    load_reranker_config,
)

# ---------------------------------------------------------------------------
# MockRerankerClient
# ---------------------------------------------------------------------------


def test_mock_reranker_client_is_deterministic() -> None:
    config = RerankerConfig(provider="mock", model_name="mock-reranker")
    client = MockRerankerClient(config)

    first = client.rerank("total revenue", ["Total revenue was $10M.", "Unrelated text about cats."])
    second = client.rerank("total revenue", ["Total revenue was $10M.", "Unrelated text about cats."])

    assert first == second


def test_mock_reranker_client_scores_are_orderable() -> None:
    config = RerankerConfig(provider="mock", model_name="mock-reranker")
    client = MockRerankerClient(config)

    scores = client.rerank(
        "total revenue increased",
        ["Unrelated text about cats.", "Total revenue increased this year."],
    )
    assert scores[1] > scores[0]


def test_mock_reranker_client_uses_injected_scores_when_given() -> None:
    config = RerankerConfig(provider="mock", model_name="mock-reranker")
    client = MockRerankerClient(config, scores=[0.9, 0.1])

    assert client.rerank("query", ["passage a", "passage b"]) == [0.9, 0.1]


def test_mock_reranker_client_empty_passages_returns_empty() -> None:
    config = RerankerConfig(provider="mock", model_name="mock-reranker")
    client = MockRerankerClient(config)
    assert client.rerank("query", []) == []


# ---------------------------------------------------------------------------
# HTTPRerankerClient — transport injection
# ---------------------------------------------------------------------------


def test_http_reranker_client_round_trips_fake_transport() -> None:
    requests: list[dict[str, object]] = []
    config = RerankerConfig(
        provider="http",
        model_name="bge-reranker-base",
        base_url="http://localhost:8000",
        timeout_seconds=5.0,
    )

    def fake_transport(request_url: str, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
        requests.append({"request_url": request_url, "payload": payload, "timeout_seconds": timeout_seconds})
        return {"scores": [0.8, 0.2], "model": "bge-reranker-base", "adapter": None}

    client = HTTPRerankerClient(config, transport=fake_transport)
    scores = client.rerank("query", ["passage a", "passage b"])

    assert scores == [0.8, 0.2]
    assert requests == [
        {
            "request_url": "http://localhost:8000/rerank",
            "payload": {"query": "query", "passages": ["passage a", "passage b"]},
            "timeout_seconds": 5.0,
        }
    ]


def test_http_reranker_client_empty_passages_returns_empty_without_calling_transport() -> None:
    config = RerankerConfig(provider="http", model_name="m", base_url="http://localhost:8000")
    calls: list[object] = []

    def fake_transport(request_url, payload, timeout_seconds):
        calls.append(1)
        return {"scores": []}

    client = HTTPRerankerClient(config, transport=fake_transport)
    assert client.rerank("query", []) == []
    assert calls == []


def test_http_reranker_client_raises_on_malformed_transport_response() -> None:
    config = RerankerConfig(provider="http", model_name="m", base_url="http://localhost:8000")

    def bad_transport(request_url, payload, timeout_seconds):
        return {"model": "m"}  # missing 'scores'

    client = HTTPRerankerClient(config, transport=bad_transport)
    with pytest.raises(RerankerProviderError):
        client.rerank("query", ["a passage"])


def test_http_reranker_client_requires_http_provider() -> None:
    config = RerankerConfig(provider="mock", model_name="m", base_url="http://localhost:8000")
    with pytest.raises(RerankerConfigError):
        HTTPRerankerClient(config)


# ---------------------------------------------------------------------------
# HTTPRerankerClient — default transport (real urllib path)
# ---------------------------------------------------------------------------


def test_default_http_transport_reports_server_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RerankerConfig(provider="http", model_name="m", base_url="http://localhost:8000")

    def fake_urlopen(request, timeout: float):
        raise URLError("Connection refused")

    monkeypatch.setattr(reranker_module.request, "urlopen", fake_urlopen)
    client = HTTPRerankerClient(config)

    with pytest.raises(RerankerProviderError) as exc_info:
        client.rerank("query", ["a passage"])
    assert "Could not reach" in str(exc_info.value)


def test_default_http_transport_reports_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RerankerConfig(provider="http", model_name="m", base_url="http://localhost:8000")

    def fake_urlopen(request, timeout: float):
        raise HTTPError(
            url=request.full_url, code=500, msg="Internal Server Error",
            hdrs=None, fp=BytesIO(b"internal error"),
        )

    monkeypatch.setattr(reranker_module.request, "urlopen", fake_urlopen)
    client = HTTPRerankerClient(config)

    with pytest.raises(RerankerProviderError) as exc_info:
        client.rerank("query", ["a passage"])
    assert "500" in str(exc_info.value)


def test_default_http_transport_reports_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RerankerConfig(provider="http", model_name="m", base_url="http://localhost:8000")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"not json"

    def fake_urlopen(request, timeout: float):
        return _FakeResponse()

    monkeypatch.setattr(reranker_module.request, "urlopen", fake_urlopen)
    client = HTTPRerankerClient(config)

    with pytest.raises(RerankerProviderError) as exc_info:
        client.rerank("query", ["a passage"])
    assert "JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_reranker_config
# ---------------------------------------------------------------------------


def test_load_reranker_config_reads_mock_provider(tmp_path: Path) -> None:
    path = tmp_path / "mock.yaml"
    path.write_text(
        "reranker:\n  provider: mock\n  model_name: mock-reranker\n",
        encoding="utf-8",
    )
    config = load_reranker_config(path)
    assert config.provider == "mock"
    assert config.model_name == "mock-reranker"


def test_load_reranker_config_reads_http_provider_fields(tmp_path: Path) -> None:
    path = tmp_path / "http.yaml"
    path.write_text(
        "reranker:\n"
        "  provider: http\n"
        "  model_name: bge-reranker-base\n"
        "  base_url: http://localhost:8000\n"
        "  timeout_seconds: 10.0\n"
        "  batch_size: 8\n"
        "  top_n: 20\n",
        encoding="utf-8",
    )
    config = load_reranker_config(path)
    assert config.base_url == "http://localhost:8000"
    assert config.timeout_seconds == 10.0
    assert config.batch_size == 8
    assert config.top_n == 20


def test_load_reranker_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RerankerConfigError):
        load_reranker_config(tmp_path / "missing.yaml")


def test_load_reranker_config_missing_required_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("reranker:\n  provider: mock\n", encoding="utf-8")
    with pytest.raises(RerankerConfigError):
        load_reranker_config(path)


def test_load_reranker_config_invalid_provider_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "reranker:\n  provider: not_a_real_provider\n  model_name: m\n",
        encoding="utf-8",
    )
    with pytest.raises(RerankerConfigError):
        load_reranker_config(path)


def test_repo_mock_config_loads() -> None:
    config = load_reranker_config(Path("configs/reranker/mock.yaml"))
    assert config.provider == "mock"


def test_repo_http_config_loads() -> None:
    config = load_reranker_config(Path("configs/reranker/http.yaml"))
    assert config.provider == "http"
    assert config.base_url == "http://localhost:8000"


def test_repo_bge_base_local_config_loads() -> None:
    config = load_reranker_config(Path("configs/reranker/bge_base_local.yaml"))
    assert config.provider == "local"
    assert config.model_name == "BAAI/bge-reranker-base"


# ---------------------------------------------------------------------------
# LocalRerankerClient — M9.6
# ---------------------------------------------------------------------------


class _StubLocalModel:
    """Stand-in for a loaded tokenizer+model pair — no torch import required."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def score_batch(self, query: str, passages: list[str]) -> list[float]:
        self.batches.append(list(passages))
        return [float(len(p)) for p in passages]


def test_local_reranker_client_batches_according_to_batch_size() -> None:
    config = RerankerConfig(provider="local", model_name="stub-model", batch_size=2)
    stub = _StubLocalModel()
    client = LocalRerankerClient(config, model_factory=lambda model_name, adapter_path: stub)

    scores = client.rerank("query", ["a", "bb", "ccc", "dddd", "e"])

    assert scores == [1.0, 2.0, 3.0, 4.0, 1.0]
    assert stub.batches == [["a", "bb"], ["ccc", "dddd"], ["e"]]


def test_local_reranker_client_loads_model_lazily_once() -> None:
    config = RerankerConfig(provider="local", model_name="stub-model")
    stub = _StubLocalModel()
    factory_calls: list[tuple[str, str | None]] = []

    def factory(model_name: str, adapter_path: str | None):
        factory_calls.append((model_name, adapter_path))
        return stub

    client = LocalRerankerClient(config, model_factory=factory)
    client.rerank("q1", ["a"])
    client.rerank("q2", ["b"])

    assert factory_calls == [("stub-model", None)]


def test_local_reranker_client_empty_passages_returns_empty_without_loading_model() -> None:
    config = RerankerConfig(provider="local", model_name="stub-model")
    factory_calls: list[object] = []

    def factory(model_name: str, adapter_path: str | None):
        factory_calls.append(1)
        return _StubLocalModel()

    client = LocalRerankerClient(config, model_factory=factory)
    assert client.rerank("query", []) == []
    assert factory_calls == []


def test_local_reranker_client_requires_local_provider() -> None:
    config = RerankerConfig(provider="mock", model_name="m")
    with pytest.raises(RerankerConfigError):
        LocalRerankerClient(config)


def test_local_reranker_client_missing_adapter_path_raises_clear_error(tmp_path: Path) -> None:
    config = RerankerConfig(
        provider="local", model_name="m", adapter_path=str(tmp_path / "does_not_exist"),
    )
    with pytest.raises(RerankerConfigError) as exc_info:
        LocalRerankerClient(config)
    assert "adapter" in str(exc_info.value).lower()


def test_local_reranker_client_missing_torch_gives_clear_install_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the lazy-import ImportError branch regardless of whether
    torch/transformers happen to be installed in the test environment."""
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    config = RerankerConfig(provider="local", model_name="m")
    client = LocalRerankerClient(config)

    with pytest.raises(RerankerProviderError) as exc_info:
        client.rerank("query", ["a passage"])
    assert "pip install .[rerank]" in str(exc_info.value)


def test_local_reranker_client_real_model_path_wires_tokenizer_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers")
    import torch

    class _FakeOutput:
        def __init__(self, n: int) -> None:
            self.logits = torch.tensor([[float(i)] for i in range(n)])

    class _FakeModel:
        def eval(self) -> "_FakeModel":
            return self

        def __call__(self, **kwargs):
            return _FakeOutput(kwargs["input_ids"].shape[0])

    class _FakeTokenizer:
        def __call__(self, queries, passages, return_tensors=None, padding=None, truncation=None):
            n = len(passages)
            return {"input_ids": torch.zeros((n, 4), dtype=torch.long)}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    monkeypatch.setattr(
        AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: _FakeModel()
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer())

    config = RerankerConfig(provider="local", model_name="fake-model")
    client = LocalRerankerClient(config)
    scores = client.rerank("query", ["p1", "p2", "p3"])

    assert scores == [0.0, 1.0, 2.0]
