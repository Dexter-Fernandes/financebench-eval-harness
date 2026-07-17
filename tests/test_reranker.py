"""Tests for reranker.py — M9.5 reranker seam (config/Protocol/mock/HTTP)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import financebench_eval_harness.reranker as reranker_module
from financebench_eval_harness.reranker import (
    HTTPRerankerClient,
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
