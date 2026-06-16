from __future__ import annotations

from typing import Any

import pytest

from financebench_eval_harness.embedding import (
    EmbeddingConfig,
    EmbeddingProviderError,
    MockEmbeddingClient,
    OllamaEmbeddingClient,
    SentenceTransformersEmbeddingClient,
)


MOCK_CONFIG = EmbeddingConfig(provider="mock", model_name="mock-embed")
OLLAMA_CONFIG = EmbeddingConfig(
    provider="ollama",
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    timeout_seconds=30.0,
)
ST_CONFIG = EmbeddingConfig(
    provider="sentence_transformers",
    model_name="all-MiniLM-L6-v2",
)


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------


class TestEmbeddingConfig:
    def test_stores_provider_and_model_name(self) -> None:
        cfg = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text")
        assert cfg.provider == "ollama"
        assert cfg.model_name == "nomic-embed-text"

    def test_is_immutable(self) -> None:
        cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
        with pytest.raises(Exception):
            cfg.provider = "ollama"  # type: ignore[misc]

    def test_base_url_defaults_to_none(self) -> None:
        cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
        assert cfg.base_url is None

    def test_timeout_seconds_has_default(self) -> None:
        cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
        assert cfg.timeout_seconds > 0

    def test_batch_size_has_default(self) -> None:
        cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
        assert cfg.batch_size >= 1

    def test_optional_fields_configurable(self) -> None:
        cfg = EmbeddingConfig(
            provider="ollama",
            model_name="nomic-embed-text",
            base_url="http://myhost:11434",
            timeout_seconds=60.0,
            batch_size=16,
        )
        assert cfg.base_url == "http://myhost:11434"
        assert cfg.timeout_seconds == pytest.approx(60.0)
        assert cfg.batch_size == 16

    def test_dimensions_defaults_to_none(self) -> None:
        cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
        assert cfg.dimensions is None

    def test_normalize_defaults_to_true(self) -> None:
        cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
        assert cfg.normalize is True

    def test_category_defaults_to_none(self) -> None:
        cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
        assert cfg.category is None

    def test_m6_fields_configurable(self) -> None:
        cfg = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",
            dimensions=1536,
            normalize=False,
            category="cheap_api",
        )
        assert cfg.dimensions == 1536
        assert cfg.normalize is False
        assert cfg.category == "cheap_api"


# ---------------------------------------------------------------------------
# MockEmbeddingClient
# ---------------------------------------------------------------------------


class TestMockEmbeddingClient:
    def test_empty_input_returns_empty_list(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG)
        assert client.embed_texts([]) == []

    def test_single_text_returns_one_vector(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG)
        result = client.embed_texts(["hello world"])
        assert len(result) == 1

    def test_batch_returns_one_vector_per_text(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG)
        texts = ["alpha", "beta", "gamma"]
        result = client.embed_texts(texts)
        assert len(result) == len(texts)

    def test_vector_length_matches_embedding_dim(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG, embedding_dim=16)
        result = client.embed_texts(["hello"])
        assert len(result[0]) == 16

    def test_default_embedding_dim_is_positive(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG)
        result = client.embed_texts(["hello"])
        assert len(result[0]) > 0

    def test_all_elements_are_floats(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG, embedding_dim=8)
        result = client.embed_texts(["hello", "world"])
        for vec in result:
            for v in vec:
                assert isinstance(v, float)

    def test_deterministic_same_text_same_vector(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG, embedding_dim=8)
        first = client.embed_texts(["The quick brown fox"])
        second = client.embed_texts(["The quick brown fox"])
        assert first == second

    def test_different_texts_produce_different_vectors(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG, embedding_dim=8)
        result = client.embed_texts(["apple", "zebra"])
        assert result[0] != result[1]

    def test_records_calls(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG)
        client.embed_texts(["first call"])
        client.embed_texts(["second", "call"])
        assert client.calls == [["first call"], ["second", "call"]]

    def test_all_vectors_have_same_dimension(self) -> None:
        client = MockEmbeddingClient(MOCK_CONFIG, embedding_dim=12)
        result = client.embed_texts(["a", "bb", "ccc", "dddd"])
        dims = [len(v) for v in result]
        assert len(set(dims)) == 1


# ---------------------------------------------------------------------------
# OllamaEmbeddingClient
# ---------------------------------------------------------------------------


def _make_ollama_transport(response: dict[str, Any]):
    """Return a transport stub that ignores request and always returns `response`."""
    def transport(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        transport.last_url = url
        transport.last_payload = payload
        return response
    transport.last_url = None
    transport.last_payload = None
    return transport


class TestOllamaEmbeddingClient:
    def test_rejects_wrong_provider(self) -> None:
        wrong_config = EmbeddingConfig(provider="mock", model_name="nomic-embed-text")
        with pytest.raises(Exception):
            OllamaEmbeddingClient(wrong_config)

    def test_calls_embed_endpoint(self) -> None:
        transport = _make_ollama_transport({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        client = OllamaEmbeddingClient(OLLAMA_CONFIG, transport=transport)
        client.embed_texts(["hello", "world"])
        assert "/api/embed" in transport.last_url

    def test_sends_model_and_input_in_payload(self) -> None:
        transport = _make_ollama_transport({"embeddings": [[0.1]]})
        client = OllamaEmbeddingClient(OLLAMA_CONFIG, transport=transport)
        client.embed_texts(["test text"])
        assert transport.last_payload["model"] == "nomic-embed-text"
        assert transport.last_payload["input"] == ["test text"]

    def test_returns_embeddings_from_response(self) -> None:
        expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        transport = _make_ollama_transport({"embeddings": expected})
        client = OllamaEmbeddingClient(OLLAMA_CONFIG, transport=transport)
        result = client.embed_texts(["a", "b"])
        assert result == expected

    def test_raises_provider_error_on_error_field(self) -> None:
        transport = _make_ollama_transport({"error": "model not found"})
        client = OllamaEmbeddingClient(OLLAMA_CONFIG, transport=transport)
        with pytest.raises(EmbeddingProviderError, match="model not found"):
            client.embed_texts(["hello"])

    def test_raises_provider_error_on_missing_embeddings(self) -> None:
        transport = _make_ollama_transport({"status": "ok"})
        client = OllamaEmbeddingClient(OLLAMA_CONFIG, transport=transport)
        with pytest.raises(EmbeddingProviderError, match="embeddings"):
            client.embed_texts(["hello"])

    def test_uses_configured_base_url(self) -> None:
        cfg = EmbeddingConfig(
            provider="ollama",
            model_name="nomic-embed-text",
            base_url="http://myhost:9999",
        )
        transport = _make_ollama_transport({"embeddings": [[0.1]]})
        client = OllamaEmbeddingClient(cfg, transport=transport)
        client.embed_texts(["hello"])
        assert "myhost:9999" in transport.last_url

    def test_empty_input_returns_empty_list(self) -> None:
        transport = _make_ollama_transport({"embeddings": []})
        client = OllamaEmbeddingClient(OLLAMA_CONFIG, transport=transport)
        assert client.embed_texts([]) == []


# ---------------------------------------------------------------------------
# SentenceTransformersEmbeddingClient
# ---------------------------------------------------------------------------


class _FakeSTModel:
    """Fake sentence-transformers model for injection in tests."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.encoded: list[list[str]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        self.encoded.append(sentences)
        return [[float(i) / 10 for i in range(self._dim)] for _ in sentences]


def _make_st_factory(model: _FakeSTModel):
    def factory(model_name: str) -> _FakeSTModel:
        factory.last_model_name = model_name
        return model
    factory.last_model_name = None
    return factory


class TestSentenceTransformersEmbeddingClient:
    def test_encodes_texts_with_model(self) -> None:
        fake_model = _FakeSTModel(dim=4)
        client = SentenceTransformersEmbeddingClient(
            ST_CONFIG, model_factory=_make_st_factory(fake_model)
        )
        client.embed_texts(["hello", "world"])
        assert fake_model.encoded == [["hello", "world"]]

    def test_returns_float_vectors(self) -> None:
        fake_model = _FakeSTModel(dim=4)
        client = SentenceTransformersEmbeddingClient(
            ST_CONFIG, model_factory=_make_st_factory(fake_model)
        )
        result = client.embed_texts(["hello"])
        assert len(result) == 1
        assert all(isinstance(v, float) for v in result[0])

    def test_batch_returns_one_vector_per_text(self) -> None:
        fake_model = _FakeSTModel(dim=4)
        client = SentenceTransformersEmbeddingClient(
            ST_CONFIG, model_factory=_make_st_factory(fake_model)
        )
        result = client.embed_texts(["a", "b", "c"])
        assert len(result) == 3

    def test_passes_model_name_to_factory(self) -> None:
        fake_model = _FakeSTModel()
        factory = _make_st_factory(fake_model)
        client = SentenceTransformersEmbeddingClient(ST_CONFIG, model_factory=factory)
        client.embed_texts(["hello"])
        assert factory.last_model_name == "all-MiniLM-L6-v2"

    def test_raises_if_library_not_available(self) -> None:
        def unavailable_factory(model_name: str):
            raise ImportError("No module named 'sentence_transformers'")

        client = SentenceTransformersEmbeddingClient(
            ST_CONFIG, model_factory=unavailable_factory
        )
        with pytest.raises(EmbeddingProviderError, match="sentence.transformers"):
            client.embed_texts(["hello"])

    def test_empty_input_returns_empty_list(self) -> None:
        fake_model = _FakeSTModel()
        client = SentenceTransformersEmbeddingClient(
            ST_CONFIG, model_factory=_make_st_factory(fake_model)
        )
        assert client.embed_texts([]) == []
