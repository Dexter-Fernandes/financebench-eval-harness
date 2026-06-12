from pathlib import Path
from io import BytesIO
from urllib.error import HTTPError

import pytest

import financebench_eval_harness.llm as llm_module
from financebench_eval_harness.llm import (
    DEFAULT_LLM_CONFIG_PATH,
    LLMClient,
    LLMConfigError,
    LLMGenerationConfig,
    LLMGenerationResult,
    MockLLMClient,
    LLMProviderError,
    OllamaLLMClient,
    load_llm_config,
)


def test_load_llm_config_reads_default_generation_settings() -> None:
    config = load_llm_config()

    assert DEFAULT_LLM_CONFIG_PATH == Path("configs/llm/local.yaml")
    assert config.provider == "ollama"
    assert config.model_name == "llama3.2:3b"
    assert config.temperature == 0.0
    assert config.max_tokens == 512
    assert config.timeout_seconds == 60.0
    assert config.base_url == "http://localhost:11434"


def test_load_llm_config_reads_custom_model_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: ollama",
                "  model_name: local-finance-model",
                "  temperature: 0.2",
                "  max_tokens: 128",
                "  timeout_seconds: 5",
                "  base_url: http://localhost:11434",
            ]
        ),
        encoding="utf-8",
    )

    config = load_llm_config(config_path)

    assert config == LLMGenerationConfig(
        provider="ollama",
        model_name="local-finance-model",
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=5.0,
        base_url="http://localhost:11434",
    )


def test_load_llm_config_reports_missing_required_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: ollama",
                "  model_name: local-finance-model",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(LLMConfigError) as exc_info:
        load_llm_config(config_path)

    message = str(exc_info.value)
    assert "LLM config missing required key(s):" in message
    assert "max_tokens" in message
    assert "temperature" in message
    assert "timeout_seconds" in message


def test_mock_llm_client_implements_generate_without_api_call() -> None:
    config = LLMGenerationConfig(
        provider="mock",
        model_name="mock-model",
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=1.0,
        base_url=None,
    )
    client: LLMClient = MockLLMClient(config, responses=["first answer"])

    assert client.generate("Question?") == LLMGenerationResult(
        text="first answer",
        prompt_tokens=None,
        output_tokens=None,
    )
    assert client.calls == ["Question?"]
    assert client.config == config


def test_ollama_llm_client_generate_uses_configured_model_settings() -> None:
    requests: list[dict[str, object]] = []
    config = LLMGenerationConfig(
        provider="ollama",
        model_name="local-finance-model",
        temperature=0.25,
        max_tokens=64,
        timeout_seconds=3.0,
        base_url="http://localhost:11434",
    )

    def fake_transport(
        request_url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        requests.append(
            {
                "request_url": request_url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "response": "generated answer",
            "prompt_eval_count": 17,
            "eval_count": 9,
        }

    client = OllamaLLMClient(config, transport=fake_transport)

    assert client.generate("Prompt text") == LLMGenerationResult(
        text="generated answer",
        prompt_tokens=17,
        output_tokens=9,
    )
    assert requests == [
        {
            "request_url": "http://localhost:11434/api/generate",
            "payload": {
                "model": "local-finance-model",
                "prompt": "Prompt text",
                "stream": False,
                "options": {
                    "temperature": 0.25,
                    "num_predict": 64,
                },
            },
            "timeout_seconds": 3.0,
        }
    ]


def test_ollama_llm_client_reports_server_connection_error() -> None:
    config = LLMGenerationConfig(
        provider="ollama",
        model_name="local-finance-model",
        temperature=0.0,
        max_tokens=64,
        timeout_seconds=3.0,
        base_url="http://localhost:11434",
    )

    def failing_transport(
        request_url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        raise LLMProviderError("Could not reach Ollama server at http://localhost:11434")

    client = OllamaLLMClient(config, transport=failing_transport)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate("Prompt text")

    assert "Could not reach Ollama server at http://localhost:11434" in str(
        exc_info.value
    )


def test_ollama_llm_client_reports_missing_model_error() -> None:
    config = LLMGenerationConfig(
        provider="ollama",
        model_name="missing-model",
        temperature=0.0,
        max_tokens=64,
        timeout_seconds=3.0,
        base_url="http://localhost:11434",
    )

    def missing_model_transport(
        request_url: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {"error": "model 'missing-model' not found"}

    client = OllamaLLMClient(config, transport=missing_model_transport)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate("Prompt text")

    assert "Ollama model not found: missing-model" in str(exc_info.value)


def test_ollama_http_transport_preserves_missing_model_error_from_http_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LLMGenerationConfig(
        provider="ollama",
        model_name="missing-model",
        temperature=0.0,
        max_tokens=64,
        timeout_seconds=3.0,
        base_url="http://localhost:11434",
    )

    def fake_urlopen(request, timeout: float):
        raise HTTPError(
            url=request.full_url,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=BytesIO(b"{\"error\":\"model 'missing-model' not found\"}"),
        )

    monkeypatch.setattr(llm_module.request, "urlopen", fake_urlopen)
    client = OllamaLLMClient(config)

    with pytest.raises(LLMProviderError) as exc_info:
        client.generate("Prompt text")

    assert "Ollama model not found: missing-model" in str(exc_info.value)
