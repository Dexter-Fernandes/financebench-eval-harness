from pathlib import Path

import pytest

from financebench_eval_harness.llm import (
    DEFAULT_LLM_CONFIG_PATH,
    LLMClient,
    LLMConfigError,
    LLMGenerationConfig,
    MockLLMClient,
    OllamaClient,
    load_llm_config,
)


def test_load_llm_config_reads_default_generation_settings() -> None:
    config = load_llm_config()

    assert DEFAULT_LLM_CONFIG_PATH == Path("configs/llm/local.yaml")
    assert config.provider == "ollama"
    assert config.model_name
    assert config.temperature == 0.0
    assert config.max_tokens == 512
    assert config.timeout_seconds == 30.0


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
    )
    client: LLMClient = MockLLMClient(config, responses=["first answer"])

    assert client.generate("Question?") == "first answer"
    assert client.calls == ["Question?"]
    assert client.config == config


def test_ollama_client_generate_uses_configured_model_settings() -> None:
    requests: list[dict[str, object]] = []
    config = LLMGenerationConfig(
        provider="ollama",
        model_name="local-finance-model",
        temperature=0.25,
        max_tokens=64,
        timeout_seconds=3.0,
    )

    def fake_transport(payload: dict[str, object], timeout_seconds: float) -> dict[str, str]:
        requests.append({"payload": payload, "timeout_seconds": timeout_seconds})
        return {"response": "generated answer"}

    client = OllamaClient(config, transport=fake_transport)

    assert client.generate("Prompt text") == "generated answer"
    assert requests == [
        {
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
