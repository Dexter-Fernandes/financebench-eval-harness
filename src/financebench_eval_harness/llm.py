from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib import request
from urllib.error import HTTPError, URLError

import yaml


DEFAULT_LLM_CONFIG_PATH = Path("configs/llm/local.yaml")
REQUIRED_LLM_KEYS = {
    "provider",
    "model_name",
    "temperature",
    "max_tokens",
    "timeout_seconds",
}


@dataclass(frozen=True)
class LLMGenerationConfig:
    """Configured model provider and generation settings."""

    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    base_url: str | None = None


@dataclass(frozen=True)
class LLMGenerationResult:
    """Generated text plus optional provider usage metadata."""

    text: str
    prompt_tokens: int | None
    output_tokens: int | None


class LLMClient(Protocol):
    """Common interface for model providers."""

    config: LLMGenerationConfig

    def generate(self, prompt: str) -> LLMGenerationResult:
        """Generate text from a prompt."""


class LLMConfigError(ValueError):
    """Raised when an LLM config cannot be loaded or validated."""


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider call fails or returns an invalid response."""


class MockLLMClient:
    """Deterministic test client that never calls an external API."""

    def __init__(
        self,
        config: LLMGenerationConfig,
        *,
        responses: list[str] | tuple[str, ...],
    ) -> None:
        self.config = config
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> LLMGenerationResult:
        self.calls.append(prompt)
        if not self._responses:
            raise LLMProviderError("Mock LLM has no response configured")
        return LLMGenerationResult(
            text=self._responses.pop(0),
            prompt_tokens=None,
            output_tokens=None,
        )


OllamaTransport = Callable[[str, dict[str, object], float], dict[str, Any]]


class OllamaLLMClient:
    """LLM client for a local Ollama generate endpoint."""

    def __init__(
        self,
        config: LLMGenerationConfig,
        *,
        transport: OllamaTransport | None = None,
    ) -> None:
        if config.provider != "ollama":
            raise LLMConfigError(
                f"OllamaLLMClient requires provider 'ollama', got: {config.provider}"
            )
        self.config = config
        self._transport = transport or _ollama_http_transport

    def generate(self, prompt: str) -> LLMGenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMProviderError("LLM prompt must be a non-empty string")

        base_url = self.config.base_url or "http://localhost:11434"
        payload: dict[str, object] = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = self._transport(
            f"{base_url.rstrip('/')}/api/generate",
            payload,
            self.config.timeout_seconds,
        )
        error = response.get("error")
        if isinstance(error, str) and error.strip():
            if "not found" in error.lower():
                raise LLMProviderError(
                    f"Ollama model not found: {self.config.model_name}. "
                    f"Pull it first or update the config model name."
                )
            raise LLMProviderError(f"Ollama request failed: {error}")
        text = response.get("response")
        if not isinstance(text, str):
            raise LLMProviderError("Ollama response missing string field: response")
        return LLMGenerationResult(
            text=text,
            prompt_tokens=_optional_int(response.get("prompt_eval_count")),
            output_tokens=_optional_int(response.get("eval_count")),
        )


def load_llm_config(
    config_path: str | Path = DEFAULT_LLM_CONFIG_PATH,
) -> LLMGenerationConfig:
    path = Path(config_path)
    if not path.is_file():
        raise LLMConfigError(f"LLM config file not found: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LLMConfigError(f"LLM config is not valid YAML: {path}") from exc

    if not isinstance(raw_config, dict):
        raise LLMConfigError(f"LLM config must be a mapping: {path}")

    llm = raw_config.get("llm")
    if not isinstance(llm, dict):
        raise LLMConfigError(f"LLM config must contain an 'llm' mapping: {path}")

    missing_keys = sorted(REQUIRED_LLM_KEYS - llm.keys())
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise LLMConfigError(f"LLM config missing required key(s): {missing}")

    provider = _required_string(llm, "provider")
    model_name = _required_string(llm, "model_name")
    temperature = _required_float(llm, "temperature")
    max_tokens = _required_positive_int(llm, "max_tokens")
    timeout_seconds = _required_positive_float(llm, "timeout_seconds")
    base_url = _optional_string(llm.get("base_url"), "base_url")

    return LLMGenerationConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        base_url=base_url,
    )


def _ollama_http_transport(
    request_url: str,
    payload: dict[str, object],
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
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        try:
            decoded_error = json.loads(response_body)
        except json.JSONDecodeError:
            decoded_error = None
        if isinstance(decoded_error, dict):
            error = decoded_error.get("error")
            if isinstance(error, str) and error.strip():
                raise LLMProviderError(f"Ollama request failed: {error}") from exc
        raise LLMProviderError(
            f"Ollama HTTP error from {request_url}: {exc.code} {exc.reason}"
        ) from exc
    except URLError as exc:
        base_url = request_url.removesuffix("/api/generate")
        raise LLMProviderError(
            f"Could not reach Ollama server at {base_url}. "
            "Start Ollama or update the configured base_url."
        ) from exc

    try:
        decoded_response = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise LLMProviderError("Ollama response was not valid JSON") from exc

    if not isinstance(decoded_response, dict):
        raise LLMProviderError("Ollama response must be a JSON object")
    return decoded_response


def _required_string(config: dict[str, Any], key: str) -> str:
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigError(f"LLM config key '{key}' must be a non-empty string")
    return value


def _required_float(config: dict[str, Any], key: str) -> float:
    value = config[key]
    if type(value) not in (float, int):
        raise LLMConfigError(f"LLM config key '{key}' must be a number")
    return float(value)


def _required_positive_float(config: dict[str, Any], key: str) -> float:
    value = _required_float(config, key)
    if value <= 0:
        raise LLMConfigError(f"LLM config key '{key}' must be greater than 0")
    return value


def _required_positive_int(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if type(value) is not int or value <= 0:
        raise LLMConfigError(f"LLM config key '{key}' must be a positive integer")
    return value


def _optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise LLMConfigError(f"LLM config key '{key}' must be a non-empty string")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise LLMProviderError("Ollama usage fields must be integers when present")
    return value


__all__ = [
    "DEFAULT_LLM_CONFIG_PATH",
    "LLMClient",
    "LLMConfigError",
    "LLMGenerationConfig",
    "LLMGenerationResult",
    "LLMProviderError",
    "MockLLMClient",
    "OllamaLLMClient",
    "load_llm_config",
]
