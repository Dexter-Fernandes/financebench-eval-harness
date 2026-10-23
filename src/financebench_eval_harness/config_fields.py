from __future__ import annotations

from pathlib import Path
from typing import Any

from financebench_eval_harness.evaluation import EvaluationMode


REQUIRED_MODEL_KEYS = {"provider", "model_name", "temperature", "max_tokens", "timeout_seconds"}
REQUIRED_JUDGE_KEYS = REQUIRED_MODEL_KEYS | {"enabled", "prompt"}
REQUIRED_JUDGE_PROMPT_KEYS = {"id", "version", "template_path"}


class FieldReader:
    """Validates fields of a YAML mapping, raising `error` with `label`-prefixed messages."""

    def __init__(self, error: type[Exception], label: str) -> None:
        self._error = error
        self._label = label

    def fail(self, message: str) -> Exception:
        return self._error(f"{self._label} {message}")

    def require_keys(self, mapping: dict[str, Any], required: set[str], section: str) -> None:
        missing = sorted(required - mapping.keys())
        if missing:
            raise self.fail(f"missing required {section} key(s): {', '.join(missing)}")

    def string(self, config: dict[str, Any], key: str, section: str = "") -> str:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise self.fail(f"key '{_dotted(section, key)}' must be a non-empty string")
        return value

    def optional_string(self, config: dict[str, Any], key: str) -> str | None:
        return None if config.get(key) is None else self.string(config, key)

    def path(self, config: dict[str, Any], key: str) -> Path:
        return Path(self.string(config, key))

    def optional_path(self, config: dict[str, Any], key: str) -> Path | None:
        return None if config.get(key) is None else self.path(config, key)

    def float(self, config: dict[str, Any], key: str, section: str = "") -> float:
        value = config.get(key)
        if type(value) not in (float, int):
            raise self.fail(f"key '{_dotted(section, key)}' must be a number")
        return float(value)

    def positive_float(self, config: dict[str, Any], key: str, section: str = "") -> float:
        value = self.float(config, key, section)
        if value <= 0:
            raise self.fail(f"key '{_dotted(section, key)}' must be greater than 0")
        return value

    def positive_int(self, config: dict[str, Any], key: str, section: str = "") -> int:
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise self.fail(f"key '{_dotted(section, key)}' must be a positive integer")
        return value

    def optional_positive_int(self, config: dict[str, Any], key: str) -> int | None:
        return None if config.get(key) is None else self.positive_int(config, key)

    def mode(self, config: dict[str, Any]) -> EvaluationMode:
        value = self.string(config, "mode")
        try:
            return EvaluationMode(value)
        except ValueError as exc:
            raise self._error(f"Unknown evaluation mode: {value}") from exc


def _dotted(section: str, key: str) -> str:
    return f"{section}.{key}" if section else key
