from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import LLMGenerationConfig


DEFAULT_EVALUATION_RUN_CONFIG_PATH = Path("configs/evaluation/local_mock.yaml")
REQUIRED_EVAL_KEYS = {"dataset_path", "output_dir", "mode", "limit"}
REQUIRED_MODEL_KEYS = {
    "provider",
    "model_name",
    "temperature",
    "max_tokens",
    "timeout_seconds",
}
REQUIRED_JUDGE_KEYS = REQUIRED_MODEL_KEYS | {"enabled", "prompt"}
REQUIRED_JUDGE_PROMPT_KEYS = {"id", "version", "template_path"}


@dataclass(frozen=True)
class EvaluationRunSettings:
    """Configured dataset, output, and baseline mode for one run."""

    dataset_path: Path
    output_dir: Path
    mode: EvaluationMode
    limit: int


@dataclass(frozen=True)
class JudgePromptConfig:
    """Versioned prompt file used for LLM-as-judge scoring."""

    id: str
    version: str
    template_path: Path


@dataclass(frozen=True)
class JudgeConfig:
    """Optional LLM-as-judge configuration for one run."""

    enabled: bool
    model: LLMGenerationConfig
    prompt: JudgePromptConfig


@dataclass(frozen=True)
class EvaluationRunConfig:
    """Full reproducible evaluation run configuration."""

    settings: EvaluationRunSettings
    model: LLMGenerationConfig
    judge: JudgeConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "eval": {
                "dataset_path": str(self.settings.dataset_path),
                "output_dir": str(self.settings.output_dir),
                "mode": self.settings.mode.value,
                "limit": self.settings.limit,
            },
            "model": {
                "provider": self.model.provider,
                "model_name": self.model.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "timeout_seconds": self.model.timeout_seconds,
            },
        }
        if self.judge is not None:
            config["judge"] = {
                "enabled": self.judge.enabled,
                "provider": self.judge.model.provider,
                "model_name": self.judge.model.model_name,
                "temperature": self.judge.model.temperature,
                "max_tokens": self.judge.model.max_tokens,
                "timeout_seconds": self.judge.model.timeout_seconds,
                "prompt": {
                    "id": self.judge.prompt.id,
                    "version": self.judge.prompt.version,
                    "template_path": str(self.judge.prompt.template_path),
                },
            }
        return config


class EvaluationRunConfigError(ValueError):
    """Raised when an evaluation run config cannot be loaded or validated."""


def load_evaluation_run_config(
    config_path: str | Path = DEFAULT_EVALUATION_RUN_CONFIG_PATH,
) -> EvaluationRunConfig:
    path = Path(config_path)
    if not path.is_file():
        raise EvaluationRunConfigError(f"Evaluation run config file not found: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationRunConfigError(
            f"Evaluation run config is not valid YAML: {path}"
        ) from exc

    if not isinstance(raw_config, dict):
        raise EvaluationRunConfigError(
            f"Evaluation run config must be a mapping: {path}"
        )

    eval_config = raw_config.get("eval")
    if not isinstance(eval_config, dict):
        raise EvaluationRunConfigError(
            f"Evaluation run config must contain an 'eval' mapping: {path}"
        )
    model_config = raw_config.get("model")
    if not isinstance(model_config, dict):
        raise EvaluationRunConfigError(
            f"Evaluation run config must contain a 'model' mapping: {path}"
        )

    _validate_required_eval_keys(eval_config)
    _validate_required_model_keys(model_config)

    judge_config = _judge_config_from_mapping(raw_config.get("judge"))

    return EvaluationRunConfig(
        settings=EvaluationRunSettings(
            dataset_path=_path_from_mapping(eval_config, "dataset_path"),
            output_dir=_path_from_mapping(eval_config, "output_dir"),
            mode=_mode_from_mapping(eval_config),
            limit=_positive_int_from_mapping(eval_config, "limit"),
        ),
        model=LLMGenerationConfig(
            provider=_string_from_mapping(model_config, "provider"),
            model_name=_string_from_mapping(model_config, "model_name"),
            temperature=_float_from_mapping(model_config, "temperature"),
            max_tokens=_positive_int_from_mapping(model_config, "max_tokens"),
            timeout_seconds=_positive_float_from_mapping(
                model_config,
                "timeout_seconds",
            ),
        ),
        judge=judge_config,
    )


def _validate_required_eval_keys(eval_config: dict[str, Any]) -> None:
    missing_keys = sorted(REQUIRED_EVAL_KEYS - eval_config.keys())
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise EvaluationRunConfigError(
            f"Evaluation run config missing required eval key(s): {missing}"
        )


def _validate_required_model_keys(model_config: dict[str, Any]) -> None:
    missing_keys = sorted(REQUIRED_MODEL_KEYS - model_config.keys())
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise EvaluationRunConfigError(
            f"Evaluation run config missing required model key(s): {missing}"
        )


def _judge_config_from_mapping(raw_judge_config: Any) -> JudgeConfig | None:
    if raw_judge_config is None:
        return None
    if not isinstance(raw_judge_config, dict):
        raise EvaluationRunConfigError(
            "Evaluation run config key 'judge' must be a mapping"
        )

    enabled = raw_judge_config.get("enabled")
    if type(enabled) is not bool:
        raise EvaluationRunConfigError(
            "Evaluation run config key 'judge.enabled' must be a boolean"
        )
    if not enabled:
        return None

    missing_keys = sorted(REQUIRED_JUDGE_KEYS - raw_judge_config.keys())
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise EvaluationRunConfigError(
            f"Evaluation run config missing required judge key(s): {missing}"
        )

    prompt_config = raw_judge_config["prompt"]
    if not isinstance(prompt_config, dict):
        raise EvaluationRunConfigError(
            "Evaluation run config key 'judge.prompt' must be a mapping"
        )

    missing_prompt_keys = sorted(REQUIRED_JUDGE_PROMPT_KEYS - prompt_config.keys())
    if missing_prompt_keys:
        missing = ", ".join(missing_prompt_keys)
        raise EvaluationRunConfigError(
            f"Evaluation run config missing required judge prompt key(s): {missing}"
        )

    return JudgeConfig(
        enabled=True,
        model=LLMGenerationConfig(
            provider=_string_from_mapping(raw_judge_config, "provider"),
            model_name=_string_from_mapping(raw_judge_config, "model_name"),
            temperature=_float_from_mapping(raw_judge_config, "temperature"),
            max_tokens=_positive_int_from_mapping(raw_judge_config, "max_tokens"),
            timeout_seconds=_positive_float_from_mapping(
                raw_judge_config,
                "timeout_seconds",
            ),
        ),
        prompt=JudgePromptConfig(
            id=_string_from_mapping(prompt_config, "id"),
            version=_string_from_mapping(prompt_config, "version"),
            template_path=_path_from_mapping(prompt_config, "template_path"),
        ),
    )


def _mode_from_mapping(config: dict[str, Any]) -> EvaluationMode:
    value = _string_from_mapping(config, "mode")
    try:
        return EvaluationMode(value)
    except ValueError as exc:
        raise EvaluationRunConfigError(f"Unknown evaluation mode: {value}") from exc


def _path_from_mapping(config: dict[str, Any], key: str) -> Path:
    return Path(_string_from_mapping(config, key))


def _string_from_mapping(config: dict[str, Any], key: str) -> str:
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise EvaluationRunConfigError(
            f"Evaluation run config key '{key}' must be a non-empty string"
        )
    return value


def _float_from_mapping(config: dict[str, Any], key: str) -> float:
    value = config[key]
    if type(value) not in (float, int):
        raise EvaluationRunConfigError(
            f"Evaluation run config key '{key}' must be a number"
        )
    return float(value)


def _positive_float_from_mapping(config: dict[str, Any], key: str) -> float:
    value = _float_from_mapping(config, key)
    if value <= 0:
        raise EvaluationRunConfigError(
            f"Evaluation run config key '{key}' must be greater than 0"
        )
    return value


def _positive_int_from_mapping(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if type(value) is not int or value <= 0:
        raise EvaluationRunConfigError(
            f"Evaluation run config key '{key}' must be a positive integer"
        )
    return value


__all__ = [
    "DEFAULT_EVALUATION_RUN_CONFIG_PATH",
    "EvaluationRunConfig",
    "EvaluationRunConfigError",
    "EvaluationRunSettings",
    "JudgeConfig",
    "JudgePromptConfig",
    "load_evaluation_run_config",
]
