from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from financebench_eval_harness.config_fields import (
    REQUIRED_JUDGE_KEYS,
    REQUIRED_JUDGE_PROMPT_KEYS,
    REQUIRED_MODEL_KEYS,
    FieldReader,
)
from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import LLMGenerationConfig


DEFAULT_EVALUATION_RUN_CONFIG_PATH = Path("configs/evaluation/local_mock.yaml")
REQUIRED_EVAL_KEYS = {"dataset_path", "output_dir", "mode", "limit"}


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
                "base_url": self.model.base_url,
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
                "base_url": self.judge.model.base_url,
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

    _reader.require_keys(eval_config, REQUIRED_EVAL_KEYS, "eval")
    _reader.require_keys(model_config, REQUIRED_MODEL_KEYS, "model")

    judge_config = parse_judge_config(_reader, raw_config.get("judge"))

    return EvaluationRunConfig(
        settings=EvaluationRunSettings(
            dataset_path=_path_from_mapping(eval_config, "dataset_path"),
            output_dir=_path_from_mapping(eval_config, "output_dir"),
            mode=_reader.mode(eval_config),
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
            base_url=_optional_string_from_mapping(model_config, "base_url"),
        ),
        judge=judge_config,
    )


def parse_judge_config(
    reader: FieldReader, raw_judge: Any, section: str = "judge"
) -> JudgeConfig | None:
    """Parse an optional judge block; shared by the run, RAG run and RAG score configs."""
    if raw_judge is None:
        return None
    if not isinstance(raw_judge, dict):
        raise reader.fail(f"key '{section}' must be a mapping")

    if type(raw_judge.get("enabled")) is not bool:
        raise reader.fail(f"key '{section}.enabled' must be a boolean")
    if not raw_judge["enabled"]:
        return None

    reader.require_keys(raw_judge, REQUIRED_JUDGE_KEYS, section)

    prompt_raw = raw_judge["prompt"]
    prompt_section = f"{section}.prompt"
    if not isinstance(prompt_raw, dict):
        raise reader.fail(f"key '{prompt_section}' must be a mapping")
    reader.require_keys(prompt_raw, REQUIRED_JUDGE_PROMPT_KEYS, prompt_section)

    return JudgeConfig(
        enabled=True,
        model=LLMGenerationConfig(
            provider=reader.string(raw_judge, "provider", section),
            model_name=reader.string(raw_judge, "model_name", section),
            temperature=reader.float(raw_judge, "temperature", section),
            max_tokens=reader.positive_int(raw_judge, "max_tokens", section),
            timeout_seconds=reader.positive_float(raw_judge, "timeout_seconds", section),
            base_url=reader.optional_string(raw_judge, "base_url"),
        ),
        prompt=JudgePromptConfig(
            id=reader.string(prompt_raw, "id", prompt_section),
            version=reader.string(prompt_raw, "version", prompt_section),
            template_path=reader.path(prompt_raw, "template_path"),
        ),
    )


_reader = FieldReader(EvaluationRunConfigError, "Evaluation run config")
_path_from_mapping = _reader.path
_string_from_mapping = _reader.string
_float_from_mapping = _reader.float
_positive_float_from_mapping = _reader.positive_float
_positive_int_from_mapping = _reader.positive_int
_optional_string_from_mapping = _reader.optional_string


__all__ = [
    "DEFAULT_EVALUATION_RUN_CONFIG_PATH",
    "EvaluationRunConfig",
    "EvaluationRunConfigError",
    "EvaluationRunSettings",
    "JudgeConfig",
    "JudgePromptConfig",
    "load_evaluation_run_config",
]
