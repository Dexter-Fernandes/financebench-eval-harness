from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from financebench_eval_harness.llm import LLMGenerationConfig
from financebench_eval_harness.run_config import JudgeConfig, JudgePromptConfig


DEFAULT_RAG_SCORE_CONFIG_PATH = Path("configs/evaluation/rag_score_local.yaml")

REQUIRED_RAG_SCORE_KEYS = {"run_dir"}
REQUIRED_MODEL_KEYS = {"provider", "model_name", "temperature", "max_tokens", "timeout_seconds"}
REQUIRED_JUDGE_KEYS = REQUIRED_MODEL_KEYS | {"enabled", "prompt"}
REQUIRED_JUDGE_PROMPT_KEYS = {"id", "version", "template_path"}


class RAGScoreConfigError(ValueError):
    """Raised when a RAG score config cannot be loaded or validated."""


@dataclass(frozen=True)
class RAGScoreSettings:
    run_dir: Path
    retrieval_results_path: Path | None = None
    output_dir: Path | None = None
    k: int = 5


@dataclass(frozen=True)
class RAGScoreConfig:
    settings: RAGScoreSettings
    answer_judge: JudgeConfig | None = None
    grounding_judge: JudgeConfig | None = None


def load_rag_score_config(
    config_path: str | Path = DEFAULT_RAG_SCORE_CONFIG_PATH,
) -> RAGScoreConfig:
    path = Path(config_path)
    if not path.is_file():
        raise RAGScoreConfigError(f"RAG score config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RAGScoreConfigError(f"RAG score config is not valid YAML: {path}") from exc

    if not isinstance(raw, dict):
        raise RAGScoreConfigError(f"RAG score config must be a mapping: {path}")

    rag_score = raw.get("rag_score")
    if not isinstance(rag_score, dict):
        raise RAGScoreConfigError(
            f"RAG score config must contain a 'rag_score' mapping: {path}"
        )

    _validate_required_keys(rag_score, REQUIRED_RAG_SCORE_KEYS, "rag_score")

    k_raw = rag_score.get("k", 5)
    if type(k_raw) is not int or k_raw <= 0:
        raise RAGScoreConfigError("RAG score config key 'rag_score.k' must be a positive integer")

    return RAGScoreConfig(
        settings=RAGScoreSettings(
            run_dir=_path_from_mapping(rag_score, "run_dir"),
            retrieval_results_path=_optional_path_from_mapping(rag_score, "retrieval_results_path"),
            output_dir=_optional_path_from_mapping(rag_score, "output_dir"),
            k=k_raw,
        ),
        answer_judge=_judge_config_from_mapping(raw.get("answer_judge"), section="answer_judge"),
        grounding_judge=_judge_config_from_mapping(raw.get("grounding_judge"), section="grounding_judge"),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_required_keys(
    mapping: dict[str, Any],
    required: set[str],
    section: str,
) -> None:
    missing = sorted(required - mapping.keys())
    if missing:
        raise RAGScoreConfigError(
            f"RAG score config missing required {section} key(s): {', '.join(missing)}"
        )


def _judge_config_from_mapping(raw_judge: Any, *, section: str) -> JudgeConfig | None:
    if raw_judge is None:
        return None
    if not isinstance(raw_judge, dict):
        raise RAGScoreConfigError(f"RAG score config key '{section}' must be a mapping")

    enabled = raw_judge.get("enabled")
    if type(enabled) is not bool:
        raise RAGScoreConfigError(
            f"RAG score config key '{section}.enabled' must be a boolean"
        )
    if not enabled:
        return None

    _validate_required_keys(raw_judge, REQUIRED_JUDGE_KEYS, section)

    prompt_raw = raw_judge["prompt"]
    if not isinstance(prompt_raw, dict):
        raise RAGScoreConfigError(
            f"RAG score config key '{section}.prompt' must be a mapping"
        )
    _validate_required_keys(prompt_raw, REQUIRED_JUDGE_PROMPT_KEYS, f"{section}.prompt")

    return JudgeConfig(
        enabled=True,
        model=LLMGenerationConfig(
            provider=_string_from_mapping(raw_judge, "provider", section),
            model_name=_string_from_mapping(raw_judge, "model_name", section),
            temperature=_float_from_mapping(raw_judge, "temperature", section),
            max_tokens=_positive_int_from_mapping(raw_judge, "max_tokens", section),
            timeout_seconds=_positive_float_from_mapping(raw_judge, "timeout_seconds", section),
            base_url=_optional_string_from_mapping(raw_judge, "base_url"),
        ),
        prompt=JudgePromptConfig(
            id=_string_from_mapping(prompt_raw, "id", f"{section}.prompt"),
            version=_string_from_mapping(prompt_raw, "version", f"{section}.prompt"),
            template_path=_path_from_mapping(prompt_raw, "template_path"),
        ),
    )


def _path_from_mapping(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RAGScoreConfigError(f"RAG score config key '{key}' must be a non-empty string")
    return Path(value)


def _optional_path_from_mapping(config: dict[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RAGScoreConfigError(f"RAG score config key '{key}' must be a non-empty string")
    return Path(value)


def _optional_string_from_mapping(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RAGScoreConfigError(f"RAG score config key '{key}' must be a non-empty string")
    return value


def _string_from_mapping(config: dict[str, Any], key: str, section: str = "") -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        prefix = f"{section}." if section else ""
        raise RAGScoreConfigError(
            f"RAG score config key '{prefix}{key}' must be a non-empty string"
        )
    return value


def _float_from_mapping(config: dict[str, Any], key: str, section: str = "") -> float:
    value = config.get(key)
    if type(value) not in (float, int):
        prefix = f"{section}." if section else ""
        raise RAGScoreConfigError(f"RAG score config key '{prefix}{key}' must be a number")
    return float(value)


def _positive_float_from_mapping(config: dict[str, Any], key: str, section: str = "") -> float:
    value = _float_from_mapping(config, key, section)
    if value <= 0:
        prefix = f"{section}." if section else ""
        raise RAGScoreConfigError(
            f"RAG score config key '{prefix}{key}' must be greater than 0"
        )
    return value


def _positive_int_from_mapping(config: dict[str, Any], key: str, section: str = "") -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        prefix = f"{section}." if section else ""
        raise RAGScoreConfigError(
            f"RAG score config key '{prefix}{key}' must be a positive integer"
        )
    return value


__all__ = [
    "DEFAULT_RAG_SCORE_CONFIG_PATH",
    "RAGScoreConfig",
    "RAGScoreConfigError",
    "RAGScoreSettings",
    "load_rag_score_config",
]
