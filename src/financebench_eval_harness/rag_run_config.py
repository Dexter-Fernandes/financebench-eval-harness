from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from financebench_eval_harness.config_fields import REQUIRED_MODEL_KEYS, FieldReader
from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import LLMGenerationConfig
from financebench_eval_harness.run_config import JudgeConfig, parse_judge_config


DEFAULT_RAG_RUN_CONFIG_PATH = Path("configs/evaluation/rag_dense_local.yaml")

REQUIRED_RAG_EVAL_KEYS = {"examples_path", "retrieval_results_path", "output_dir", "mode", "top_k"}


class RAGRunConfigError(ValueError):
    """Raised when a RAG run config cannot be loaded or validated."""


@dataclass(frozen=True)
class RAGEvalSettings:
    """Input paths, mode, and context-budget settings for one RAG generation run."""

    examples_path: Path
    retrieval_results_path: Path
    output_dir: Path
    mode: EvaluationMode
    top_k: int
    retrieval_run_id: str | None = None
    max_context_chars: int | None = None
    eval_config_path: Path = Path("configs/evaluation/rag_modes.yaml")


@dataclass(frozen=True)
class RAGRunConfig:
    """Full reproducible RAG generation run configuration."""

    settings: RAGEvalSettings
    model: LLMGenerationConfig
    judge: JudgeConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        rag_eval: dict[str, Any] = {
            "examples_path": str(self.settings.examples_path),
            "retrieval_results_path": str(self.settings.retrieval_results_path),
            "output_dir": str(self.settings.output_dir),
            "mode": self.settings.mode.value,
            "top_k": self.settings.top_k,
            "eval_config_path": str(self.settings.eval_config_path),
        }
        if self.settings.retrieval_run_id is not None:
            rag_eval["retrieval_run_id"] = self.settings.retrieval_run_id
        if self.settings.max_context_chars is not None:
            rag_eval["max_context_chars"] = self.settings.max_context_chars

        result: dict[str, Any] = {
            "rag_eval": rag_eval,
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
            result["judge"] = {
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
        return result


def load_rag_run_config(
    config_path: str | Path = DEFAULT_RAG_RUN_CONFIG_PATH,
) -> RAGRunConfig:
    path = Path(config_path)
    if not path.is_file():
        raise RAGRunConfigError(f"RAG run config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RAGRunConfigError(f"RAG run config is not valid YAML: {path}") from exc

    if not isinstance(raw, dict):
        raise RAGRunConfigError(f"RAG run config must be a mapping: {path}")

    rag_eval = raw.get("rag_eval")
    if not isinstance(rag_eval, dict):
        raise RAGRunConfigError(
            f"RAG run config must contain a 'rag_eval' mapping: {path}"
        )
    model_raw = raw.get("model")
    if not isinstance(model_raw, dict):
        raise RAGRunConfigError(
            f"RAG run config must contain a 'model' mapping: {path}"
        )

    _reader.require_keys(rag_eval, REQUIRED_RAG_EVAL_KEYS, "rag_eval")
    _reader.require_keys(model_raw, REQUIRED_MODEL_KEYS, "model")

    return RAGRunConfig(
        settings=RAGEvalSettings(
            examples_path=_path_from_mapping(rag_eval, "examples_path"),
            retrieval_results_path=_path_from_mapping(rag_eval, "retrieval_results_path"),
            output_dir=_path_from_mapping(rag_eval, "output_dir"),
            mode=_reader.mode(rag_eval),
            top_k=_positive_int_from_mapping(rag_eval, "top_k"),
            retrieval_run_id=_optional_string_from_mapping(rag_eval, "retrieval_run_id"),
            max_context_chars=_optional_positive_int_from_mapping(rag_eval, "max_context_chars"),
            eval_config_path=Path(
                rag_eval.get("eval_config_path", "configs/evaluation/rag_modes.yaml")
            ),
        ),
        model=LLMGenerationConfig(
            provider=_string_from_mapping(model_raw, "provider"),
            model_name=_string_from_mapping(model_raw, "model_name"),
            temperature=_float_from_mapping(model_raw, "temperature"),
            max_tokens=_positive_int_from_mapping(model_raw, "max_tokens"),
            timeout_seconds=_positive_float_from_mapping(model_raw, "timeout_seconds"),
            base_url=_optional_string_from_mapping(model_raw, "base_url"),
        ),
        judge=parse_judge_config(_reader, raw.get("judge")),
    )


_reader = FieldReader(RAGRunConfigError, "RAG run config")
_path_from_mapping = _reader.path
_string_from_mapping = _reader.string
_float_from_mapping = _reader.float
_positive_float_from_mapping = _reader.positive_float
_positive_int_from_mapping = _reader.positive_int
_optional_string_from_mapping = _reader.optional_string
_optional_positive_int_from_mapping = _reader.optional_positive_int


__all__ = [
    "DEFAULT_RAG_RUN_CONFIG_PATH",
    "RAGEvalSettings",
    "RAGRunConfig",
    "RAGRunConfigError",
    "load_rag_run_config",
]
