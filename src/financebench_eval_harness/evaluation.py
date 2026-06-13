from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EVALUATION_CONFIG_PATH = Path("configs/evaluation/baselines.yaml")
REQUIRED_EVALUATION_MODES = {"closed_book", "oracle_context"}


class EvaluationMode(str, Enum):
    """Supported evaluation modes (baseline and RAG)."""

    CLOSED_BOOK = "closed_book"
    ORACLE_CONTEXT = "oracle_context"
    RAG_DENSE = "rag_dense"
    RAG_ORACLE = "rag_oracle"
    RAG_NO_CONTEXT = "rag_no_context"
    RAG_HYBRID = "rag_hybrid"
    RAG_RERANKED = "rag_reranked"


VALID_CONTEXT_SOURCES = frozenset({"retrieved_chunks", "gold_evidence", "none"})


@dataclass(frozen=True)
class PromptTemplate:
    """Prompt template configured for one evaluation mode."""

    id: str
    version: str
    mode: EvaluationMode
    template_path: Path
    template: str
    context_source: str = "none"  # "retrieved_chunks" | "gold_evidence" | "none"


@dataclass(frozen=True)
class RenderedPrompt:
    """Rendered prompt text and metadata for future run artefacts."""

    text: str
    mode: EvaluationMode
    prompt_id: str
    prompt_version: str
    template_path: Path

    @property
    def run_metadata(self) -> dict[str, str]:
        return {
            "evaluation_mode": self.mode.value,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_template_path": str(self.template_path),
        }


@dataclass(frozen=True)
class EvaluationConfig:
    """Configured baseline evaluation modes and prompts."""

    modes: dict[EvaluationMode, PromptTemplate]

    def prompt_for(self, mode: EvaluationMode | str) -> PromptTemplate:
        evaluation_mode = _evaluation_mode_from_value(mode)
        try:
            return self.modes[evaluation_mode]
        except KeyError as exc:
            raise EvaluationConfigError(
                f"Evaluation mode is not configured: {evaluation_mode.value}"
            ) from exc


class EvaluationConfigError(ValueError):
    """Raised when an evaluation config cannot be loaded or validated."""


def load_evaluation_config(
    config_path: str | Path = DEFAULT_EVALUATION_CONFIG_PATH,
    required_modes: frozenset[str] | None = None,
) -> EvaluationConfig:
    path = Path(config_path)

    if not path.is_file():
        raise EvaluationConfigError(f"Evaluation config file not found: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationConfigError(
            f"Evaluation config is not valid YAML: {path}"
        ) from exc

    if not isinstance(raw_config, dict):
        raise EvaluationConfigError(f"Evaluation config must be a mapping: {path}")

    evaluation = raw_config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise EvaluationConfigError(
            f"Evaluation config must contain an 'evaluation' mapping: {path}"
        )

    modes = evaluation.get("modes")
    if not isinstance(modes, dict):
        raise EvaluationConfigError(
            f"Evaluation config must contain an 'evaluation.modes' mapping: {path}"
        )

    effective_required = REQUIRED_EVALUATION_MODES if required_modes is None else required_modes
    missing_modes = sorted(effective_required - modes.keys())
    if missing_modes:
        missing = ", ".join(missing_modes)
        raise EvaluationConfigError(
            f"Evaluation config missing required mode(s): {missing}"
        )

    return EvaluationConfig(
        modes={
            _evaluation_mode_from_value(mode): _prompt_template_from_mapping(
                mode=mode,
                raw_mode_config=raw_mode_config,
            )
            for mode, raw_mode_config in modes.items()
        }
    )


def render_prompt(
    config: EvaluationConfig,
    mode: EvaluationMode | str,
    *,
    question: str,
    evidence_texts: list[str] | tuple[str, ...] | None = None,
) -> RenderedPrompt:
    prompt_template = config.prompt_for(mode)
    clean_question = _required_text(question, "question")
    evidence_text = ""
    if prompt_template.context_source in ("retrieved_chunks", "gold_evidence"):
        evidence_text = _joined_evidence_context(evidence_texts)

    text = prompt_template.template.format(
        question=clean_question,
        evidence_text=evidence_text,
    )
    return RenderedPrompt(
        text=text,
        mode=prompt_template.mode,
        prompt_id=prompt_template.id,
        prompt_version=prompt_template.version,
        template_path=prompt_template.template_path,
    )


def render_prompt_for_processed_example(
    config: EvaluationConfig,
    mode: EvaluationMode | str,
    processed_example: dict[str, Any],
) -> RenderedPrompt:
    question = processed_example.get("question")
    evidence = processed_example.get("evidence")
    evidence_texts: list[str] = []
    if isinstance(evidence, list):
        evidence_texts = [
            item["evidence_text"]
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("evidence_text"), str)
        ]

    return render_prompt(
        config,
        mode,
        question=question if isinstance(question, str) else "",
        evidence_texts=evidence_texts,
    )


def _prompt_template_from_mapping(
    *,
    mode: str,
    raw_mode_config: Any,
) -> PromptTemplate:
    evaluation_mode = _evaluation_mode_from_value(mode)
    if not isinstance(raw_mode_config, dict):
        raise EvaluationConfigError(
            f"Evaluation mode config must be a mapping: {mode}"
        )

    prompt = raw_mode_config.get("prompt")
    if not isinstance(prompt, dict):
        raise EvaluationConfigError(
            f"Evaluation mode config must contain a 'prompt' mapping: {mode}"
        )

    prompt_id = prompt.get("id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise EvaluationConfigError(
            f"Evaluation prompt id must be a non-empty string: {mode}"
        )

    version = prompt.get("version")
    if not isinstance(version, str) or not version.strip():
        raise EvaluationConfigError(
            f"Evaluation prompt version must be a non-empty string: {mode}"
        )

    template_path = _path_from_prompt_mapping(prompt, mode)
    template = _read_prompt_template(template_path)
    if not template.strip():
        raise EvaluationConfigError(
            f"Evaluation prompt template file must be non-empty: {template_path}"
        )

    context_source = raw_mode_config.get("context_source", "none")
    if context_source not in VALID_CONTEXT_SOURCES:
        raise EvaluationConfigError(
            f"Evaluation mode config 'context_source' must be one of"
            f" {sorted(VALID_CONTEXT_SOURCES)}: {mode}"
        )

    return PromptTemplate(
        id=prompt_id,
        version=version,
        mode=evaluation_mode,
        template_path=template_path,
        template=template,
        context_source=context_source,
    )


def _evaluation_mode_from_value(mode: EvaluationMode | str) -> EvaluationMode:
    if isinstance(mode, EvaluationMode):
        return mode

    try:
        return EvaluationMode(mode)
    except ValueError as exc:
        raise EvaluationConfigError(f"Unknown evaluation mode: {mode}") from exc


def _path_from_prompt_mapping(prompt: dict[str, Any], mode: str) -> Path:
    value = prompt.get("template_path")
    if not isinstance(value, str) or not value.strip():
        raise EvaluationConfigError(
            f"Evaluation prompt template_path must be a non-empty string: {mode}"
        )
    return Path(value)


def _read_prompt_template(template_path: Path) -> str:
    if not template_path.is_file():
        raise EvaluationConfigError(
            f"Evaluation prompt template file not found: {template_path}"
        )

    try:
        return template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationConfigError(
            f"Could not read evaluation prompt template file: {template_path}"
        ) from exc


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationConfigError(
            f"Prompt render field must be non-empty: {field_name}"
        )
    return value


def _joined_evidence_context(
    evidence_texts: list[str] | tuple[str, ...] | None,
) -> str:
    clean_evidence_texts = [
        text.strip()
        for text in evidence_texts or []
        if isinstance(text, str) and text.strip()
    ]
    if not clean_evidence_texts:
        raise EvaluationConfigError(
            "Prompt render field must include at least one evidence text: evidence_texts"
        )

    return "\n\n".join(
        f"[Evidence {index}]\n{text}"
        for index, text in enumerate(clean_evidence_texts, start=1)
    )


__all__ = [
    "DEFAULT_EVALUATION_CONFIG_PATH",
    "EvaluationConfig",
    "EvaluationConfigError",
    "EvaluationMode",
    "PromptTemplate",
    "RenderedPrompt",
    "load_evaluation_config",
    "render_prompt",
    "render_prompt_for_processed_example",
]
