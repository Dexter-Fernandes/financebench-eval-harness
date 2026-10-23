from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from financebench_eval_harness.config_fields import FieldReader
from financebench_eval_harness.run_config import JudgeConfig, parse_judge_config


DEFAULT_RAG_SCORE_CONFIG_PATH = Path("configs/evaluation/rag_score_local.yaml")

REQUIRED_RAG_SCORE_KEYS = {"run_dir"}


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

    _reader.require_keys(rag_score, REQUIRED_RAG_SCORE_KEYS, "rag_score")

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
        answer_judge=parse_judge_config(_reader, raw.get("answer_judge"), "answer_judge"),
        grounding_judge=parse_judge_config(_reader, raw.get("grounding_judge"), "grounding_judge"),
    )


_reader = FieldReader(RAGScoreConfigError, "RAG score config")
_path_from_mapping = _reader.path
_optional_path_from_mapping = _reader.optional_path


__all__ = [
    "DEFAULT_RAG_SCORE_CONFIG_PATH",
    "RAGScoreConfig",
    "RAGScoreConfigError",
    "RAGScoreSettings",
    "load_rag_score_config",
]
