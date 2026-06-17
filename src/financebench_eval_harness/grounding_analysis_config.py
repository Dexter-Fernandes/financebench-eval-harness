"""M7: Config loader for grounding analysis (analyze-grounding command)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from financebench_eval_harness.run_config import JudgeConfig, JudgePromptConfig
from financebench_eval_harness.llm import LLMGenerationConfig


DEFAULT_GROUNDING_ANALYSIS_CONFIG_PATH = Path("configs/grounding_analysis.yaml")


class GroundingAnalysisConfigError(ValueError):
    """Raised when a grounding analysis config cannot be loaded or validated."""


@dataclass(frozen=True)
class GroundingAnalysisSettings:
    run_dir: Path
    retrieval_results_path: Path | None = None
    examples_path: Path | None = None
    output_dir: Path | None = None
    k: int = 5


@dataclass(frozen=True)
class GroundingAnalysisConfig:
    settings: GroundingAnalysisSettings
    grounding_judge: JudgeConfig | None = None


def load_grounding_analysis_config(
    config_path: str | Path = DEFAULT_GROUNDING_ANALYSIS_CONFIG_PATH,
) -> GroundingAnalysisConfig:
    path = Path(config_path)
    if not path.is_file():
        raise GroundingAnalysisConfigError(
            f"Grounding analysis config file not found: {path}"
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GroundingAnalysisConfigError(
            f"Grounding analysis config is not valid YAML: {path}"
        ) from exc

    if not isinstance(raw, dict):
        raise GroundingAnalysisConfigError(
            f"Grounding analysis config must be a mapping: {path}"
        )

    section = raw.get("grounding_analysis")
    if not isinstance(section, dict):
        raise GroundingAnalysisConfigError(
            f"Config must contain a 'grounding_analysis' mapping: {path}"
        )

    run_dir_raw = section.get("run_dir")
    if not run_dir_raw:
        raise GroundingAnalysisConfigError(
            f"'grounding_analysis.run_dir' is required: {path}"
        )

    settings = GroundingAnalysisSettings(
        run_dir=Path(run_dir_raw),
        retrieval_results_path=Path(section["retrieval_results_path"])
        if section.get("retrieval_results_path")
        else None,
        examples_path=Path(section["examples_path"])
        if section.get("examples_path")
        else None,
        output_dir=Path(section["output_dir"]) if section.get("output_dir") else None,
        k=int(section.get("k", 5)),
    )

    judge_config: JudgeConfig | None = None
    judge_raw = raw.get("grounding_judge")
    if isinstance(judge_raw, dict) and judge_raw.get("enabled", False):
        model_raw = judge_raw.get("model") or judge_raw
        prompt_raw = judge_raw.get("prompt", {})
        judge_config = JudgeConfig(
            enabled=True,
            model=LLMGenerationConfig(
                provider=str(model_raw.get("provider", "ollama")),
                model_name=str(model_raw.get("model_name", "llama3.2:3b")),
                temperature=float(model_raw.get("temperature", 0.0)),
                max_tokens=int(model_raw.get("max_tokens", 512)),
                timeout_seconds=int(model_raw.get("timeout_seconds", 90)),
                base_url=str(model_raw.get("base_url", "http://localhost:11434")),
            ),
            prompt=JudgePromptConfig(
                id=str(prompt_raw.get("id", "answer_grounding_v2")),
                version=str(prompt_raw.get("version", "v2")),
                template_path=Path(
                    prompt_raw.get(
                        "template_path",
                        "prompts/judges/answer_grounding_v2.txt",
                    )
                ),
            ),
        )

    return GroundingAnalysisConfig(settings=settings, grounding_judge=judge_config)


__all__ = [
    "GroundingAnalysisConfig",
    "GroundingAnalysisConfigError",
    "GroundingAnalysisSettings",
    "load_grounding_analysis_config",
]
