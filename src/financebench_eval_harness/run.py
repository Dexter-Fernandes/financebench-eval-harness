from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from financebench_eval_harness.evaluation import (
    load_evaluation_config,
    render_prompt_for_processed_example,
)
from financebench_eval_harness.llm import LLMClient, LLMProviderError
from financebench_eval_harness.run_config import EvaluationRunConfig


@dataclass(frozen=True)
class EvaluationRunResult:
    """Files written by one evaluation run."""

    output_dir: Path
    config_path: Path
    outputs_path: Path
    run_metadata_path: Path
    example_count: int
    attempted_count: int
    success_count: int
    error_count: int


class EvaluationRunError(ValueError):
    """Raised when an evaluation run cannot be completed."""


def run_evaluation_from_config(
    config: EvaluationRunConfig,
    llm_client: LLMClient,
    *,
    run_id: str | None = None,
) -> EvaluationRunResult:
    examples = _load_processed_examples_from_path(config.settings.dataset_path)
    limited_examples = examples[: config.settings.limit]
    resolved_run_id = run_id or _timestamp_run_id()
    output_dir = config.settings.output_dir / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = perf_counter()

    config_path = output_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )

    outputs_path = output_dir / "outputs.jsonl"
    run_metadata_path = output_dir / "run_metadata.json"
    evaluation_config = load_evaluation_config()
    success_count = 0
    error_count = 0
    with outputs_path.open("w", encoding="utf-8") as outputs_file:
        for example in limited_examples:
            rendered_prompt = render_prompt_for_processed_example(
                evaluation_config,
                config.settings.mode,
                example,
            )
            question = _string_field(example, "question")
            prediction = ""
            status = "success"
            error: str | None = None
            started_at = perf_counter()
            try:
                prediction = llm_client.generate(rendered_prompt.text)
                success_count += 1
            except LLMProviderError as exc:
                status = "error"
                error = str(exc)
                error_count += 1
            latency_ms = int(round((perf_counter() - started_at) * 1000))

            output_row = {
                "question_id": _string_field(example, "question_id"),
                "question": question,
                "gold_answer": _string_field(example, "gold_answer"),
                "prediction": prediction,
                "mode": rendered_prompt.mode.value,
                "model_provider": config.model.provider,
                "model_name": config.model.model_name,
                "prompt_id": rendered_prompt.prompt_id,
                "prompt_version": rendered_prompt.prompt_version,
                "prompt": rendered_prompt.text,
                "latency_ms": latency_ms,
                "input_tokens": None,
                "output_tokens": None,
                "status": status,
                "error": error,
            }
            outputs_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            outputs_file.flush()

    attempted_count = len(limited_examples)
    duration_ms = int(round((perf_counter() - run_started_at) * 1000))
    run_metadata = {
        "run_id": resolved_run_id,
        "output_dir": str(output_dir),
        "dataset_path": str(config.settings.dataset_path),
        "mode": config.settings.mode.value,
        "limit": config.settings.limit,
        "model_provider": config.model.provider,
        "model_name": config.model.model_name,
        "temperature": config.model.temperature,
        "max_tokens": config.model.max_tokens,
        "timeout_seconds": config.model.timeout_seconds,
        "outputs_path": str(outputs_path),
        "output_filename": outputs_path.name,
        "duration_ms": duration_ms,
        "attempted_count": attempted_count,
        "success_count": success_count,
        "error_count": error_count,
    }
    run_metadata_path.write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return EvaluationRunResult(
        output_dir=output_dir,
        config_path=config_path,
        outputs_path=outputs_path,
        run_metadata_path=run_metadata_path,
        example_count=attempted_count,
        attempted_count=attempted_count,
        success_count=success_count,
        error_count=error_count,
    )


def _load_processed_examples_from_path(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationRunError(f"Evaluation dataset file not found: {path}")

    examples: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as examples_file:
            for line_number, line in enumerate(examples_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except JSONDecodeError as exc:
                    raise EvaluationRunError(
                        f"Invalid evaluation dataset JSONL at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise EvaluationRunError(
                        f"Invalid evaluation dataset row at line {line_number}: expected object"
                    )
                examples.append(row)
    except OSError as exc:
        raise EvaluationRunError(f"Could not read evaluation dataset file: {path}") from exc

    return examples


def _string_field(row: dict[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str):
        raise EvaluationRunError(
            f"Evaluation dataset row missing string field: {field_name}"
        )
    return value


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "EvaluationRunError",
    "EvaluationRunResult",
    "run_evaluation_from_config",
]
