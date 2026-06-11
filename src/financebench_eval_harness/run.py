from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import yaml

from financebench_eval_harness.data_common import write_jsonl_records
from financebench_eval_harness.evaluation import (
    load_evaluation_config,
    render_prompt_for_processed_example,
)
from financebench_eval_harness.llm import LLMClient
from financebench_eval_harness.run_config import EvaluationRunConfig


@dataclass(frozen=True)
class EvaluationRunResult:
    """Files written by one evaluation run."""

    output_dir: Path
    config_path: Path
    outputs_path: Path
    example_count: int


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
    output_dir = config.settings.output_dir / (run_id or _timestamp_run_id())
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )

    evaluation_config = load_evaluation_config()
    output_rows: list[dict[str, str]] = []
    for example in limited_examples:
        rendered_prompt = render_prompt_for_processed_example(
            evaluation_config,
            config.settings.mode,
            example,
        )
        response = llm_client.generate(rendered_prompt.text)
        output_rows.append(
            {
                "question_id": _string_field(example, "question_id"),
                "mode": rendered_prompt.mode.value,
                "prompt_id": rendered_prompt.prompt_id,
                "prompt_version": rendered_prompt.prompt_version,
                "model_provider": config.model.provider,
                "model_name": config.model.model_name,
                "prompt": rendered_prompt.text,
                "response": response,
                "gold_answer": _string_field(example, "gold_answer"),
            }
        )

    outputs_path = output_dir / "outputs.jsonl"
    write_jsonl_records(outputs_path, output_rows)

    return EvaluationRunResult(
        output_dir=output_dir,
        config_path=config_path,
        outputs_path=outputs_path,
        example_count=len(output_rows),
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
