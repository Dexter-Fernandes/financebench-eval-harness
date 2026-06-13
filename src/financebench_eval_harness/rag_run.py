from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from financebench_eval_harness.evaluation import load_evaluation_config, render_prompt
from financebench_eval_harness.judge import (
    JudgeError,
    parse_judge_response,
    render_judge_prompt_for_processed_example,
    summarize_judges,
)
from financebench_eval_harness.llm import LLMClient, LLMProviderError
from financebench_eval_harness.rag_run_config import RAGRunConfig
from financebench_eval_harness.rag_types import format_retrieved_context, load_rag_inputs
from financebench_eval_harness.scoring import score_prediction, summarize_scores


@dataclass(frozen=True)
class RAGRunResult:
    """Files written by one RAG generation run."""

    output_dir: Path
    config_path: Path
    predictions_path: Path
    scores_path: Path
    run_metadata_path: Path
    judge_failures_path: Path
    example_count: int
    attempted_count: int
    success_count: int
    error_count: int


class RAGRunError(ValueError):
    """Raised when a RAG generation run cannot be completed."""


def run_rag_from_config(
    config: RAGRunConfig,
    llm_client: LLMClient,
    *,
    judge_client: LLMClient | None = None,
    run_id: str | None = None,
    limit: int | None = None,
) -> RAGRunResult:
    if config.judge is not None and judge_client is None:
        raise RAGRunError("Judge client is required when judge scoring is enabled")

    rag_inputs = load_rag_inputs(
        config.settings.retrieval_results_path,
        config.settings.examples_path,
    )
    limited_inputs = rag_inputs[:limit] if limit is not None else rag_inputs
    resolved_run_id = run_id or _timestamp_run_id()
    output_dir = config.settings.output_dir / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = perf_counter()

    config_path = output_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )

    predictions_path = output_dir / "predictions.jsonl"
    scores_path = output_dir / "scores.jsonl"
    run_metadata_path = output_dir / "run_metadata.json"
    judge_failures_path = output_dir / "judge_failures.jsonl"

    eval_config = load_evaluation_config(
        config.settings.eval_config_path,
        required_modes=frozenset({config.settings.mode.value}),
    )
    top_k = config.settings.top_k
    max_context_chars = config.settings.max_context_chars
    mode = config.settings.mode

    success_count = 0
    error_count = 0
    scores: list[dict[str, object]] = []
    judge_rows: list[dict[str, object]] = []
    judge_failures: list[dict[str, object]] = []

    with (
        predictions_path.open("w", encoding="utf-8") as predictions_file,
        scores_path.open("w", encoding="utf-8") as scores_file,
    ):
        for rag_input in limited_inputs:
            used_chunks = rag_input.retrieved_context[:top_k]
            formatted_context = format_retrieved_context(
                used_chunks, max_context_chars=max_context_chars
            )
            context_for_prompt = formatted_context if formatted_context else "(no retrieved context)"
            rendered_prompt = render_prompt(
                eval_config,
                mode,
                question=rag_input.question,
                evidence_texts=[context_for_prompt],
            )
            prediction = ""
            status = "success"
            error: str | None = None
            input_tokens: int | None = None
            output_tokens: int | None = None
            started_at = perf_counter()
            try:
                generation = llm_client.generate(rendered_prompt.text)
                prediction = generation.text
                input_tokens = generation.prompt_tokens
                output_tokens = generation.output_tokens
                success_count += 1
            except LLMProviderError as exc:
                status = "error"
                error = str(exc)
                error_count += 1
            latency_ms = int(round((perf_counter() - started_at) * 1000))

            score = score_prediction(rag_input.gold_answer, prediction)
            scores.append(score)

            judge_row: dict[str, object] | None = None
            if config.judge is not None and judge_client is not None:
                example_dict: dict[str, object] = {
                    "question": rag_input.question,
                    "gold_answer": rag_input.gold_answer,
                    "evidence": list(rag_input.gold_evidence),
                }
                judge_row = _score_with_judge(
                    config=config,
                    judge_client=judge_client,
                    example=example_dict,
                    prediction=prediction,
                )
                judge_rows.append(judge_row)
                if judge_row["status"] == "error":
                    judge_failures.append({
                        "question_id": rag_input.question_id,
                        "error": judge_row["error"],
                        "raw_response": judge_row["raw_response"],
                        "model_provider": judge_row["model_provider"],
                        "model_name": judge_row["model_name"],
                        "prompt_id": judge_row["prompt_id"],
                        "prompt_version": judge_row["prompt_version"],
                    })

            retrieved_chunk_ids = [c.chunk_id for c in used_chunks]
            prediction_row: dict[str, object] = {
                "question_id": rag_input.question_id,
                "question": rag_input.question,
                "gold_answer": rag_input.gold_answer,
                "prediction": prediction,
                "mode": rendered_prompt.mode.value,
                "model_provider": config.model.provider,
                "model_name": config.model.model_name,
                "prompt_id": rendered_prompt.prompt_id,
                "prompt_version": rendered_prompt.prompt_version,
                "prompt": rendered_prompt.text,
                "top_k": top_k,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "status": status,
                "error": error,
            }
            score_row: dict[str, object] = {
                "question_id": rag_input.question_id,
                "scores": score,
                "judge": judge_row,
                "status": status,
                "error": error,
            }
            predictions_file.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")
            predictions_file.flush()
            scores_file.write(json.dumps(score_row, ensure_ascii=False) + "\n")
            scores_file.flush()

    attempted_count = len(limited_inputs)
    if judge_failures:
        judge_failures_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in judge_failures) + "\n",
            encoding="utf-8",
        )
    elif judge_failures_path.exists():
        judge_failures_path.unlink()

    duration_ms = int(round((perf_counter() - run_started_at) * 1000))
    run_metadata: dict[str, object] = {
        "run_id": resolved_run_id,
        "output_dir": str(output_dir),
        "examples_path": str(config.settings.examples_path),
        "retrieval_results_path": str(config.settings.retrieval_results_path),
        "mode": config.settings.mode.value,
        "limit": limit,
        "top_k": top_k,
        "max_context_chars": max_context_chars,
        "model_provider": config.model.provider,
        "model_name": config.model.model_name,
        "temperature": config.model.temperature,
        "max_tokens": config.model.max_tokens,
        "timeout_seconds": config.model.timeout_seconds,
        "base_url": config.model.base_url,
        "predictions_path": str(predictions_path),
        "scores_path": str(scores_path),
        "duration_ms": duration_ms,
        "attempted_count": attempted_count,
        "success_count": success_count,
        "error_count": error_count,
        "score_summary": summarize_scores(scores),
        "judge": _judge_metadata(config),
        "judge_failures_path": str(judge_failures_path),
        "judge_summary": summarize_judges(judge_rows),
    }
    run_metadata_path.write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return RAGRunResult(
        output_dir=output_dir,
        config_path=config_path,
        predictions_path=predictions_path,
        scores_path=scores_path,
        run_metadata_path=run_metadata_path,
        judge_failures_path=judge_failures_path,
        example_count=attempted_count,
        attempted_count=attempted_count,
        success_count=success_count,
        error_count=error_count,
    )


def _score_with_judge(
    *,
    config: RAGRunConfig,
    judge_client: LLMClient,
    example: dict[str, Any],
    prediction: str,
) -> dict[str, object]:
    assert config.judge is not None
    raw_response: str | None = None
    verdict: str | None = None
    reason: str | None = None
    status = "success"
    error: str | None = None
    started_at = perf_counter()
    try:
        rendered = render_judge_prompt_for_processed_example(
            config.judge.prompt,
            example,
            prediction=prediction,
        )
        raw_response = judge_client.generate(rendered.text).text
        parsed = parse_judge_response(raw_response)
        verdict = parsed["verdict"]
        reason = parsed["reason"]
    except (JudgeError, LLMProviderError) as exc:
        status = "error"
        error = str(exc)

    latency_ms = int(round((perf_counter() - started_at) * 1000))
    return {
        "status": status,
        "verdict": verdict,
        "reason": reason,
        "error": error,
        "raw_response": raw_response,
        "model_provider": config.judge.model.provider,
        "model_name": config.judge.model.model_name,
        "prompt_id": config.judge.prompt.id,
        "prompt_version": config.judge.prompt.version,
        "latency_ms": latency_ms,
    }


def _judge_metadata(config: RAGRunConfig) -> dict[str, object]:
    if config.judge is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "provider": config.judge.model.provider,
        "model_name": config.judge.model.model_name,
        "temperature": config.judge.model.temperature,
        "max_tokens": config.judge.model.max_tokens,
        "timeout_seconds": config.judge.model.timeout_seconds,
        "base_url": config.judge.model.base_url,
        "prompt_id": config.judge.prompt.id,
        "prompt_version": config.judge.prompt.version,
        "prompt_template_path": str(config.judge.prompt.template_path),
    }


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "RAGRunError",
    "RAGRunResult",
    "run_rag_from_config",
]
