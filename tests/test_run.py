import json
from pathlib import Path

import yaml

from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import (
    LLMGenerationResult,
    LLMGenerationConfig,
    LLMProviderError,
    MockLLMClient,
)
from financebench_eval_harness.run import run_evaluation_from_config
from financebench_eval_harness.run_config import EvaluationRunConfig, EvaluationRunSettings
from financebench_eval_harness.run_config import JudgeConfig, JudgePromptConfig


def test_run_evaluation_with_mock_llm_writes_config_snapshot_predictions_and_scores(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=2)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.CLOSED_BOOK,
        limit=2,
        model_name="mock-model",
    )
    llm_client = MockLLMClient(config.model, responses=["answer 1", "answer 2"])

    result = run_evaluation_from_config(config, llm_client, run_id="fixed-run")

    assert result.output_dir == tmp_path / "runs" / "fixed-run"
    assert result.config_path == result.output_dir / "config.yaml"
    assert result.predictions_path == result.output_dir / "predictions.jsonl"
    assert result.scores_path == result.output_dir / "scores.jsonl"
    assert not (result.output_dir / "outputs.jsonl").exists()
    assert result.run_metadata_path == result.output_dir / "run_metadata.json"
    assert result.example_count == 2
    assert result.attempted_count == 2
    assert result.success_count == 2
    assert result.error_count == 0

    snapshot = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert snapshot == {
        "eval": {
            "dataset_path": str(dataset_path),
            "output_dir": str(tmp_path / "runs"),
            "mode": "closed_book",
            "limit": 2,
        },
        "model": {
            "provider": "mock",
            "model_name": "mock-model",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": 30.0,
            "base_url": None,
        },
    }

    rows = _read_jsonl(result.predictions_path)
    score_rows = _read_jsonl(result.scores_path)
    assert [row["question_id"] for row in rows] == ["q0", "q1"]
    assert [row["question"] for row in rows] == ["Question 0?", "Question 1?"]
    assert [row["prediction"] for row in rows] == ["answer 1", "answer 2"]
    assert [row["status"] for row in rows] == ["success", "success"]
    assert [row["error"] for row in rows] == [None, None]
    assert "response" not in rows[0]
    assert rows[0]["mode"] == "closed_book"
    assert rows[0]["prompt_id"] == "closed_book_v1"
    assert rows[0]["prompt_version"] == "v1"
    assert rows[0]["model_provider"] == "mock"
    assert rows[0]["model_name"] == "mock-model"
    assert isinstance(rows[0]["latency_ms"], int)
    assert rows[0]["latency_ms"] >= 0
    assert rows[0]["input_tokens"] is None
    assert rows[0]["output_tokens"] is None
    assert score_rows[0] == {
        "question_id": "q0",
        "scores": {
            "exact_match": False,
            "normalized_string_match": False,
            "contains_gold_answer": False,
            "numeric_match": False,
            "unit_match": False,
            "gold_numeric_values": [0.0],
            "prediction_numeric_values": [1.0],
        },
        "judge": None,
        "status": "success",
        "error": None,
    }
    assert score_rows[1]["scores"] == {
        "exact_match": False,
        "normalized_string_match": False,
        "contains_gold_answer": False,
        "numeric_match": False,
        "unit_match": False,
        "gold_numeric_values": [1.0],
        "prediction_numeric_values": [2.0],
    }
    assert "Question 0?" in rows[0]["prompt"]
    assert rows[0]["gold_answer"] == "Gold answer 0"
    assert llm_client.calls == [rows[0]["prompt"], rows[1]["prompt"]]

    run_metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
    assert run_metadata["run_id"] == "fixed-run"
    assert run_metadata["output_dir"] == str(result.output_dir)
    assert run_metadata["dataset_path"] == str(dataset_path)
    assert run_metadata["mode"] == "closed_book"
    assert run_metadata["limit"] == 2
    assert run_metadata["model_provider"] == "mock"
    assert run_metadata["model_name"] == "mock-model"
    assert run_metadata["temperature"] == 0.0
    assert run_metadata["max_tokens"] == 512
    assert run_metadata["timeout_seconds"] == 30.0
    assert run_metadata["base_url"] is None
    assert run_metadata["predictions_path"] == str(result.predictions_path)
    assert run_metadata["scores_path"] == str(result.scores_path)
    assert run_metadata["prediction_filename"] == "predictions.jsonl"
    assert run_metadata["scores_filename"] == "scores.jsonl"
    assert isinstance(run_metadata["duration_ms"], int)
    assert run_metadata["duration_ms"] >= 0
    assert run_metadata["attempted_count"] == 2
    assert run_metadata["success_count"] == 2
    assert run_metadata["error_count"] == 0
    assert run_metadata["score_summary"] == {
        "example_count": 2,
        "exact_match_count": 0,
        "exact_match_rate": 0.0,
        "normalized_string_match_count": 0,
        "normalized_string_match_rate": 0.0,
        "contains_gold_answer_count": 0,
        "contains_gold_answer_rate": 0.0,
        "numeric_match_count": 0,
        "numeric_match_rate": 0.0,
        "unit_match_count": 0,
        "unit_match_rate": 0.0,
    }


def test_run_evaluation_config_changes_mode_and_model_metadata(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=1)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.ORACLE_CONTEXT,
        limit=1,
        model_name="changed-model",
    )
    llm_client = MockLLMClient(config.model, responses=["oracle answer"])

    result = run_evaluation_from_config(config, llm_client, run_id="oracle-run")

    rows = _read_jsonl(result.predictions_path)
    assert rows[0]["mode"] == "oracle_context"
    assert rows[0]["prompt_id"] == "oracle_context_v1"
    assert rows[0]["model_name"] == "changed-model"
    assert "Evidence 0" in rows[0]["prompt"]


def test_run_evaluation_limit_caps_examples_deterministically(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=3)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.CLOSED_BOOK,
        limit=1,
        model_name="mock-model",
    )
    llm_client = MockLLMClient(config.model, responses=["answer 1"])

    result = run_evaluation_from_config(config, llm_client, run_id="limited-run")

    rows = _read_jsonl(result.predictions_path)
    assert result.example_count == 1
    assert [row["question_id"] for row in rows] == ["q0"]


def test_run_evaluation_writes_each_output_before_next_generation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=2)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.CLOSED_BOOK,
        limit=2,
        model_name="mock-model",
    )

    class InspectingLLM:
        def __init__(self) -> None:
            self.config = config.model
            self.calls = 0

        def generate(self, prompt: str) -> LLMGenerationResult:
            self.calls += 1
            predictions_path = tmp_path / "runs" / "streamed-run" / "predictions.jsonl"
            if self.calls == 2:
                rows = _read_jsonl(predictions_path)
                assert [row["question_id"] for row in rows] == ["q0"]
                assert rows[0]["status"] == "success"
            return LLMGenerationResult(
                text=f"answer {self.calls}",
                prompt_tokens=None,
                output_tokens=None,
            )

    result = run_evaluation_from_config(
        config,
        InspectingLLM(),
        run_id="streamed-run",
    )

    rows = _read_jsonl(result.predictions_path)
    assert [row["question_id"] for row in rows] == ["q0", "q1"]
    assert [row["prediction"] for row in rows] == ["answer 1", "answer 2"]


def test_run_evaluation_records_llm_error_and_continues(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=3)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.CLOSED_BOOK,
        limit=3,
        model_name="mock-model",
    )

    class FailingOnceLLM:
        def __init__(self) -> None:
            self.config = config.model
            self.calls = 0

        def generate(self, prompt: str) -> LLMGenerationResult:
            self.calls += 1
            if self.calls == 2:
                raise LLMProviderError("provider timeout")
            return LLMGenerationResult(
                text=f"answer {self.calls}",
                prompt_tokens=None,
                output_tokens=None,
            )

    result = run_evaluation_from_config(
        config,
        FailingOnceLLM(),
        run_id="error-run",
    )

    rows = _read_jsonl(result.predictions_path)
    score_rows = _read_jsonl(result.scores_path)
    assert result.example_count == 3
    assert result.attempted_count == 3
    assert result.success_count == 2
    assert result.error_count == 1
    assert [row["status"] for row in rows] == ["success", "error", "success"]
    assert rows[1]["question_id"] == "q1"
    assert rows[1]["question"] == "Question 1?"
    assert rows[1]["prediction"] == ""
    assert rows[1]["error"] == "provider timeout"
    assert isinstance(rows[1]["latency_ms"], int)
    assert rows[1]["latency_ms"] >= 0
    assert rows[1]["input_tokens"] is None
    assert rows[1]["output_tokens"] is None
    assert score_rows[1]["scores"]["exact_match"] is False
    assert score_rows[1]["scores"]["numeric_match"] is False
    assert rows[2]["question_id"] == "q2"


def test_run_evaluation_records_provider_token_counts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=1)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.CLOSED_BOOK,
        limit=1,
        model_name="ollama-model",
    )

    class TokenReportingLLM:
        def __init__(self) -> None:
            self.config = config.model

        def generate(self, prompt: str) -> LLMGenerationResult:
            return LLMGenerationResult(
                text="answer with usage",
                prompt_tokens=42,
                output_tokens=11,
            )

    result = run_evaluation_from_config(
        config,
        TokenReportingLLM(),
        run_id="token-run",
    )

    rows = _read_jsonl(result.predictions_path)
    assert rows[0]["prediction"] == "answer with usage"
    assert rows[0]["input_tokens"] == 42
    assert rows[0]["output_tokens"] == 11


def test_run_evaluation_does_not_crash_on_duplicate_minus_numeric_output(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=1)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.ORACLE_CONTEXT,
        limit=1,
        model_name="ollama-model",
    )

    class DuplicateMinusLLM:
        def __init__(self) -> None:
            self.config = config.model

        def generate(self, prompt: str) -> LLMGenerationResult:
            return LLMGenerationResult(
                text="Translation was --4.6% and total sales change was (3.9)%.",
                prompt_tokens=None,
                output_tokens=None,
            )

    result = run_evaluation_from_config(
        config,
        DuplicateMinusLLM(),
        run_id="duplicate-minus-run",
    )

    score_rows = _read_jsonl(result.scores_path)
    assert score_rows[0]["status"] == "success"
    assert score_rows[0]["scores"]["prediction_numeric_values"] == [-4.6, -3.9]


def test_run_evaluation_attaches_successful_judge_result(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=1)
    prompt_path = _write_judge_prompt(tmp_path)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.CLOSED_BOOK,
        limit=1,
        model_name="mock-model",
        judge=_judge_config(prompt_path),
    )
    llm_client = MockLLMClient(config.model, responses=["Gold answer 0"])
    judge_client = MockLLMClient(
        config.judge.model,
        responses=['{"verdict": "correct", "reason": "Matches the gold answer."}'],
    )

    result = run_evaluation_from_config(
        config,
        llm_client,
        judge_client=judge_client,
        run_id="judge-run",
    )

    rows = _read_jsonl(result.predictions_path)
    score_rows = _read_jsonl(result.scores_path)
    assert "judge" not in rows[0]
    assert score_rows[0]["judge"] == {
        "status": "success",
        "verdict": "correct",
        "reason": "Matches the gold answer.",
        "error": None,
        "raw_response": '{"verdict": "correct", "reason": "Matches the gold answer."}',
        "model_provider": "mock",
        "model_name": "mock-judge",
        "prompt_id": "answer_correctness_v1",
        "prompt_version": "v1",
        "prompt_template_path": str(prompt_path),
        "latency_ms": score_rows[0]["judge"]["latency_ms"],
    }
    assert score_rows[0]["judge"]["latency_ms"] >= 0
    assert "Question 0?" in judge_client.calls[0]
    assert "Gold answer 0" in judge_client.calls[0]
    assert "[Evidence 1]\nEvidence 0" in judge_client.calls[0]
    assert not result.judge_failures_path.exists()

    run_metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
    assert run_metadata["judge"] == {
        "enabled": True,
        "provider": "mock",
        "model_name": "mock-judge",
        "temperature": 0.0,
        "max_tokens": 256,
        "timeout_seconds": 30.0,
        "base_url": None,
        "prompt_id": "answer_correctness_v1",
        "prompt_version": "v1",
        "prompt_template_path": str(prompt_path),
    }
    assert run_metadata["judge_failures_path"] == str(result.judge_failures_path)
    assert run_metadata["judge_summary"] == {
        "attempted_count": 1,
        "success_count": 1,
        "error_count": 0,
        "correct_count": 1,
        "correct_rate": 1.0,
        "partially_correct_count": 0,
        "partially_correct_rate": 0.0,
        "incorrect_count": 0,
        "incorrect_rate": 0.0,
        "not_answered_count": 0,
        "not_answered_rate": 0.0,
    }


def test_run_evaluation_logs_invalid_judge_output_and_continues(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, count=1)
    prompt_path = _write_judge_prompt(tmp_path)
    config = _run_config(
        dataset_path=dataset_path,
        output_dir=tmp_path / "runs",
        mode=EvaluationMode.CLOSED_BOOK,
        limit=1,
        model_name="mock-model",
        judge=_judge_config(prompt_path),
    )
    llm_client = MockLLMClient(config.model, responses=["not sure"])
    judge_client = MockLLMClient(config.judge.model, responses=["not json"])

    result = run_evaluation_from_config(
        config,
        llm_client,
        judge_client=judge_client,
        run_id="judge-error-run",
    )

    rows = _read_jsonl(result.predictions_path)
    score_rows = _read_jsonl(result.scores_path)
    assert rows[0]["status"] == "success"
    assert score_rows[0]["judge"]["status"] == "error"
    assert score_rows[0]["judge"]["verdict"] is None
    assert score_rows[0]["judge"]["reason"] is None
    assert score_rows[0]["judge"]["raw_response"] == "not json"
    assert "Judge response was not valid JSON" in score_rows[0]["judge"]["error"]

    failures = _read_jsonl(result.judge_failures_path)
    assert failures == [
        {
            "question_id": "q0",
            "error": score_rows[0]["judge"]["error"],
            "raw_response": "not json",
            "model_provider": "mock",
            "model_name": "mock-judge",
            "prompt_id": "answer_correctness_v1",
            "prompt_version": "v1",
            "prompt_template_path": str(prompt_path),
        }
    ]

    run_metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
    assert run_metadata["judge_summary"]["attempted_count"] == 1
    assert run_metadata["judge_summary"]["success_count"] == 0
    assert run_metadata["judge_summary"]["error_count"] == 1


def _run_config(
    *,
    dataset_path: Path,
    output_dir: Path,
    mode: EvaluationMode,
    limit: int,
    model_name: str,
    judge: JudgeConfig | None = None,
) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        settings=EvaluationRunSettings(
            dataset_path=dataset_path,
            output_dir=output_dir,
            mode=mode,
            limit=limit,
        ),
        model=LLMGenerationConfig(
            provider="mock",
            model_name=model_name,
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=30.0,
            base_url=None,
        ),
        judge=judge,
    )


def _write_processed_examples(path: Path, *, count: int) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for index in range(count):
            output_file.write(
                json.dumps(
                    {
                        "question_id": f"q{index}",
                        "question": f"Question {index}?",
                        "gold_answer": f"Gold answer {index}",
                        "evidence": [
                            {"evidence_text": f"Evidence {index}"},
                        ],
                    }
                )
                + "\n"
            )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_judge_prompt(tmp_path: Path) -> Path:
    prompt_path = tmp_path / "judge_prompt.txt"
    prompt_path.write_text(
        "\n".join(
            [
                "Question:",
                "{question}",
                "Gold answer:",
                "{gold_answer}",
                "Prediction:",
                "{prediction}",
                "Evidence:",
                "{evidence_text}",
            ]
        ),
        encoding="utf-8",
    )
    return prompt_path


def _judge_config(prompt_path: Path) -> JudgeConfig:
    return JudgeConfig(
        enabled=True,
        model=LLMGenerationConfig(
            provider="mock",
            model_name="mock-judge",
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=30.0,
        ),
        prompt=JudgePromptConfig(
            id="answer_correctness_v1",
            version="v1",
            template_path=prompt_path,
        ),
    )
