import json
from pathlib import Path

import yaml

from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import (
    LLMGenerationConfig,
    LLMProviderError,
    MockLLMClient,
)
from financebench_eval_harness.run import run_evaluation_from_config
from financebench_eval_harness.run_config import EvaluationRunConfig, EvaluationRunSettings


def test_run_evaluation_with_mock_llm_writes_config_snapshot_and_outputs(
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
    assert result.outputs_path == result.output_dir / "outputs.jsonl"
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
        },
    }

    rows = _read_jsonl(result.outputs_path)
    assert [row["question_id"] for row in rows] == ["q0", "q1"]
    assert [row["response"] for row in rows] == ["answer 1", "answer 2"]
    assert [row["status"] for row in rows] == ["success", "success"]
    assert [row["error"] for row in rows] == [None, None]
    assert rows[0]["mode"] == "closed_book"
    assert rows[0]["prompt_id"] == "closed_book_v1"
    assert rows[0]["prompt_version"] == "v1"
    assert rows[0]["model_provider"] == "mock"
    assert rows[0]["model_name"] == "mock-model"
    assert "Question 0?" in rows[0]["prompt"]
    assert rows[0]["gold_answer"] == "Gold answer 0"
    assert llm_client.calls == [rows[0]["prompt"], rows[1]["prompt"]]


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

    rows = _read_jsonl(result.outputs_path)
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

    rows = _read_jsonl(result.outputs_path)
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

        def generate(self, prompt: str) -> str:
            self.calls += 1
            outputs_path = tmp_path / "runs" / "streamed-run" / "outputs.jsonl"
            if self.calls == 2:
                rows = _read_jsonl(outputs_path)
                assert [row["question_id"] for row in rows] == ["q0"]
                assert rows[0]["status"] == "success"
            return f"answer {self.calls}"

    result = run_evaluation_from_config(
        config,
        InspectingLLM(),
        run_id="streamed-run",
    )

    rows = _read_jsonl(result.outputs_path)
    assert [row["question_id"] for row in rows] == ["q0", "q1"]
    assert [row["response"] for row in rows] == ["answer 1", "answer 2"]


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

        def generate(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 2:
                raise LLMProviderError("provider timeout")
            return f"answer {self.calls}"

    result = run_evaluation_from_config(
        config,
        FailingOnceLLM(),
        run_id="error-run",
    )

    rows = _read_jsonl(result.outputs_path)
    assert result.example_count == 3
    assert result.attempted_count == 3
    assert result.success_count == 2
    assert result.error_count == 1
    assert [row["status"] for row in rows] == ["success", "error", "success"]
    assert rows[1]["question_id"] == "q1"
    assert rows[1]["response"] == ""
    assert rows[1]["error"] == "provider timeout"
    assert rows[2]["question_id"] == "q2"


def _run_config(
    *,
    dataset_path: Path,
    output_dir: Path,
    mode: EvaluationMode,
    limit: int,
    model_name: str,
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
        ),
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
