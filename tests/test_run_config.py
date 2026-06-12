from pathlib import Path

import pytest

from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import LLMGenerationConfig
from financebench_eval_harness.run_config import (
    DEFAULT_EVALUATION_RUN_CONFIG_PATH,
    EvaluationRunConfig,
    EvaluationRunConfigError,
    EvaluationRunSettings,
    JudgeConfig,
    JudgePromptConfig,
    load_evaluation_run_config,
)


def test_load_evaluation_run_config_reads_default_local_mock_config() -> None:
    config = load_evaluation_run_config()

    assert DEFAULT_EVALUATION_RUN_CONFIG_PATH == Path(
        "configs/evaluation/local_mock.yaml"
    )
    assert config.settings == EvaluationRunSettings(
        dataset_path=Path("data/processed/financebench/examples.jsonl"),
        output_dir=Path("runs"),
        mode=EvaluationMode.CLOSED_BOOK,
        limit=20,
    )
    assert config.model == LLMGenerationConfig(
        provider="mock",
        model_name="mock-llm",
        temperature=0.0,
        max_tokens=512,
        timeout_seconds=30.0,
        base_url=None,
    )
    assert config.judge == JudgeConfig(
        enabled=True,
        model=LLMGenerationConfig(
            provider="mock",
            model_name="mock-judge",
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=30.0,
            base_url=None,
        ),
        prompt=JudgePromptConfig(
            id="answer_correctness_v1",
            version="v1",
            template_path=Path("prompts/judges/answer_correctness_v1.txt"),
        ),
    )


def test_load_evaluation_run_config_reads_custom_model_and_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                f"  dataset_path: {tmp_path / 'examples.jsonl'}",
                f"  output_dir: {tmp_path / 'runs'}",
                "  mode: oracle_context",
                "  limit: 2",
                "model:",
                "  provider: ollama",
                "  model_name: custom-model",
                "  temperature: 0.3",
                "  max_tokens: 128",
                "  timeout_seconds: 5",
                "  base_url: http://localhost:11434",
            ]
        ),
        encoding="utf-8",
    )

    config = load_evaluation_run_config(config_path)

    assert config == EvaluationRunConfig(
        settings=EvaluationRunSettings(
            dataset_path=tmp_path / "examples.jsonl",
            output_dir=tmp_path / "runs",
            mode=EvaluationMode.ORACLE_CONTEXT,
            limit=2,
        ),
        model=LLMGenerationConfig(
            provider="ollama",
            model_name="custom-model",
            temperature=0.3,
            max_tokens=128,
            timeout_seconds=5.0,
            base_url="http://localhost:11434",
        ),
        judge=None,
    )


def test_load_evaluation_run_config_reads_enabled_judge_config(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                f"  dataset_path: {tmp_path / 'examples.jsonl'}",
                f"  output_dir: {tmp_path / 'runs'}",
                "  mode: closed_book",
                "  limit: 1",
                "model:",
                "  provider: mock",
                "  model_name: mock-llm",
                "  temperature: 0.0",
                "  max_tokens: 512",
                "  timeout_seconds: 30",
                "judge:",
                "  enabled: true",
                "  provider: ollama",
                "  model_name: judge-model",
                "  temperature: 0.0",
                "  max_tokens: 256",
                "  timeout_seconds: 10",
                "  base_url: http://localhost:11434",
                "  prompt:",
                "    id: answer_correctness_v1",
                "    version: v1",
                "    template_path: prompts/judges/answer_correctness_v1.txt",
            ]
        ),
        encoding="utf-8",
    )

    config = load_evaluation_run_config(config_path)

    assert config.judge == JudgeConfig(
        enabled=True,
        model=LLMGenerationConfig(
            provider="ollama",
            model_name="judge-model",
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=10.0,
            base_url="http://localhost:11434",
        ),
        prompt=JudgePromptConfig(
            id="answer_correctness_v1",
            version="v1",
            template_path=Path("prompts/judges/answer_correctness_v1.txt"),
        ),
    )


def test_load_evaluation_run_config_serializes_base_url_in_snapshot(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                "  dataset_path: examples.jsonl",
                "  output_dir: runs",
                "  mode: closed_book",
                "  limit: 1",
                "model:",
                "  provider: ollama",
                "  model_name: answer-model",
                "  temperature: 0.0",
                "  max_tokens: 64",
                "  timeout_seconds: 30",
                "  base_url: http://localhost:11434",
                "judge:",
                "  enabled: true",
                "  provider: ollama",
                "  model_name: judge-model",
                "  temperature: 0.0",
                "  max_tokens: 32",
                "  timeout_seconds: 10",
                "  base_url: http://localhost:11434",
                "  prompt:",
                "    id: answer_correctness_v1",
                "    version: v1",
                "    template_path: prompts/judges/answer_correctness_v1.txt",
            ]
        ),
        encoding="utf-8",
    )

    config = load_evaluation_run_config(config_path)

    assert config.to_dict()["model"]["base_url"] == "http://localhost:11434"
    assert config.to_dict()["judge"]["base_url"] == "http://localhost:11434"


def test_load_evaluation_run_config_reports_missing_enabled_judge_keys(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                "  dataset_path: examples.jsonl",
                "  output_dir: runs",
                "  mode: closed_book",
                "  limit: 1",
                "model:",
                "  provider: mock",
                "  model_name: mock-llm",
                "  temperature: 0.0",
                "  max_tokens: 512",
                "  timeout_seconds: 30",
                "judge:",
                "  enabled: true",
                "  provider: mock",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationRunConfigError) as exc_info:
        load_evaluation_run_config(config_path)

    message = str(exc_info.value)
    assert "Evaluation run config missing required judge key(s):" in message
    assert "model_name" in message
    assert "prompt" in message


def test_load_evaluation_run_config_reports_invalid_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                "  dataset_path: examples.jsonl",
                "  output_dir: runs",
                "  mode: unsupported",
                "  limit: 1",
                "model:",
                "  provider: mock",
                "  model_name: mock-llm",
                "  temperature: 0.0",
                "  max_tokens: 512",
                "  timeout_seconds: 30",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationRunConfigError) as exc_info:
        load_evaluation_run_config(config_path)

    assert "Unknown evaluation mode: unsupported" in str(exc_info.value)


def test_load_evaluation_run_config_reports_missing_required_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                "  dataset_path: examples.jsonl",
                "  mode: closed_book",
                "model:",
                "  provider: mock",
                "  model_name: mock-llm",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationRunConfigError) as exc_info:
        load_evaluation_run_config(config_path)

    message = str(exc_info.value)
    assert "Evaluation run config missing required eval key(s):" in message
    assert "limit" in message
    assert "output_dir" in message
