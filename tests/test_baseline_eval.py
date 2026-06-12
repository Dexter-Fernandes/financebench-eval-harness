import json
from pathlib import Path

from financebench_eval_harness.evaluation import (
    EvaluationMode,
    load_evaluation_config,
    render_prompt_for_processed_example,
)
from financebench_eval_harness.llm import LLMGenerationConfig, MockLLMClient
from financebench_eval_harness.report import generate_baseline_report
from financebench_eval_harness.run import run_evaluation_from_config
from financebench_eval_harness.run_config import (
    EvaluationRunConfig,
    EvaluationRunSettings,
    JudgeConfig,
    JudgePromptConfig,
)


def test_baseline_prompts_render_closed_book_and_oracle_context() -> None:
    evaluation_config = load_evaluation_config()
    processed_example = _processed_example()

    closed_book = render_prompt_for_processed_example(
        evaluation_config,
        EvaluationMode.CLOSED_BOOK,
        processed_example,
    )
    oracle_context = render_prompt_for_processed_example(
        evaluation_config,
        EvaluationMode.ORACLE_CONTEXT,
        processed_example,
    )

    assert closed_book.prompt_id == "closed_book_v1"
    assert "What was ACME revenue?" in closed_book.text
    assert "Revenue was $123." not in closed_book.text
    assert oracle_context.prompt_id == "oracle_context_v1"
    assert "What was ACME revenue?" in oracle_context.text
    assert "[Evidence 1]\nRevenue was $123." in oracle_context.text


def test_baseline_eval_loop_writes_predictions_scores_and_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    _write_processed_examples(dataset_path, [_processed_example()])
    run_config = _run_config(dataset_path=dataset_path, output_dir=tmp_path / "runs")
    llm_client = MockLLMClient(run_config.model, responses=["$123"])
    judge_client = MockLLMClient(
        run_config.judge.model,
        responses=['{"verdict": "correct", "reason": "Matches the gold answer."}'],
    )

    run_result = run_evaluation_from_config(
        run_config,
        llm_client,
        judge_client=judge_client,
        run_id="smoke-run",
    )

    rows = _read_jsonl(run_result.predictions_path)
    score_rows = _read_jsonl(run_result.scores_path)
    assert run_result.example_count == 1
    assert len(rows) == 1
    assert rows[0]["question_id"] == "financebench_smoke_001"
    assert rows[0]["question"] == "What was ACME revenue?"
    assert rows[0]["gold_answer"] == "$123"
    assert rows[0]["prediction"] == "$123"
    assert rows[0]["mode"] == "closed_book"
    assert rows[0]["prompt_id"] == "closed_book_v1"
    assert rows[0]["model_provider"] == "mock"
    assert rows[0]["model_name"] == "mock-model"
    assert score_rows[0]["scores"]["exact_match"] is True
    assert score_rows[0]["scores"]["numeric_match"] is True
    assert score_rows[0]["judge"]["status"] == "success"
    assert score_rows[0]["judge"]["verdict"] == "correct"

    run_metadata = json.loads(run_result.run_metadata_path.read_text(encoding="utf-8"))
    assert run_metadata["judge_summary"]["correct_count"] == 1
    assert run_metadata["score_summary"]["exact_match_count"] == 1

    report_result = generate_baseline_report(
        run_result.output_dir,
        output_dir=tmp_path / "reports",
    )

    report = report_result.report_path.read_text(encoding="utf-8")
    assert report_result.evaluated_count == 1
    assert report_result.report_path == tmp_path / "reports" / "baseline_closed_book_mock-model.md"
    assert "| Questions evaluated | 1 |" in report
    assert "| Correct | 1 |" in report
    assert "| Accuracy estimate | 100% |" in report


def _processed_example() -> dict[str, object]:
    return {
        "question_id": "financebench_smoke_001",
        "question": "What was ACME revenue?",
        "gold_answer": "$123",
        "evidence": [{"evidence_text": "Revenue was $123."}],
    }


def _run_config(*, dataset_path: Path, output_dir: Path) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        settings=EvaluationRunSettings(
            dataset_path=dataset_path,
            output_dir=output_dir,
            mode=EvaluationMode.CLOSED_BOOK,
            limit=1,
        ),
        model=LLMGenerationConfig(
            provider="mock",
            model_name="mock-model",
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=30.0,
        ),
        judge=JudgeConfig(
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
                template_path=Path("prompts/judges/answer_correctness_v1.txt"),
            ),
        ),
    )


def _write_processed_examples(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
