import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_python_module_run_baseline_writes_final_m2_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "examples.jsonl"
    output_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    config_path = tmp_path / "baseline_closed_book.yaml"
    _write_processed_examples(dataset_path)
    _write_baseline_config(config_path, dataset_path=dataset_path, output_dir=output_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "financebench_eval",
            "run-baseline",
            "--config",
            str(config_path),
            "--run-id",
            "final-run",
            "--report-dir",
            str(reports_dir),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = output_dir / "final-run"
    predictions_path = run_dir / "predictions.jsonl"
    scores_path = run_dir / "scores.jsonl"
    metadata_path = run_dir / "run_metadata.json"
    report_path = reports_dir / "baseline_final-run.md"
    assert (run_dir / "config.yaml").is_file()
    assert predictions_path.is_file()
    assert scores_path.is_file()
    assert metadata_path.is_file()
    assert report_path.is_file()
    assert not (run_dir / "outputs.jsonl").exists()
    assert f"Baseline run output: {run_dir}" in result.stdout
    assert f"Baseline report: {report_path}" in result.stdout

    snapshot = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["eval"]["mode"] == "closed_book"
    assert snapshot["model"]["model_name"] == "mock-llm"

    predictions = _read_jsonl(predictions_path)
    assert predictions == [
        {
            "question_id": "financebench_final_001",
            "question": "What was ACME revenue?",
            "gold_answer": "$123",
            "prediction": "mock response",
            "mode": "closed_book",
            "model_provider": "mock",
            "model_name": "mock-llm",
            "prompt_id": "closed_book_v1",
            "prompt_version": "v1",
            "prompt": predictions[0]["prompt"],
            "latency_ms": predictions[0]["latency_ms"],
            "input_tokens": None,
            "output_tokens": None,
            "status": "success",
            "error": None,
        }
    ]
    assert "What was ACME revenue?" in predictions[0]["prompt"]

    scores = _read_jsonl(scores_path)
    assert scores[0]["question_id"] == "financebench_final_001"
    assert scores[0]["scores"]["exact_match"] is False
    assert scores[0]["scores"]["numeric_match"] is False
    assert scores[0]["judge"]["status"] == "success"
    assert scores[0]["judge"]["verdict"] == "incorrect"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "final-run"
    assert metadata["predictions_path"] == str(predictions_path)
    assert metadata["scores_path"] == str(scores_path)
    assert metadata["prediction_filename"] == "predictions.jsonl"
    assert metadata["scores_filename"] == "scores.jsonl"
    assert metadata["report_path"] == str(report_path)


def _write_processed_examples(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "question_id": "financebench_final_001",
                "question": "What was ACME revenue?",
                "gold_answer": "$123",
                "evidence": [{"evidence_text": "Revenue was $123."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_baseline_config(
    path: Path,
    *,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "eval:",
                f"  dataset_path: {dataset_path}",
                f"  output_dir: {output_dir}",
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
                "  model_name: mock-judge",
                "  temperature: 0.0",
                "  max_tokens: 256",
                "  timeout_seconds: 30",
                "  prompt:",
                "    id: answer_correctness_v1",
                "    version: v1",
                "    template_path: prompts/judges/answer_correctness_v1.txt",
            ]
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
