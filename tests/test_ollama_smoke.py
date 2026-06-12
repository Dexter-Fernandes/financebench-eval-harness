import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import request
from urllib.error import URLError

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_SMOKE") != "1",
    reason="Set RUN_OLLAMA_SMOKE=1 to run local Ollama smoke tests.",
)


@pytest.mark.parametrize(
    ("mode", "run_id"),
    [
        ("closed_book", "ollama-closed-book-smoke"),
        ("oracle_context", "ollama-oracle-context-smoke"),
    ],
)
def test_ollama_smoke_run_eval_cli_end_to_end(
    tmp_path: Path,
    mode: str,
    run_id: str,
) -> None:
    _assert_ollama_ready(model_name="llama3.2:3b")

    dataset_path = tmp_path / "examples.jsonl"
    output_dir = tmp_path / "runs"
    config_path = tmp_path / f"{mode}.yaml"
    _write_processed_examples(dataset_path)
    _write_ollama_config(
        config_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        mode=mode,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "financebench_eval",
            "run-eval",
            "--config",
            str(config_path),
            "--run-id",
            run_id,
            "--limit",
            "1",
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = output_dir / run_id
    predictions_path = run_dir / "predictions.jsonl"
    scores_path = run_dir / "scores.jsonl"
    metadata_path = run_dir / "run_metadata.json"

    assert f"Evaluation run output: {run_dir}" in result.stdout
    assert predictions_path.is_file()
    assert scores_path.is_file()
    assert metadata_path.is_file()

    predictions = _read_jsonl(predictions_path)
    scores = _read_jsonl(scores_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(predictions) == 1
    assert predictions[0]["mode"] == mode
    assert predictions[0]["model_provider"] == "ollama"
    assert predictions[0]["model_name"] == "llama3.2:3b"
    assert predictions[0]["status"] == "success"
    assert isinstance(predictions[0]["prediction"], str)
    assert predictions[0]["prediction"].strip()

    assert len(scores) == 1
    assert scores[0]["question_id"] == "financebench_ollama_smoke_001"
    assert scores[0]["status"] == "success"
    assert isinstance(scores[0]["judge"], dict)
    assert scores[0]["judge"]["status"] == "success"

    assert metadata["mode"] == mode
    assert metadata["model_provider"] == "ollama"
    assert metadata["model_name"] == "llama3.2:3b"
    assert metadata["attempted_count"] == 1
    assert metadata["success_count"] == 1
    assert metadata["error_count"] == 0
    assert metadata["predictions_path"] == str(predictions_path)
    assert metadata["scores_path"] == str(scores_path)
    assert metadata["judge"]["enabled"] is True
    assert metadata["judge"]["provider"] == "ollama"
    assert metadata["judge"]["model_name"] == "llama3.2:3b"


def _write_processed_examples(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "question_id": "financebench_ollama_smoke_001",
                "question": "What was ACME revenue?",
                "gold_answer": "$123",
                "evidence": [{"evidence_text": "Revenue was $123."}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_ollama_config(
    path: Path,
    *,
    dataset_path: Path,
    output_dir: Path,
    mode: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "eval:",
                f"  dataset_path: {dataset_path}",
                f"  output_dir: {output_dir}",
                f"  mode: {mode}",
                "  limit: 1",
                "model:",
                "  provider: ollama",
                "  model_name: llama3.2:3b",
                "  temperature: 0.0",
                "  max_tokens: 512",
                "  timeout_seconds: 60",
                "  base_url: http://localhost:11434",
                "judge:",
                "  enabled: true",
                "  provider: ollama",
                "  model_name: llama3.2:3b",
                "  temperature: 0.0",
                "  max_tokens: 256",
                "  timeout_seconds: 60",
                "  base_url: http://localhost:11434",
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


def _assert_ollama_ready(*, model_name: str) -> None:
    tags_url = "http://localhost:11434/api/tags"
    try:
        with request.urlopen(tags_url, timeout=3.0) as response:
            response_body = response.read().decode("utf-8")
    except URLError as exc:
        pytest.fail(
            f"Ollama smoke prerequisite failed: could not reach {tags_url}. "
            "Start the local Ollama server before setting RUN_OLLAMA_SMOKE=1."
        )

    try:
        decoded_response = json.loads(response_body)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"Ollama smoke prerequisite failed: {tags_url} did not return valid JSON."
        )

    if not isinstance(decoded_response, dict):
        pytest.fail(
            f"Ollama smoke prerequisite failed: {tags_url} did not return a JSON object."
        )

    raw_models = decoded_response.get("models")
    if not isinstance(raw_models, list):
        pytest.fail(
            "Ollama smoke prerequisite failed: /api/tags response did not include a models list."
        )

    available_models = {
        model.get("name")
        for model in raw_models
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    }
    if model_name not in available_models:
        available_display = ", ".join(sorted(available_models)) or "(none)"
        pytest.fail(
            "Ollama smoke prerequisite failed: required model "
            f"'{model_name}' is not available locally. "
            f"Available models: {available_display}. "
            f"Run `ollama pull {model_name}` before setting RUN_OLLAMA_SMOKE=1."
        )
