from pathlib import Path

import json
import pytest

from financebench_eval_harness.cli import main


def test_validate_data_command_succeeds_for_expected_layout(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)
    _write_valid_questions(data_root / "questions.jsonl")

    exit_code = main(["validate-data", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"FinanceBench data layout is valid: {data_root}" in captured.out
    assert "Loaded 1 FinanceBench examples." in captured.out
    assert captured.err == ""


def test_validate_data_command_returns_clear_error_for_missing_layout(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"

    exit_code = main(["validate-data", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FinanceBench data is missing." in captured.err
    assert str(data_root / "questions.jsonl") in captured.err
    assert str(data_root / "documents") in captured.err


def test_validate_data_command_succeeds_with_config_path(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    config_path = tmp_path / "dataset.yaml"
    (data_root / "documents").mkdir(parents=True)
    _write_valid_questions(data_root / "questions.jsonl")
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: financebench",
                f"  questions_path: {data_root / 'questions.jsonl'}",
                f"  documents_dir: {data_root / 'documents'}",
                f"  processed_dir: {tmp_path / 'processed'}",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-data", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"FinanceBench data layout is valid: {data_root}" in captured.out
    assert "Loaded 1 FinanceBench examples." in captured.out
    assert captured.err == ""


def test_validate_data_command_uses_configured_paths_in_missing_data_error(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: financebench",
                f"  questions_path: {data_root / 'questions.jsonl'}",
                f"  documents_dir: {data_root / 'documents'}",
                f"  processed_dir: {tmp_path / 'processed'}",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-data", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert str(data_root / "questions.jsonl") in captured.err
    assert str(data_root / "documents") in captured.err


def test_validate_data_command_uses_default_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs" / "datasets"
    data_root = tmp_path / "data" / "raw" / "financebench"
    config_dir.mkdir(parents=True)
    (data_root / "documents").mkdir(parents=True)
    _write_valid_questions(data_root / "questions.jsonl")
    (config_dir / "financebench.yaml").write_text(
        "\n".join(
            [
                "dataset:",
                "  name: financebench",
                "  questions_path: data/raw/financebench/questions.jsonl",
                "  documents_dir: data/raw/financebench/documents",
                "  processed_dir: data/processed/financebench",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-data"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FinanceBench data layout is valid: data/raw/financebench" in captured.out
    assert "Loaded 1 FinanceBench examples." in captured.out


def test_validate_data_command_returns_clear_error_for_invalid_questions(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)
    (data_root / "questions.jsonl").write_text(
        '{"financebench_id": "bad"}\n',
        encoding="utf-8",
    )

    exit_code = main(["validate-data", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Invalid FinanceBench row at line 1" in captured.err
    assert "question" in captured.err


def test_validate_dataset_command_succeeds_for_valid_dataset(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)
    _write_valid_questions(data_root / "questions.jsonl")

    exit_code = main(["validate-dataset", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid examples: 1" in captured.out
    assert "Invalid examples: 0" in captured.out
    assert "Dataset schema validation passed." in captured.out
    assert captured.err == ""


def test_validate_dataset_command_reports_invalid_counts_and_issues(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)
    (data_root / "questions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(_question_row("financebench_id_1")),
                json.dumps(_question_row("financebench_id_1")),
                json.dumps({"financebench_id": "financebench_id_3"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(["validate-dataset", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Valid examples: 1" in captured.out
    assert "Invalid examples: 2" in captured.out
    assert "Dataset schema validation failed." in captured.out
    assert "line 2 [financebench_id_1] duplicate question_id" in captured.out
    assert "line 3 [financebench_id_3] missing or empty 'question'" in captured.out


def test_validate_dataset_command_supports_config_path(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    config_path = tmp_path / "dataset.yaml"
    (data_root / "documents").mkdir(parents=True)
    _write_valid_questions(data_root / "questions.jsonl")
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: financebench",
                f"  questions_path: {data_root / 'questions.jsonl'}",
                f"  documents_dir: {data_root / 'documents'}",
                f"  processed_dir: {tmp_path / 'processed'}",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-dataset", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid examples: 1" in captured.out
    assert "Invalid examples: 0" in captured.out


def test_validate_dataset_command_supports_default_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "configs" / "datasets"
    data_root = tmp_path / "data" / "raw" / "financebench"
    config_dir.mkdir(parents=True)
    (data_root / "documents").mkdir(parents=True)
    _write_valid_questions(data_root / "questions.jsonl")
    (config_dir / "financebench.yaml").write_text(
        "\n".join(
            [
                "dataset:",
                "  name: financebench",
                "  questions_path: data/raw/financebench/questions.jsonl",
                "  documents_dir: data/raw/financebench/documents",
                "  processed_dir: data/processed/financebench",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-dataset"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Valid examples: 1" in captured.out
    assert "Dataset schema validation passed." in captured.out


def _write_valid_questions(path: Path) -> None:
    path.write_text(
        json.dumps(_question_row("financebench_id_1"))
        + "\n",
        encoding="utf-8",
    )


def _question_row(question_id: str) -> dict:
    return {
        "financebench_id": question_id,
        "question": "What is revenue?",
        "answer": "$123.00",
        "doc_name": "ACME_2022_10K",
        "evidence": [
            {
                "doc_name": "ACME_2022_10K",
                "evidence_page_num": 12,
                "evidence_text": "Revenue was $123.",
            }
        ],
    }
