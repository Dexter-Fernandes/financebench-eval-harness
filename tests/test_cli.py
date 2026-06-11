from pathlib import Path

import pytest

from financebench_eval_harness.cli import main


def test_validate_data_command_succeeds_for_expected_layout(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)
    (data_root / "questions.jsonl").write_text("{}", encoding="utf-8")

    exit_code = main(["validate-data", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"FinanceBench data layout is valid: {data_root}" in captured.out
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
    (data_root / "questions.jsonl").write_text("{}", encoding="utf-8")
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
    (data_root / "questions.jsonl").write_text("{}", encoding="utf-8")
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
