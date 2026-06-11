from pathlib import Path

import pytest

from financebench_eval_harness.config import (
    DatasetConfigError,
    load_dataset_config,
)


def test_load_dataset_config_reads_default_yaml_shape() -> None:
    config = load_dataset_config()

    assert config.name == "financebench"
    assert config.questions_path == Path("data/raw/financebench/questions.jsonl")
    assert config.documents_dir == Path("data/raw/financebench/documents")
    assert config.processed_dir == Path("data/processed/financebench")


def test_load_dataset_config_reads_custom_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: custombench",
                "  questions_path: custom/questions.jsonl",
                "  documents_dir: custom/documents",
                "  processed_dir: custom/processed",
            ]
        ),
        encoding="utf-8",
    )

    config = load_dataset_config(config_path)

    assert config.name == "custombench"
    assert config.questions_path == Path("custom/questions.jsonl")
    assert config.documents_dir == Path("custom/documents")
    assert config.processed_dir == Path("custom/processed")


def test_load_dataset_config_reports_missing_file(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(DatasetConfigError) as exc_info:
        load_dataset_config(config_path)

    assert f"Dataset config file not found: {config_path}" in str(exc_info.value)


def test_load_dataset_config_reports_missing_required_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: financebench",
                "  questions_path: data/raw/financebench/questions.jsonl",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetConfigError) as exc_info:
        load_dataset_config(config_path)

    message = str(exc_info.value)
    assert "Dataset config missing required key(s):" in message
    assert "documents_dir" in message
    assert "processed_dir" in message
