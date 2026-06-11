from pathlib import Path

import pytest

from financebench_eval_harness.data import (
    MissingFinanceBenchDataError,
    validate_financebench_data_layout,
)


def test_validate_financebench_data_layout_accepts_expected_files(tmp_path: Path) -> None:
    data_root = tmp_path / "financebench"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (data_root / "questions.jsonl").write_text("{}", encoding="utf-8")

    layout = validate_financebench_data_layout(data_root)

    assert layout.root == data_root
    assert layout.questions_path == data_root / "questions.jsonl"
    assert layout.documents_path == documents


def test_validate_financebench_data_layout_reports_missing_questions(tmp_path: Path) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)

    with pytest.raises(MissingFinanceBenchDataError) as exc_info:
        validate_financebench_data_layout(data_root)

    message = str(exc_info.value)
    assert "FinanceBench data is missing." in message
    assert str(data_root / "questions.jsonl") in message
    assert f"{data_root / 'documents'}/" in message
    assert "--data-root PATH" in message


def test_validate_financebench_data_layout_reports_missing_documents(tmp_path: Path) -> None:
    data_root = tmp_path / "financebench"
    data_root.mkdir(parents=True)
    (data_root / "questions.jsonl").write_text("{}", encoding="utf-8")

    with pytest.raises(MissingFinanceBenchDataError) as exc_info:
        validate_financebench_data_layout(data_root)

    message = str(exc_info.value)
    assert "FinanceBench data is missing." in message
    assert str(data_root / "questions.jsonl") in message
    assert str(data_root / "documents") in message
    assert "source documents under documents/" in message
