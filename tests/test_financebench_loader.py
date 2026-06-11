from pathlib import Path

import json

import pytest

from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data import (
    FinanceBenchQuestionLoadError,
    load_financebench_examples,
    validate_financebench_dataset,
    validate_financebench_document_registry,
    validate_financebench_evidence_pages,
)


def test_loads_valid_sample_file(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(config.questions_path, [_question_row()])

    examples = load_financebench_examples(config)

    assert len(examples) == 1
    example = examples[0]
    assert example.question_id == "financebench_id_03029"
    assert example.company == "3M"
    assert example.doc_name == "3M_2018_10K"
    assert example.question == "What is the FY2018 capital expenditure amount for 3M?"
    assert example.gold_answer == "$1577.00"
    assert example.evidence_doc_name == "3M_2018_10K"
    assert example.evidence_page_num == 59
    assert example.evidence_text == "Purchases of property, plant and equipment were $1,577."


def test_rejects_missing_question_field(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    row = _question_row()
    del row["question"]
    _write_questions(config.questions_path, [row])

    with pytest.raises(FinanceBenchQuestionLoadError) as exc_info:
        load_financebench_examples(config)

    assert "question" in str(exc_info.value)


def test_rejects_missing_answer_field(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    row = _question_row()
    del row["answer"]
    _write_questions(config.questions_path, [row])

    with pytest.raises(FinanceBenchQuestionLoadError) as exc_info:
        load_financebench_examples(config)

    assert "answer" in str(exc_info.value)


def test_detects_duplicate_ids(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(
        config.questions_path,
        [
            _question_row(question_id="financebench_id_duplicate"),
            _question_row(question_id="financebench_id_duplicate"),
        ],
    )

    result = validate_financebench_dataset(config)

    assert result.valid_count == 1
    assert result.invalid_count == 1
    assert result.issues[0].question_id == "financebench_id_duplicate"
    assert result.issues[0].field == "financebench_id"
    assert result.issues[0].message == "duplicate question_id"


def test_detects_missing_evidence_documents(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    _write_questions(config.questions_path, [_question_row()])

    result = validate_financebench_document_registry(config)

    assert not result.is_valid
    assert result.missing_documents == ("3M_2018_10K.pdf",)
    assert result.resolved_documents == {}


def test_links_evidence_page_correctly(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    document_path = config.documents_dir / "3M_2018_10K.pdf"
    document_path.write_text("pdf", encoding="utf-8")
    _write_questions(config.questions_path, [_question_row()])
    _write_pages(
        config.processed_dir / "pages.jsonl",
        [
            {
                "doc_name": "3M_2018_10K.pdf",
                "page_num": 60,
                "text": "Purchases of property, plant and equipment were $1,577.",
            }
        ],
    )

    result = validate_financebench_evidence_pages(config)

    assert result.is_valid
    assert result.total_count == 1
    assert result.matched_count == 1
    assert result.mismatch_count == 0
    check = result.checks[0]
    assert check.question_id == "financebench_id_03029"
    assert check.document_filename == "3M_2018_10K.pdf"
    assert check.document_path == document_path
    assert check.evidence_page_num == 59
    assert check.extracted_page_num == 60
    assert check.reason == "matched"


def _dataset_config(tmp_path: Path) -> DatasetConfig:
    data_root = tmp_path / "financebench"
    return DatasetConfig(
        name="financebench",
        questions_path=data_root / "questions.jsonl",
        documents_dir=data_root / "documents",
        processed_dir=tmp_path / "processed",
    )


def _question_row(
    *,
    question_id: str = "financebench_id_03029",
) -> dict:
    return {
        "financebench_id": question_id,
        "company": "3M",
        "doc_name": "3M_2018_10K",
        "question": "What is the FY2018 capital expenditure amount for 3M?",
        "answer": "$1577.00",
        "evidence": [
            {
                "doc_name": "3M_2018_10K",
                "evidence_page_num": 59,
                "evidence_text": (
                    "Purchases of property, plant and equipment were $1,577."
                ),
            }
        ],
    }


def _write_questions(path: Path, rows: list[dict]) -> None:
    _write_jsonl(path, rows)


def _write_pages(path: Path, rows: list[dict]) -> None:
    _write_jsonl(path, rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
