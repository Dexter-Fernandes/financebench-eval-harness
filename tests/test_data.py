from pathlib import Path

import json
import pytest

from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data import (
    FinanceBenchQuestionLoadError,
    MissingFinanceBenchDataError,
    load_financebench_examples,
    validate_financebench_dataset,
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


def test_load_financebench_examples_reads_all_examples(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(
        config.questions_path,
        [
            _question_row(question_id="financebench_id_1", answer="$1.00"),
            _question_row(question_id="financebench_id_2", answer="$2.00"),
        ],
    )
    config.documents_dir.mkdir(parents=True)

    examples = load_financebench_examples(config)

    assert len(examples) == 2
    assert examples[0].question_id == "financebench_id_1"
    assert examples[1].gold_answer == "$2.00"


def test_load_financebench_examples_normalizes_evidence_fields(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(config.questions_path, [_question_row()])

    example = load_financebench_examples(config)[0]

    assert example.question_id == "financebench_id_1"
    assert example.question == "What is revenue?"
    assert example.gold_answer == "$123.00"
    assert example.evidence_doc_name == "ACME_2022_10K"
    assert example.evidence_page_num == 12
    assert example.evidence_text == "Revenue was $123."
    assert len(example.evidence) == 1
    assert example.evidence[0].doc_name == "ACME_2022_10K"
    assert example.evidence[0].page_num == 12
    assert example.evidence[0].text == "Revenue was $123."


def test_load_financebench_examples_preserves_multiple_evidence_items(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    row = _question_row(
        evidence=[
            {
                "doc_name": "ACME_2022_10K",
                "evidence_page_num": 12,
                "evidence_text": "Revenue was $123.",
            },
            {
                "doc_name": "ACME_2022_10K",
                "evidence_page_num": 13,
                "evidence_text": "Costs were $100.",
            },
        ]
    )
    _write_questions(config.questions_path, [row])

    example = load_financebench_examples(config)[0]

    assert len(example.evidence) == 2
    assert example.evidence_doc_name == "ACME_2022_10K"
    assert example.evidence_page_num == 12
    assert example.evidence[1].page_num == 13
    assert example.evidence[1].text == "Costs were $100."


def test_load_financebench_examples_reports_missing_questions_file(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)

    with pytest.raises(FinanceBenchQuestionLoadError) as exc_info:
        load_financebench_examples(config)

    assert f"FinanceBench questions file not found: {config.questions_path}" in str(
        exc_info.value
    )


def test_load_financebench_examples_reports_invalid_json_line(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    config.questions_path.parent.mkdir(parents=True)
    config.questions_path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(FinanceBenchQuestionLoadError) as exc_info:
        load_financebench_examples(config)

    assert "Invalid FinanceBench JSONL at line 1" in str(exc_info.value)


def test_load_financebench_examples_reports_missing_top_level_field(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    row = _question_row()
    del row["answer"]
    _write_questions(config.questions_path, [row])

    with pytest.raises(FinanceBenchQuestionLoadError) as exc_info:
        load_financebench_examples(config)

    message = str(exc_info.value)
    assert "line 1" in message
    assert "answer" in message


def test_load_financebench_examples_reports_missing_evidence_metadata(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    row = _question_row(
        evidence=[
            {
                "doc_name": "ACME_2022_10K",
                "evidence_text": "Revenue was $123.",
            }
        ]
    )
    _write_questions(config.questions_path, [row])

    with pytest.raises(FinanceBenchQuestionLoadError) as exc_info:
        load_financebench_examples(config)

    message = str(exc_info.value)
    assert "line 1" in message
    assert "evidence[1].evidence_page_num" in message


def test_validate_financebench_dataset_counts_valid_examples(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(
        config.questions_path,
        [
            _question_row(question_id="financebench_id_1"),
            _question_row(question_id="financebench_id_2"),
        ],
    )

    result = validate_financebench_dataset(config)

    assert result.valid_count == 2
    assert result.invalid_count == 0
    assert result.issues == ()


def test_validate_financebench_dataset_reports_missing_required_fields(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    missing_question = _question_row(question_id="financebench_id_1")
    missing_question["question"] = ""
    missing_answer = _question_row(question_id="financebench_id_2")
    del missing_answer["answer"]
    missing_evidence = _question_row(question_id="financebench_id_3")
    missing_evidence["evidence"] = []
    _write_questions(config.questions_path, [missing_question, missing_answer, missing_evidence])

    result = validate_financebench_dataset(config)

    assert result.valid_count == 0
    assert result.invalid_count == 3
    assert [issue.line_number for issue in result.issues] == [1, 2, 3]
    assert [issue.field for issue in result.issues] == ["question", "answer", "evidence"]


def test_validate_financebench_dataset_reports_missing_evidence_metadata(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    row = _question_row(
        evidence=[
            {
                "doc_name": "",
                "evidence_page_num": "12",
                "evidence_text": "Revenue was $123.",
            }
        ]
    )
    _write_questions(config.questions_path, [row])

    result = validate_financebench_dataset(config)

    assert result.valid_count == 0
    assert result.invalid_count == 1
    assert result.issues[0].line_number == 1
    assert result.issues[0].field == "evidence[1].doc_name"
    assert "missing or empty" in result.issues[0].message


def test_validate_financebench_dataset_detects_duplicate_question_ids(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(
        config.questions_path,
        [
            _question_row(question_id="financebench_id_1"),
            _question_row(question_id="financebench_id_1"),
        ],
    )

    result = validate_financebench_dataset(config)

    assert result.valid_count == 1
    assert result.invalid_count == 1
    assert result.issues[0].line_number == 2
    assert result.issues[0].question_id == "financebench_id_1"
    assert result.issues[0].field == "financebench_id"
    assert result.issues[0].message == "duplicate question_id"


def test_validate_financebench_dataset_reports_malformed_json(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.questions_path.parent.mkdir(parents=True)
    config.questions_path.write_text("{not json}\n", encoding="utf-8")

    result = validate_financebench_dataset(config)

    assert result.valid_count == 0
    assert result.invalid_count == 1
    assert result.issues[0].line_number == 1
    assert result.issues[0].field == "json"
    assert result.issues[0].question_id is None


def test_validate_financebench_data_layout_accepts_dataset_config(tmp_path: Path) -> None:
    data_root = tmp_path / "financebench"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (data_root / "questions.jsonl").write_text("{}", encoding="utf-8")
    config = DatasetConfig(
        name="financebench",
        questions_path=data_root / "questions.jsonl",
        documents_dir=documents,
        processed_dir=tmp_path / "processed",
    )

    layout = validate_financebench_data_layout(config)

    assert layout.questions_path == config.questions_path
    assert layout.documents_path == config.documents_dir
    assert layout.processed_dir == config.processed_dir


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


def _dataset_config(tmp_path: Path) -> DatasetConfig:
    data_root = tmp_path / "financebench"
    return DatasetConfig(
        name="financebench",
        questions_path=data_root / "questions.jsonl",
        documents_dir=data_root / "documents",
        processed_dir=tmp_path / "processed",
    )


def _write_questions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _question_row(
    *,
    question_id: str = "financebench_id_1",
    answer: str = "$123.00",
    evidence: list[dict] | None = None,
) -> dict:
    return {
        "financebench_id": question_id,
        "question": "What is revenue?",
        "answer": answer,
        "doc_name": "ACME_2022_10K",
        "evidence": evidence
        or [
            {
                "doc_name": "ACME_2022_10K",
                "evidence_page_num": 12,
                "evidence_text": "Revenue was $123.",
            }
        ],
    }
