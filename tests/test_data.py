from pathlib import Path

import json
import pytest

import financebench_eval_harness.data as data_module
from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data import (
    DocumentExtractionError,
    FinanceBenchQuestionLoadError,
    MissingFinanceBenchDataError,
    build_document_registry,
    extract_document_pages,
    extract_financebench_documents,
    load_financebench_examples,
    validate_financebench_document_registry,
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


def test_build_document_registry_maps_filenames_to_paths(tmp_path: Path) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    document_path = config.documents_dir / "ACME_2022_10K.pdf"
    document_path.write_text("pdf", encoding="utf-8")

    registry = build_document_registry(config)

    assert registry == {"ACME_2022_10K.pdf": document_path}


def test_validate_financebench_document_registry_resolves_evidence_docs(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    document_path = config.documents_dir / "ACME_2022_10K.pdf"
    document_path.write_text("pdf", encoding="utf-8")
    _write_questions(config.questions_path, [_question_row()])

    result = validate_financebench_document_registry(config)

    assert result.is_valid
    assert result.resolved_documents == {"ACME_2022_10K.pdf": document_path}
    assert result.missing_documents == ()
    assert result.unused_documents == ()


def test_validate_financebench_document_registry_lists_missing_docs(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    _write_questions(config.questions_path, [_question_row()])

    result = validate_financebench_document_registry(config)

    assert not result.is_valid
    assert result.resolved_documents == {}
    assert result.missing_documents == ("ACME_2022_10K.pdf",)


def test_validate_financebench_document_registry_warns_about_unused_docs(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    used_path = config.documents_dir / "ACME_2022_10K.pdf"
    unused_path = config.documents_dir / "UNUSED_2022_10K.pdf"
    used_path.write_text("pdf", encoding="utf-8")
    unused_path.write_text("pdf", encoding="utf-8")
    _write_questions(config.questions_path, [_question_row()])

    result = validate_financebench_document_registry(config)

    assert result.is_valid
    assert result.resolved_documents == {"ACME_2022_10K.pdf": used_path}
    assert result.unused_documents == ("UNUSED_2022_10K.pdf",)


def test_extract_document_pages_returns_one_based_page_records(tmp_path: Path) -> None:
    document_path = tmp_path / "ACME_2022_10K.pdf"
    document_path.write_text("pdf", encoding="utf-8")

    pages, failures = extract_document_pages(
        document_path,
        pdf_reader_factory=_fake_reader_factory(["page one", "page two"]),
    )

    assert failures == []
    assert [page.doc_name for page in pages] == ["ACME_2022_10K.pdf", "ACME_2022_10K.pdf"]
    assert [page.page_num for page in pages] == [1, 2]
    assert [page.text for page in pages] == ["page one", "page two"]


def test_extract_document_pages_records_page_failures(tmp_path: Path) -> None:
    document_path = tmp_path / "ACME_2022_10K.pdf"
    document_path.write_text("pdf", encoding="utf-8")

    pages, failures = extract_document_pages(
        document_path,
        pdf_reader_factory=_fake_reader_factory(["page one", RuntimeError("boom")]),
    )

    assert len(pages) == 1
    assert pages[0].page_num == 1
    assert len(failures) == 1
    assert failures[0].doc_name == "ACME_2022_10K.pdf"
    assert failures[0].page_num == 2
    assert failures[0].error == "boom"


def test_extract_financebench_documents_writes_pages_and_failures_jsonl(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    (config.documents_dir / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")

    result = extract_financebench_documents(
        config,
        pdf_reader_factory=_fake_reader_factory(["page one", RuntimeError("boom")]),
    )

    assert result.document_count == 1
    assert result.page_count == 1
    assert result.failure_count == 1
    page_records = _read_jsonl(result.output_path)
    failure_records = _read_jsonl(result.failures_path)
    assert page_records == [
        {"doc_name": "ACME_2022_10K.pdf", "page_num": 1, "text": "page one"}
    ]
    assert failure_records == [
        {"doc_name": "ACME_2022_10K.pdf", "page_num": 2, "error": "boom"}
    ]


def test_extract_financebench_documents_reports_started_documents_in_sorted_order(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    (config.documents_dir / "BETA_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    (config.documents_dir / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    started_documents: list[str] = []

    extract_financebench_documents(
        config,
        pdf_reader_factory=_fake_reader_factory(["page"]),
        on_document_start=lambda path: started_documents.append(path.name),
    )

    assert started_documents == ["ACME_2022_10K.pdf", "BETA_2022_10K.pdf"]


def test_extract_financebench_documents_reports_empty_documents_dir(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)

    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_financebench_documents(config, pdf_reader_factory=_fake_reader_factory([]))

    assert "No PDF documents found" in str(exc_info.value)


def test_validate_financebench_evidence_pages_matches_extracted_offset_page(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    document_path = config.documents_dir / "ACME_2022_10K.pdf"
    document_path.write_text("pdf", encoding="utf-8")
    _write_questions(config.questions_path, [_question_row()])
    _write_pages(
        config.processed_dir / "pages.jsonl",
        [
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 12,
                "text": "Previous page.",
            },
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 13,
                "text": "Revenue was\n$\n123.",
            },
        ],
    )

    validator = getattr(data_module, "validate_financebench_evidence_pages", None)
    assert callable(validator), "validate_financebench_evidence_pages should exist"

    result = validator(config)

    assert result.is_valid
    assert result.total_count == 1
    assert result.matched_count == 1
    assert result.mismatch_count == 0
    check = result.checks[0]
    assert check.question_id == "financebench_id_1"
    assert check.evidence_index == 1
    assert check.document_filename == "ACME_2022_10K.pdf"
    assert check.document_path == document_path
    assert check.evidence_page_num == 12
    assert check.extracted_page_num == 13
    assert check.reason == "matched"
    assert check.evidence_excerpt == "Revenue was $123."
    assert "Revenue was" in check.page_excerpt


def test_validate_financebench_evidence_pages_reports_mismatch_reasons(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    acme_path = config.documents_dir / "ACME_2022_10K.pdf"
    beta_path = config.documents_dir / "BETA_2022_10K.pdf"
    acme_path.write_text("pdf", encoding="utf-8")
    beta_path.write_text("pdf", encoding="utf-8")
    _write_questions(
        config.questions_path,
        [
            _question_row(
                question_id="financebench_id_1",
                evidence=[
                    {
                        "doc_name": "MISSING_2022_10K",
                        "evidence_page_num": 1,
                        "evidence_text": "Missing document text.",
                    }
                ],
            ),
            _question_row(
                question_id="financebench_id_2",
                evidence=[
                    {
                        "doc_name": "BETA_2022_10K",
                        "evidence_page_num": 7,
                        "evidence_text": "Missing page text.",
                    }
                ],
            ),
            _question_row(
                question_id="financebench_id_3",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 5,
                        "evidence_text": "Revenue was $123.",
                    }
                ],
            ),
        ],
    )
    _write_pages(
        config.processed_dir / "pages.jsonl",
        [
            {
                "doc_name": "BETA_2022_10K.pdf",
                "page_num": 1,
                "text": "A different BETA page.",
            },
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 6,
                "text": "This page does not contain the expected snippet.",
            },
        ],
    )

    result = data_module.validate_financebench_evidence_pages(config)

    assert not result.is_valid
    assert result.total_count == 3
    assert result.matched_count == 0
    assert result.mismatch_count == 3
    assert [check.reason for check in result.checks] == [
        "missing_document",
        "missing_page",
        "text_mismatch",
    ]
    assert result.checks[0].document_path is None
    assert result.checks[1].document_path == beta_path
    assert result.checks[2].document_path == acme_path
    assert result.checks[1].extracted_page_num == 8
    assert result.checks[2].page_excerpt == (
        "This page does not contain the expected snippet."
    )


def test_validate_financebench_evidence_pages_reports_missing_pages_file(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    config.documents_dir.mkdir(parents=True)
    (config.documents_dir / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    _write_questions(config.questions_path, [_question_row()])

    with pytest.raises(data_module.DocumentPageLoadError) as exc_info:
        data_module.validate_financebench_evidence_pages(config)

    assert str(config.processed_dir / "pages.jsonl") in str(exc_info.value)


def test_canonical_pdf_name_normalizes_document_names() -> None:
    canonical_pdf_name = getattr(data_module, "canonical_pdf_name", None)
    assert callable(canonical_pdf_name), "canonical_pdf_name should exist"

    assert canonical_pdf_name("ACME_2022_10K") == "ACME_2022_10K.pdf"
    assert canonical_pdf_name("ACME_2022_10K.pdf") == "ACME_2022_10K.pdf"


def test_build_processed_financebench_examples_writes_canonical_jsonl(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(
        config.questions_path,
        [
            _question_row(
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 12,
                        "evidence_text": "Revenue was $123.",
                    },
                    {
                        "doc_name": "ACME_2022_10K.pdf",
                        "evidence_page_num": 21,
                        "evidence_text": "Costs were $100.",
                    },
                ],
            )
        ],
    )
    _write_pages(
        config.processed_dir / "pages.jsonl",
        [
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 12,
                "text": "Previous page.",
            },
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 13,
                "text": "Revenue was\n$\n123. More text.",
            },
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 21,
                "text": "Costs were $100.",
            },
        ],
    )

    builder = getattr(data_module, "build_processed_financebench_examples", None)
    assert callable(builder), "build_processed_financebench_examples should exist"
    result = builder(config)

    assert result.accepted_count == 1
    assert result.rejected_count == 0
    assert result.output_path == config.processed_dir / "examples.jsonl"
    assert result.rejected_path == config.processed_dir / "examples.rejected.jsonl"
    assert result.skip_reason_counts == {}
    assert _read_jsonl(result.rejected_path) == []

    rows = _read_jsonl(result.output_path)
    assert rows == [
        {
            "question_id": "financebench_id_1",
            "company": "ACME Corp",
            "doc_name": "ACME_2022_10K",
            "question": "What is revenue?",
            "gold_answer": "$123.00",
            "evidence": [
                {
                    "raw_evidence_index": 0,
                    "doc_name": "ACME_2022_10K",
                    "gold_page_num": 12,
                    "matched_page_num": 13,
                    "evidence_text": "Revenue was $123.",
                    "page_text": "Revenue was\n$\n123. More text.",
                    "evidence_quality": {
                        "normalizer": "financebench_text_v1",
                        "match_status": "exact_match",
                        "normalized_substring_match": True,
                        "normalized_full_page_match": False,
                        "evidence_char_count": 17,
                        "page_char_count": 29,
                        "line_coverage_ratio": 1.0,
                        "warnings": ["matched_page_num_differs_from_gold_page_num"],
                    },
                },
                {
                    "raw_evidence_index": 1,
                    "doc_name": "ACME_2022_10K.pdf",
                    "gold_page_num": 21,
                    "matched_page_num": 21,
                    "evidence_text": "Costs were $100.",
                    "page_text": "Costs were $100.",
                    "evidence_quality": {
                        "normalizer": "financebench_text_v1",
                        "match_status": "exact_match",
                        "normalized_substring_match": True,
                        "normalized_full_page_match": True,
                        "evidence_char_count": 16,
                        "page_char_count": 16,
                        "line_coverage_ratio": 1.0,
                        "warnings": [],
                    },
                },
            ],
        }
    ]


def test_build_processed_financebench_examples_writes_rejected_jsonl(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    _write_questions(
        config.questions_path,
        [
            _question_row(question_id="financebench_id_1", evidence=[]),
            _question_row(
                question_id="financebench_id_2",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": "12",
                        "evidence_text": "Revenue was $123.",
                    }
                ],
            ),
            _question_row(
                question_id="financebench_id_3",
                evidence=[
                    {
                        "doc_name": "MISSING_2022_10K",
                        "evidence_page_num": 4,
                        "evidence_text": "Missing document text.",
                    }
                ],
            ),
            _question_row(
                question_id="financebench_id_4",
                evidence=[
                    {
                        "doc_name": "BETA_2022_10K",
                        "evidence_page_num": 4,
                        "evidence_text": "Missing page text.",
                    }
                ],
            ),
            _question_row(
                question_id="financebench_id_5",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 12,
                        "evidence_text": "Revenue was $123.",
                    }
                ],
            ),
        ],
    )
    _write_pages(
        config.processed_dir / "pages.jsonl",
        [
            {
                "doc_name": "BETA_2022_10K.pdf",
                "page_num": 99,
                "text": "Missing page text.",
            },
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 13,
                "text": "This page is unrelated.",
            },
        ],
    )

    result = data_module.build_processed_financebench_examples(config)

    assert result.accepted_count == 0
    assert result.rejected_count == 5
    assert result.skip_reason_counts == {
        "missing_evidence": 1,
        "malformed_evidence": 1,
        "missing_extracted_document": 1,
        "missing_extracted_page": 1,
        "evidence_text_not_found_in_page_text": 1,
    }
    assert _read_jsonl(result.output_path) == []
    rejected = _read_jsonl(result.rejected_path)
    assert [row["question_id"] for row in rejected] == [
        "financebench_id_1",
        "financebench_id_2",
        "financebench_id_3",
        "financebench_id_4",
        "financebench_id_5",
    ]
    assert [row["skip_reason"] for row in rejected] == [
        "missing_evidence",
        "malformed_evidence",
        "missing_extracted_document",
        "missing_extracted_page",
        "evidence_text_not_found_in_page_text",
    ]
    assert rejected[0]["evidence"] == []
    assert rejected[1]["evidence"][0]["raw_evidence_index"] == 0
    assert rejected[1]["evidence"][0]["evidence_quality"]["match_status"] == (
        "malformed_evidence"
    )
    assert rejected[4]["evidence"][0]["matched_page_num"] == 13
    assert rejected[4]["evidence"][0]["evidence_quality"]["normalizer"] == (
        "financebench_text_v1"
    )
    assert rejected[4]["evidence"][0]["evidence_quality"]["normalized_substring_match"] is False


def test_load_processed_financebench_examples_reads_only_processed_jsonl(
    tmp_path: Path,
) -> None:
    config = _dataset_config(tmp_path)
    processed_row = {
        "question_id": "financebench_id_1",
        "company": "ACME Corp",
        "doc_name": "ACME_2022_10K",
        "question": "What is revenue?",
        "gold_answer": "$123.00",
        "evidence": [
            {
                "raw_evidence_index": 0,
                "doc_name": "ACME_2022_10K",
                "gold_page_num": 12,
                "matched_page_num": 13,
                "evidence_text": "Revenue was $123.",
                "page_text": "Revenue was $123.",
                "evidence_quality": {
                    "normalizer": "financebench_text_v1",
                    "match_status": "exact_match",
                    "normalized_substring_match": True,
                    "normalized_full_page_match": True,
                    "evidence_char_count": 17,
                    "page_char_count": 17,
                    "line_coverage_ratio": 1.0,
                    "warnings": [],
                },
            }
        ],
    }
    _write_jsonl(config.processed_dir / "examples.jsonl", [processed_row])

    loader = getattr(data_module, "load_processed_financebench_examples", None)
    assert callable(loader), "load_processed_financebench_examples should exist"
    rows = loader(config)

    assert rows == [processed_row]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_pages(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
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
        "company": "ACME Corp",
        "question": "What is revenue?",
        "answer": answer,
        "doc_name": "ACME_2022_10K",
        "evidence": evidence
        if evidence is not None
        else [
            {
                "doc_name": "ACME_2022_10K",
                "evidence_page_num": 12,
                "evidence_text": "Revenue was $123.",
            }
        ],
    }


def _fake_reader_factory(page_outputs: list[object]):
    class FakePage:
        def __init__(self, output: object) -> None:
            self.output = output

        def extract_text(self) -> str:
            if isinstance(self.output, Exception):
                raise self.output
            return str(self.output)

    class FakeReader:
        def __init__(self, path: str) -> None:
            self.path = path
            self.pages = [FakePage(output) for output in page_outputs]

    return FakeReader


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
