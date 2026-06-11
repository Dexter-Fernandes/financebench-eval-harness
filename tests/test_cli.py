from pathlib import Path

import json
import pytest

import financebench_eval_harness.data as data_module
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


def test_validate_documents_command_succeeds_when_all_docs_resolve(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)
    (data_root / "documents" / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    _write_valid_questions(data_root / "questions.jsonl")

    exit_code = main(["validate-documents", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Resolved documents: 1" in captured.out
    assert "Missing documents: 0" in captured.out
    assert "Unused documents: 0" in captured.out
    assert "Document registry validation passed." in captured.out


def test_validate_documents_command_lists_missing_docs(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    (data_root / "documents").mkdir(parents=True)
    _write_valid_questions(data_root / "questions.jsonl")

    exit_code = main(["validate-documents", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Resolved documents: 0" in captured.out
    assert "Missing documents: 1" in captured.out
    assert "Document registry validation failed." in captured.out
    assert "missing ACME_2022_10K.pdf" in captured.out


def test_validate_documents_command_warns_about_unused_docs(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (documents / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    (documents / "UNUSED_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    _write_valid_questions(data_root / "questions.jsonl")

    exit_code = main(["validate-documents", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Unused documents: 1" in captured.out
    assert "unused UNUSED_2022_10K.pdf" in captured.out
    assert "Document registry validation passed." in captured.out


def test_extract_documents_command_writes_processed_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "financebench"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (documents / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    monkeypatch.setattr(
        data_module,
        "_pdf_reader_class",
        lambda: _fake_reader_factory(["page one", "page two"]),
    )

    exit_code = main(["extract-documents", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    pages_path = tmp_path / "data" / "processed" / "financebench" / "pages.jsonl"
    failures_path = (
        tmp_path / "data" / "processed" / "financebench" / "extraction_failures.jsonl"
    )
    assert exit_code == 0
    assert "Extracting ACME_2022_10K.pdf" in captured.out
    assert "Extracted 1 documents." in captured.out
    assert "Wrote 2 pages" in captured.out
    assert "Extraction failures: 0" in captured.out
    assert len(_read_jsonl(pages_path)) == 2
    assert failures_path.read_text(encoding="utf-8") == ""


def test_extract_documents_command_supports_config_processed_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    data_root = tmp_path / "financebench"
    processed_dir = tmp_path / "custom_processed"
    config_path = tmp_path / "dataset.yaml"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (documents / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: financebench",
                f"  questions_path: {data_root / 'questions.jsonl'}",
                f"  documents_dir: {documents}",
                f"  processed_dir: {processed_dir}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(data_module, "_pdf_reader_class", lambda: _fake_reader_factory(["page"]))

    exit_code = main(["extract-documents", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Extracting ACME_2022_10K.pdf" in captured.out
    assert f"Wrote 1 pages to {processed_dir / 'pages.jsonl'}." in captured.out
    assert _read_jsonl(processed_dir / "pages.jsonl")[0]["text"] == "page"


def test_extract_documents_command_writes_failure_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "financebench"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (documents / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    monkeypatch.setattr(
        data_module,
        "_pdf_reader_class",
        lambda: _fake_reader_factory([RuntimeError("page broke")]),
    )

    exit_code = main(["extract-documents", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    failures_path = (
        tmp_path / "data" / "processed" / "financebench" / "extraction_failures.jsonl"
    )
    assert exit_code == 0
    assert "Extracting ACME_2022_10K.pdf" in captured.out
    assert "Extraction failures: 1" in captured.out
    assert _read_jsonl(failures_path) == [
        {"doc_name": "ACME_2022_10K.pdf", "page_num": 1, "error": "page broke"}
    ]


def test_validate_evidence_pages_command_prints_match_and_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "financebench"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    document_path = documents / "ACME_2022_10K.pdf"
    document_path.write_text("pdf", encoding="utf-8")
    _write_questions(
        data_root / "questions.jsonl",
        [
            _question_row(
                "financebench_id_1",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 3,
                        "evidence_text": "Revenue was $123.",
                    }
                ],
            ),
            _question_row(
                "financebench_id_2",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 4,
                        "evidence_text": "Gross profit was $456.",
                    }
                ],
            ),
        ],
    )
    _write_pages(
        tmp_path / "data" / "processed" / "financebench" / "pages.jsonl",
        [
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 4,
                "text": "Revenue was\n$\n123.",
            },
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 5,
                "text": "This page is unrelated.",
            },
        ],
    )

    try:
        exit_code = main(["validate-evidence-pages", "--data-root", str(data_root)])
    except SystemExit as exc:
        pytest.fail(f"validate-evidence-pages command should be registered: {exc}")

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "MATCH financebench_id_1 evidence[1]" in captured.out
    assert "MISMATCH financebench_id_2 evidence[1]" in captured.out
    assert "doc=ACME_2022_10K.pdf" in captured.out
    assert f"path={document_path}" in captured.out
    assert "evidence_page=3 extracted_page=4" in captured.out
    assert "reason=matched method=alphanumeric_substring" in captured.out
    assert "reason=text_mismatch method=alphanumeric_substring" in captured.out
    assert "Evidence page validation failed." in captured.out
    assert "Total evidence checks: 2" in captured.out
    assert "Matches: 1" in captured.out
    assert "Mismatches: 1" in captured.out
    assert captured.err == ""


def test_validate_evidence_pages_command_succeeds_when_all_evidence_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "financebench"
    documents = data_root / "documents"
    documents.mkdir(parents=True)
    (documents / "ACME_2022_10K.pdf").write_text("pdf", encoding="utf-8")
    _write_questions(
        data_root / "questions.jsonl",
        [
            _question_row(
                "financebench_id_1",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 0,
                        "evidence_text": "Revenue was $123.",
                    }
                ],
            )
        ],
    )
    _write_pages(
        tmp_path / "data" / "processed" / "financebench" / "pages.jsonl",
        [
            {
                "doc_name": "ACME_2022_10K.pdf",
                "page_num": 1,
                "text": "Revenue was\n$\n123.",
            }
        ],
    )

    exit_code = main(["validate-evidence-pages", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "MATCH financebench_id_1 evidence[1]" in captured.out
    assert "MISMATCH" not in captured.out
    assert "evidence_page=0 extracted_page=1" in captured.out
    assert "Evidence page validation passed." in captured.out
    assert "Total evidence checks: 1" in captured.out
    assert "Matches: 1" in captured.out
    assert "Mismatches: 0" in captured.out
    assert captured.err == ""


def _write_valid_questions(path: Path) -> None:
    path.write_text(
        json.dumps(_question_row("financebench_id_1"))
        + "\n",
        encoding="utf-8",
    )


def _write_questions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_pages(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _question_row(question_id: str, evidence: list[dict] | None = None) -> dict:
    return {
        "financebench_id": question_id,
        "question": "What is revenue?",
        "answer": "$123.00",
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
