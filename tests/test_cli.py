from pathlib import Path

import json
import pytest
import yaml

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


def test_build_examples_command_writes_processed_files_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "financebench"
    _write_questions(
        data_root / "questions.jsonl",
        [
            _question_row(
                "financebench_id_1",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 12,
                        "evidence_text": "Revenue was $123.",
                    }
                ],
            ),
            _question_row(
                "financebench_id_2",
                evidence=[
                    {
                        "doc_name": "ACME_2022_10K",
                        "evidence_page_num": 12,
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
                "page_num": 13,
                "text": "Revenue was\n$\n123.",
            }
        ],
    )

    exit_code = main(["build-examples", "--data-root", str(data_root)])

    captured = capsys.readouterr()
    examples_path = tmp_path / "data" / "processed" / "financebench" / "examples.jsonl"
    rejected_path = (
        tmp_path / "data" / "processed" / "financebench" / "examples.rejected.jsonl"
    )
    assert exit_code == 0
    assert "Accepted examples: 1" in captured.out
    assert "Rejected examples: 1" in captured.out
    assert "evidence_text_not_found_in_page_text: 1" in captured.out
    assert f"Wrote accepted examples to {examples_path}." in captured.out
    assert f"Wrote rejected examples to {rejected_path}." in captured.out
    assert captured.err == ""
    assert _read_jsonl(examples_path)[0]["question_id"] == "financebench_id_1"
    assert _read_jsonl(rejected_path)[0]["question_id"] == "financebench_id_2"


def test_run_eval_command_writes_mock_run_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    examples_path = tmp_path / "examples.jsonl"
    _write_pages(
        examples_path,
        [
            {
                "question_id": "q1",
                "question": "What is revenue?",
                "gold_answer": "$123.00",
                "evidence": [{"evidence_text": "Revenue was $123."}],
            },
            {
                "question_id": "q2",
                "question": "What is gross profit?",
                "gold_answer": "$456.00",
                "evidence": [{"evidence_text": "Gross profit was $456."}],
            }
        ],
    )
    config_path = tmp_path / "eval.yaml"
    output_dir = tmp_path / "runs"
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                f"  dataset_path: {examples_path}",
                f"  output_dir: {output_dir}",
                "  mode: closed_book",
                "  limit: 2",
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

    exit_code = main(
        [
            "run-eval",
            "--config",
            str(config_path),
            "--run-id",
            "cli-run",
            "--limit",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    run_dir = output_dir / "cli-run"
    assert f"Evaluation run output: {run_dir}" in captured.out
    assert f"Wrote config snapshot to {run_dir / 'config.yaml'}" in captured.out
    assert f"Wrote 1 predictions to {run_dir / 'predictions.jsonl'}" in captured.out
    assert f"Wrote 1 score rows to {run_dir / 'scores.jsonl'}" in captured.out
    assert f"Wrote run metadata to {run_dir / 'run_metadata.json'}" in captured.out
    assert "Attempted: 1" in captured.out
    assert "Succeeded: 1" in captured.out
    assert "Errors: 0" in captured.out
    assert "Judge attempted: 1" in captured.out
    assert "Judge succeeded: 1" in captured.out
    assert "Judge errors: 0" in captured.out
    assert captured.err == ""
    snapshot = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["eval"]["limit"] == 1
    rows = _read_jsonl(run_dir / "predictions.jsonl")
    score_rows = _read_jsonl(run_dir / "scores.jsonl")
    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["question"] == "What is revenue?"
    assert rows[0]["prediction"] == "mock response"
    assert rows[0]["model_provider"] == "mock"
    assert rows[0]["model_name"] == "mock-llm"
    assert rows[0]["latency_ms"] >= 0
    assert rows[0]["input_tokens"] is None
    assert rows[0]["output_tokens"] is None
    assert rows[0]["status"] == "success"
    assert score_rows[0]["scores"]["contains_gold_answer"] is False
    assert score_rows[0]["scores"]["numeric_match"] is False
    assert score_rows[0]["judge"]["status"] == "success"
    assert score_rows[0]["judge"]["verdict"] == "incorrect"
    assert score_rows[0]["judge"]["model_name"] == "mock-judge"
    assert "response" not in rows[0]
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["prediction_filename"] == "predictions.jsonl"
    assert run_metadata["scores_filename"] == "scores.jsonl"
    assert run_metadata["attempted_count"] == 1
    assert run_metadata["success_count"] == 1
    assert run_metadata["error_count"] == 0
    assert run_metadata["score_summary"]["example_count"] == 1
    assert run_metadata["score_summary"]["numeric_match_count"] == 0
    assert run_metadata["judge"]["enabled"] is True
    assert run_metadata["judge_summary"]["attempted_count"] == 1
    assert run_metadata["judge_summary"]["success_count"] == 1
    assert run_metadata["judge_summary"]["error_count"] == 0


def test_report_baseline_command_writes_markdown_report(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = tmp_path / "runs" / "report-run"
    reports_dir = tmp_path / "reports" / "generated"
    (run_dir).mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "mode": "closed_book",
                "model_provider": "mock",
                "model_name": "mock-model",
                "attempted_count": 2,
                "judge_summary": {
                    "attempted_count": 2,
                    "success_count": 2,
                    "error_count": 0,
                    "correct_count": 1,
                    "correct_rate": 0.5,
                    "partially_correct_count": 0,
                    "partially_correct_rate": 0.0,
                    "incorrect_count": 1,
                    "incorrect_rate": 0.5,
                    "not_answered_count": 0,
                    "not_answered_rate": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_pages(
        run_dir / "outputs.jsonl",
        [
            {
                "question_id": "q1",
                "question": "What was revenue?",
                "gold_answer": "$123",
                "prediction": "$123",
                "latency_ms": 1000,
                "input_tokens": None,
                "output_tokens": None,
                "judge": {
                    "status": "success",
                    "verdict": "correct",
                    "reason": "Matches.",
                    "error": None,
                },
            },
            {
                "question_id": "q2",
                "question": "What was gross profit?",
                "gold_answer": "$456",
                "prediction": "$400",
                "latency_ms": 2000,
                "input_tokens": None,
                "output_tokens": None,
                "judge": {
                    "status": "success",
                    "verdict": "incorrect",
                    "reason": "Wrong amount.",
                    "error": None,
                },
            },
        ],
    )

    exit_code = main(
        [
            "report-baseline",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(reports_dir),
        ]
    )

    captured = capsys.readouterr()
    report_path = reports_dir / "baseline_closed_book_mock-model.md"
    assert exit_code == 0
    assert f"Baseline report: {report_path}" in captured.out
    assert "Questions evaluated: 2" in captured.out
    assert "Correct: 1" in captured.out
    assert "Incorrect: 1" in captured.out
    assert captured.err == ""
    assert "| Accuracy estimate | 50% |" in report_path.read_text(encoding="utf-8")


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
        "company": "ACME Corp",
        "question": "What is revenue?",
        "answer": "$123.00",
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
