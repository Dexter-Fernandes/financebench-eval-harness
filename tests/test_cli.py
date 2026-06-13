from pathlib import Path

import json
import pytest
import yaml

import financebench_eval_harness.data as data_module
from financebench_eval_harness.cli import main
from financebench_eval_harness.llm import LLMGenerationResult


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


def test_run_eval_builds_ollama_clients_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    examples_path = tmp_path / "examples.jsonl"
    output_dir = tmp_path / "runs"
    config_path = tmp_path / "eval.yaml"
    _write_pages(
        examples_path,
        [
            {
                "question_id": "q1",
                "question": "What is revenue?",
                "gold_answer": "$123",
                "evidence": [{"evidence_text": "Revenue was $123."}],
            }
        ],
    )
    config_path.write_text(
        "\n".join(
            [
                "eval:",
                f"  dataset_path: {examples_path}",
                f"  output_dir: {output_dir}",
                "  mode: closed_book",
                "  limit: 9",
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

    built_clients: list[tuple[str, str, str | None]] = []

    class FakeOllamaClient:
        def __init__(self, config) -> None:
            built_clients.append((config.provider, config.model_name, config.base_url))
            self.config = config
            self.calls = 0

        def generate(self, prompt: str) -> LLMGenerationResult:
            self.calls += 1
            if self.calls == 2:
                return LLMGenerationResult(
                    text='{"verdict": "correct", "reason": "Matches."}',
                    prompt_tokens=20,
                    output_tokens=8,
                )
            return LLMGenerationResult(
                text="ollama response",
                prompt_tokens=10,
                output_tokens=4,
            )

    monkeypatch.setattr("financebench_eval_harness.cli.OllamaLLMClient", FakeOllamaClient)

    exit_code = main(
        [
            "run-eval",
            "--config",
            str(config_path),
            "--run-id",
            "ollama-cli-run",
            "--limit",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Attempted: 1" in captured.out
    assert built_clients == [
        ("ollama", "llama3.2:3b", "http://localhost:11434"),
        ("ollama", "llama3.2:3b", "http://localhost:11434"),
    ]


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


# ---------------------------------------------------------------------------
# build-index command
# ---------------------------------------------------------------------------


def _write_pages_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_retrieval_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "retrieval:",
            "  evidence_overlap_threshold: 0.5",
            "  chunking:",
            "    strategy: recursive_text",
            "    chunk_size: 200",
            "    chunk_overlap: 50",
            "    min_chunk_chars: 10",
        ]),
        encoding="utf-8",
    )


def _write_embedding_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "embedding:",
            "  provider: mock",
            "  model_name: mock-embed",
            "  batch_size: 32",
        ]),
        encoding="utf-8",
    )


_SAMPLE_PAGES = [
    {
        "doc_name": "ACME_2022_10K.pdf",
        "page_num": 1,
        "text": "Revenue for the fiscal year was $123 million, up 12% year over year.",
    },
    {
        "doc_name": "ACME_2022_10K.pdf",
        "page_num": 2,
        "text": "Net income for the period was $45 million after tax.",
    },
]


def test_build_index_command_succeeds_and_writes_index_files(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    index_dir = tmp_path / "index"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(index_dir),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "index_metadata.json").is_file()
    assert captured.err == ""


def test_build_index_command_prints_summary(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    index_dir = tmp_path / "index"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(index_dir),
    ])

    captured = capsys.readouterr()
    assert str(index_dir) in captured.out
    assert "chunk" in captured.out.lower()


def test_build_index_command_prints_embedding_model(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(tmp_path / "index"),
    ])

    captured = capsys.readouterr()
    assert "mock-embed" in captured.out


def test_build_index_command_fails_if_pages_file_missing(
    tmp_path: Path,
    capsys,
) -> None:
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(tmp_path / "nonexistent.jsonl"),
        "--index-dir", str(tmp_path / "index"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err != ""


def test_build_index_command_fails_if_retrieval_config_missing(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(tmp_path / "nonexistent.yaml"),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(tmp_path / "index"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err != ""


def test_build_index_command_fails_if_embedding_config_missing(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(tmp_path / "nonexistent.yaml"),
        "--pages", str(pages_path),
        "--index-dir", str(tmp_path / "index"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err != ""


def test_build_index_command_fails_if_pages_file_is_empty(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    pages_path.write_text("", encoding="utf-8")
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(tmp_path / "index"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err != ""


def test_build_index_dry_run_prints_summary_without_writing_files(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    index_dir = tmp_path / "index"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(index_dir),
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert not index_dir.exists()
    assert "dry" in captured.out.lower()
    assert captured.err == ""


def test_build_index_dry_run_reports_page_and_chunk_counts(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(tmp_path / "index"),
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert "page" in captured.out.lower()
    assert "chunk" in captured.out.lower()


def test_build_index_command_succeeds_without_progress_flag(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(tmp_path / "index"),
    ])

    assert exit_code == 0


def test_build_index_command_accepts_progress_flag(
    tmp_path: Path,
    capsys,
) -> None:
    pages_path = tmp_path / "pages.jsonl"
    retrieval_cfg = tmp_path / "retrieval.yaml"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_retrieval_config(retrieval_cfg)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "build-index",
        "--retrieval-config", str(retrieval_cfg),
        "--embedding-config", str(embedding_cfg),
        "--pages", str(pages_path),
        "--index-dir", str(tmp_path / "index"),
        "--progress",
    ])

    assert exit_code == 0


# ---------------------------------------------------------------------------
# embed-question command
# ---------------------------------------------------------------------------


def test_embed_question_command_succeeds_and_prints_dimension(
    tmp_path: Path,
    capsys,
) -> None:
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "embed-question",
        "--question", "What was the total revenue for fiscal year 2022?",
        "--embedding-config", str(embedding_cfg),
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dim" in captured.out.lower() or "dimension" in captured.out.lower()
    assert captured.err == ""


def test_embed_question_command_fails_on_empty_question(
    tmp_path: Path,
    capsys,
) -> None:
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "embed-question",
        "--question", "",
        "--embedding-config", str(embedding_cfg),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err != ""


def test_embed_question_command_prints_provider_and_model(
    tmp_path: Path,
    capsys,
) -> None:
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_embedding_config(embedding_cfg)

    main([
        "embed-question",
        "--question", "What was net income?",
        "--embedding-config", str(embedding_cfg),
    ])

    captured = capsys.readouterr()
    assert "mock-embed" in captured.out


def test_embed_question_command_fails_if_embedding_config_missing(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main([
        "embed-question",
        "--question", "What was revenue?",
        "--embedding-config", str(tmp_path / "nonexistent.yaml"),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err != ""


# ---------------------------------------------------------------------------
# retrieve command helpers
# ---------------------------------------------------------------------------


def _build_test_index(tmp_path: Path) -> Path:
    """Build a small FAISS index in tmp_path/index and return the index dir."""
    from financebench_eval_harness.chunking import chunk_pages
    from financebench_eval_harness.embedding import EmbeddingConfig, MockEmbeddingClient
    from financebench_eval_harness.index_builder import build_index
    from financebench_eval_harness.retrieval_config import load_retrieval_config
    from financebench_eval_harness.retrieval_types import DocumentPage

    pages = [
        DocumentPage(doc_id="ACME", doc_name="ACME.pdf", page_num=1,
                     text="Total revenue for fiscal year 2022 was $4.2 billion."),
        DocumentPage(doc_id="ACME", doc_name="ACME.pdf", page_num=2,
                     text="Net income was $540 million, down 3% year over year."),
    ]
    retrieval_cfg_path = tmp_path / "retrieval.yaml"
    _write_retrieval_config(retrieval_cfg_path)
    cfg = load_retrieval_config(retrieval_cfg_path)
    chunks = chunk_pages(pages, cfg.chunking)

    embed_cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
    client = MockEmbeddingClient(embed_cfg, embedding_dim=8)

    index_dir = tmp_path / "index"
    build_index(chunks, client, cfg, index_dir)
    return index_dir


def _write_questions_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


_SAMPLE_QUESTIONS = [
    {"question_id": "q001", "question": "What was the total revenue?"},
    {"question_id": "q002", "question": "What was net income?"},
]


# ---------------------------------------------------------------------------
# retrieve command
# ---------------------------------------------------------------------------


def test_retrieve_command_exits_zero(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(tmp_path / "results.jsonl"),
    ])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_retrieve_command_writes_jsonl(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
    ])

    assert output_path.is_file()
    rows = _read_jsonl(output_path)
    assert len(rows) == 2


def test_retrieve_command_output_has_question_id_and_retrieved(
    tmp_path: Path, capsys
) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
    ])

    rows = _read_jsonl(output_path)
    assert rows[0]["question_id"] == "q001"
    assert "retrieved" in rows[0]
    assert len(rows[0]["retrieved"]) > 0


def test_retrieve_command_respects_top_k(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "results.jsonl"
    _write_questions_jsonl(questions_path, [_SAMPLE_QUESTIONS[0]])
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
        "--top-k", "1",
    ])

    rows = _read_jsonl(output_path)
    assert len(rows[0]["retrieved"]) == 1


def test_retrieve_command_prints_summary(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(tmp_path / "results.jsonl"),
    ])

    out = capsys.readouterr().out
    assert "2" in out


def test_retrieve_command_fails_if_index_missing(tmp_path: Path, capsys) -> None:
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "retrieve",
        "--index-dir", str(tmp_path / "nonexistent"),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(tmp_path / "results.jsonl"),
    ])

    assert exit_code == 1
    assert capsys.readouterr().err != ""


def test_retrieve_command_fails_if_questions_file_missing(
    tmp_path: Path, capsys
) -> None:
    index_dir = _build_test_index(tmp_path)
    embedding_cfg = tmp_path / "embedding.yaml"
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(tmp_path / "nonexistent.jsonl"),
        "--embedding-config", str(embedding_cfg),
        "--output", str(tmp_path / "results.jsonl"),
    ])

    assert exit_code == 1
    assert capsys.readouterr().err != ""


def test_retrieve_command_writes_run_metadata_json(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "run_001" / "retrieval_results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
        "--run-id", "test-run",
    ])

    assert (tmp_path / "run_001" / "retrieval_run_metadata.json").is_file()


def test_retrieve_command_metadata_contains_provided_run_id(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "run_001" / "retrieval_results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
        "--run-id", "my-custom-run",
    ])

    d = json.loads((tmp_path / "run_001" / "retrieval_run_metadata.json").read_text())
    assert d["run_id"] == "my-custom-run"


def test_retrieve_command_metadata_auto_generates_run_id(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "run_001" / "retrieval_results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
    ])

    d = json.loads((tmp_path / "run_001" / "retrieval_run_metadata.json").read_text())
    assert isinstance(d["run_id"], str) and d["run_id"]


def test_retrieve_command_metadata_contains_dataset_path(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "run_001" / "retrieval_results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
        "--run-id", "r",
    ])

    d = json.loads((tmp_path / "run_001" / "retrieval_run_metadata.json").read_text())
    assert d["dataset_path"] == str(questions_path)


def test_retrieve_command_metadata_contains_top_k(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "run_001" / "retrieval_results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
        "--top-k", "3",
        "--run-id", "r",
    ])

    d = json.loads((tmp_path / "run_001" / "retrieval_run_metadata.json").read_text())
    assert d["top_k"] == 3


def test_retrieve_command_metadata_contains_chunk_size_and_overlap(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    output_path = tmp_path / "run_001" / "retrieval_results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(output_path),
        "--run-id", "r",
    ])

    d = json.loads((tmp_path / "run_001" / "retrieval_run_metadata.json").read_text())
    assert isinstance(d["chunk_size"], int)
    assert isinstance(d["chunk_overlap"], int)


def test_retrieve_command_auto_output_creates_run_001(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    runs_dir = tmp_path / "runs"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    exit_code = main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--runs-dir", str(runs_dir),
    ])

    assert exit_code == 0
    assert (runs_dir / "run_001" / "retrieval_results.jsonl").is_file()


def test_retrieve_command_auto_output_increments_run_number(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    runs_dir = tmp_path / "runs"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    (runs_dir / "run_001").mkdir(parents=True)
    (runs_dir / "run_002").mkdir()

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--runs-dir", str(runs_dir),
    ])

    assert (runs_dir / "run_003" / "retrieval_results.jsonl").is_file()


def test_retrieve_command_explicit_output_overrides_runs_dir(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    explicit_output = tmp_path / "custom" / "my_results.jsonl"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(explicit_output),
    ])

    assert explicit_output.is_file()


def test_retrieve_command_auto_output_prints_run_dir(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    embedding_cfg = tmp_path / "embedding.yaml"
    runs_dir = tmp_path / "runs"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_embedding_config(embedding_cfg)

    main([
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--runs-dir", str(runs_dir),
    ])

    out = capsys.readouterr().out
    assert "run_001" in out


# ---------------------------------------------------------------------------
# inspect-retrieval command helpers
# ---------------------------------------------------------------------------


def _build_retrieval_run(tmp_path: Path, *, with_examples: bool = False) -> Path:
    """Create a minimal run directory with retrieval_results.jsonl."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    results = [
        {
            "question_id": "q001",
            "query": "What was total revenue?",
            "retrieved": [
                {
                    "rank": 1,
                    "chunk_id": "DOC_p001_c000",
                    "doc_name": "DOC.pdf",
                    "page_num": 1,
                    "score": 0.82,
                    "text": "Revenue was $4.2 billion for the fiscal year.",
                }
            ],
        }
    ]
    (run_dir / "retrieval_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8"
    )

    if with_examples:
        examples = [
            {
                "question_id": "q001",
                "question": "What was total revenue for the year?",
                "evidence": [
                    {
                        "doc_name": "DOC.pdf",
                        "matched_page_num": 1,
                        "evidence_text": "Revenue was $4.2 billion for the fiscal year.",
                    }
                ],
            }
        ]
        examples_path = tmp_path / "examples.jsonl"
        examples_path.write_text(
            "\n".join(json.dumps(e) for e in examples) + "\n", encoding="utf-8"
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"run_id": "r", "dataset_path": str(examples_path)}),
            encoding="utf-8",
        )

    return run_dir


# ---------------------------------------------------------------------------
# inspect-retrieval command
# ---------------------------------------------------------------------------


def test_inspect_retrieval_exits_zero(tmp_path: Path, capsys) -> None:
    run_dir = _build_retrieval_run(tmp_path)

    exit_code = main([
        "inspect-retrieval",
        "--question-id", "q001",
        "--run", str(run_dir),
    ])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_inspect_retrieval_output_contains_question_text(tmp_path: Path, capsys) -> None:
    run_dir = _build_retrieval_run(tmp_path, with_examples=True)

    main([
        "inspect-retrieval",
        "--question-id", "q001",
        "--run", str(run_dir),
    ])

    out = capsys.readouterr().out
    assert "What was total revenue for the year?" in out


def test_inspect_retrieval_output_contains_chunk_score(tmp_path: Path, capsys) -> None:
    run_dir = _build_retrieval_run(tmp_path)

    main([
        "inspect-retrieval",
        "--question-id", "q001",
        "--run", str(run_dir),
    ])

    out = capsys.readouterr().out
    assert "0.82" in out


def test_inspect_retrieval_fails_when_run_dir_missing(tmp_path: Path, capsys) -> None:
    exit_code = main([
        "inspect-retrieval",
        "--question-id", "q001",
        "--run", str(tmp_path / "nonexistent"),
    ])

    assert exit_code == 1
    assert capsys.readouterr().err != ""


def test_inspect_retrieval_fails_when_question_id_not_found(tmp_path: Path, capsys) -> None:
    run_dir = _build_retrieval_run(tmp_path)

    exit_code = main([
        "inspect-retrieval",
        "--question-id", "q999",
        "--run", str(run_dir),
    ])

    assert exit_code == 1
    assert capsys.readouterr().err != ""


# ---------------------------------------------------------------------------
# --config pipeline helpers
# ---------------------------------------------------------------------------


def _write_pipeline_config(
    path: Path,
    *,
    pages_path: Path,
    chunks_path: Path,
    index_dir: Path,
    questions_path: Path,
    runs_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "retrieval:",
            f"  pages_path: {pages_path}",
            f"  chunks_path: {chunks_path}",
            f"  index_dir: {index_dir}",
            f"  questions_path: {questions_path}",
            f"  runs_dir: {runs_dir}",
            "  top_k: 3",
            "  evidence_overlap_threshold: 0.5",
            "  chunking:",
            "    strategy: recursive_text",
            "    chunk_size: 200",
            "    chunk_overlap: 20",
            "    min_chunk_chars: 10",
            "  embedding:",
            "    provider: mock",
            "    model_name: mock-embed",
            "    batch_size: 32",
        ]),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# chunk-documents command
# ---------------------------------------------------------------------------


def test_chunk_documents_exits_zero(tmp_path: Path, capsys) -> None:
    pages_path = tmp_path / "pages.jsonl"
    chunks_path = tmp_path / "chunks" / "chunks.jsonl"
    config_path = tmp_path / "retrieval.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_pipeline_config(
        config_path,
        pages_path=pages_path,
        chunks_path=chunks_path,
        index_dir=tmp_path / "idx",
        questions_path=tmp_path / "q.jsonl",
        runs_dir=tmp_path / "runs",
    )

    exit_code = main(["chunk-documents", "--config", str(config_path)])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_chunk_documents_writes_chunks_jsonl(tmp_path: Path, capsys) -> None:
    pages_path = tmp_path / "pages.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    config_path = tmp_path / "retrieval.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_pipeline_config(
        config_path,
        pages_path=pages_path,
        chunks_path=chunks_path,
        index_dir=tmp_path / "idx",
        questions_path=tmp_path / "q.jsonl",
        runs_dir=tmp_path / "runs",
    )

    main(["chunk-documents", "--config", str(config_path)])

    assert chunks_path.is_file()
    rows = _read_jsonl(chunks_path)
    assert len(rows) >= len(_SAMPLE_PAGES)
    assert all("chunk_id" in r and "text" in r and "doc_name" in r for r in rows)


def test_chunk_documents_output_has_page_provenance(tmp_path: Path, capsys) -> None:
    pages_path = tmp_path / "pages.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    config_path = tmp_path / "retrieval.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_pipeline_config(
        config_path,
        pages_path=pages_path,
        chunks_path=chunks_path,
        index_dir=tmp_path / "idx",
        questions_path=tmp_path / "q.jsonl",
        runs_dir=tmp_path / "runs",
    )

    main(["chunk-documents", "--config", str(config_path)])

    rows = _read_jsonl(chunks_path)
    assert all(isinstance(r["page_num"], int) for r in rows)
    doc_names = {r["doc_name"] for r in rows}
    assert "ACME_2022_10K.pdf" in doc_names


# ---------------------------------------------------------------------------
# build-index --config
# ---------------------------------------------------------------------------


def test_build_index_config_exits_zero(tmp_path: Path, capsys) -> None:
    pages_path = tmp_path / "pages.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "idx"
    config_path = tmp_path / "retrieval.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_pipeline_config(
        config_path,
        pages_path=pages_path,
        chunks_path=chunks_path,
        index_dir=index_dir,
        questions_path=tmp_path / "q.jsonl",
        runs_dir=tmp_path / "runs",
    )
    main(["chunk-documents", "--config", str(config_path)])

    exit_code = main(["build-index", "--config", str(config_path)])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_build_index_config_writes_index_files(tmp_path: Path, capsys) -> None:
    pages_path = tmp_path / "pages.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "idx"
    config_path = tmp_path / "retrieval.yaml"
    _write_pages_jsonl(pages_path, _SAMPLE_PAGES)
    _write_pipeline_config(
        config_path,
        pages_path=pages_path,
        chunks_path=chunks_path,
        index_dir=index_dir,
        questions_path=tmp_path / "q.jsonl",
        runs_dir=tmp_path / "runs",
    )
    main(["chunk-documents", "--config", str(config_path)])
    main(["build-index", "--config", str(config_path)])

    assert (index_dir / "index.faiss").is_file()
    assert (index_dir / "chunk_metadata.jsonl").is_file()
    assert (index_dir / "index_metadata.json").is_file()


# ---------------------------------------------------------------------------
# retrieve --config
# ---------------------------------------------------------------------------


def test_retrieve_config_exits_zero(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "q.jsonl"
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "retrieval.yaml"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_pipeline_config(
        config_path,
        pages_path=tmp_path / "pages.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=index_dir,
        questions_path=questions_path,
        runs_dir=runs_dir,
    )

    exit_code = main(["retrieve", "--config", str(config_path)])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_retrieve_config_writes_retrieval_run_metadata(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "q.jsonl"
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "retrieval.yaml"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_pipeline_config(
        config_path,
        pages_path=tmp_path / "pages.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=index_dir,
        questions_path=questions_path,
        runs_dir=runs_dir,
    )

    main(["retrieve", "--config", str(config_path)])

    run_dirs = sorted((runs_dir).iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "retrieval_run_metadata.json").is_file()


def test_retrieve_config_writes_config_yaml_to_run_dir(tmp_path: Path, capsys) -> None:
    index_dir = _build_test_index(tmp_path)
    questions_path = tmp_path / "q.jsonl"
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "retrieval.yaml"
    _write_questions_jsonl(questions_path, _SAMPLE_QUESTIONS)
    _write_pipeline_config(
        config_path,
        pages_path=tmp_path / "pages.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=index_dir,
        questions_path=questions_path,
        runs_dir=runs_dir,
    )

    main(["retrieve", "--config", str(config_path)])

    run_dirs = sorted(runs_dir.iterdir())
    assert (run_dirs[0] / "config.yaml").is_file()


# ---------------------------------------------------------------------------
# eval-retrieval command
# ---------------------------------------------------------------------------

_EVAL_GOLD_EXAMPLE = {
    "question_id": "q001",
    "company": "ACME",
    "doc_name": "ACME_2022_10K",
    "question": "What was capex?",
    "gold_answer": "$50M",
    "evidence": [
        {
            "doc_name": "ACME_2022_10K",
            "gold_page_num": 5,
            "matched_page_num": 5,
            "evidence_text": "capital expenditures were fifty million dollars",
            "page_text": "full page text here",
        }
    ],
}

_EVAL_RETRIEVAL_RESULT = {
    "question_id": "q001",
    "query": "What was capex?",
    "retrieved": [
        {
            "rank": 1,
            "chunk_id": "c1",
            "doc_name": "ACME_2022_10K.pdf",
            "page_num": 5,
            "score": 0.9,
            "text": "capital expenditures were fifty million dollars",
        }
    ],
}


def _setup_eval_retrieval(
    tmp_path: Path,
    *,
    run_id: str = "run_001",
    write_results: bool = True,
) -> tuple[Path, Path, Path]:
    """Create config, gold examples, and optionally a run dir with retrieval results.

    Returns (config_path, run_dir, reports_dir).
    """
    questions_path = tmp_path / "examples.jsonl"
    runs_dir = tmp_path / "runs"
    config_path = tmp_path / "retrieval.yaml"
    reports_dir = tmp_path / "reports"

    _write_questions_jsonl(questions_path, [_EVAL_GOLD_EXAMPLE])
    _write_pipeline_config(
        config_path,
        pages_path=tmp_path / "pages.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "idx",
        questions_path=questions_path,
        runs_dir=runs_dir,
    )

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    if write_results:
        _write_questions_jsonl(
            run_dir / "retrieval_results.jsonl", [_EVAL_RETRIEVAL_RESULT]
        )

    return config_path, run_dir, reports_dir


def test_eval_retrieval_exits_zero(tmp_path: Path, capsys) -> None:
    config_path, _run_dir, reports_dir = _setup_eval_retrieval(tmp_path)

    exit_code = main([
        "eval-retrieval",
        "--config", str(config_path),
        "--run-id", "run_001",
        "--report-dir", str(reports_dir),
    ])

    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_eval_retrieval_fails_when_results_missing(tmp_path: Path, capsys) -> None:
    config_path, _run_dir, reports_dir = _setup_eval_retrieval(tmp_path, write_results=False)

    exit_code = main([
        "eval-retrieval",
        "--config", str(config_path),
        "--run-id", "run_001",
        "--report-dir", str(reports_dir),
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not found" in captured.err


def test_eval_retrieval_writes_markdown_report(tmp_path: Path, capsys) -> None:
    config_path, _run_dir, reports_dir = _setup_eval_retrieval(tmp_path)

    main([
        "eval-retrieval",
        "--config", str(config_path),
        "--run-id", "run_001",
        "--report-dir", str(reports_dir),
    ])

    assert (reports_dir / "retrieval_eval_run_001.md").is_file()


def test_eval_retrieval_report_contains_key_metrics(tmp_path: Path, capsys) -> None:
    config_path, _run_dir, reports_dir = _setup_eval_retrieval(tmp_path)

    main([
        "eval-retrieval",
        "--config", str(config_path),
        "--run-id", "run_001",
        "--report-dir", str(reports_dir),
    ])

    content = (reports_dir / "retrieval_eval_run_001.md").read_text()
    assert "run_001" in content
    assert "num_questions" in content
    assert "doc_hit@k" in content
    assert "embedding_provider" in content
