from __future__ import annotations

import argparse
import json
from dataclasses import replace
import sys
from pathlib import Path
from typing import Sequence

from financebench_eval_harness.config import (
    DEFAULT_DATASET_CONFIG_PATH,
    DatasetConfig,
    DatasetConfigError,
    load_dataset_config,
)
from financebench_eval_harness.data import (
    DocumentExtractionError,
    DocumentPageLoadError,
    EvidencePageCheck,
    FinanceBenchQuestionLoadError,
    MissingFinanceBenchDataError,
    build_processed_financebench_examples,
    extract_financebench_documents,
    load_financebench_examples,
    validate_financebench_evidence_pages,
    validate_financebench_document_registry,
    validate_financebench_dataset,
    validate_financebench_data_layout,
)
from financebench_eval_harness.llm import (
    LLMClient,
    LLMConfigError,
    MockLLMClient,
    OllamaClient,
)
from financebench_eval_harness.run import EvaluationRunError, run_evaluation_from_config
from financebench_eval_harness.run_config import (
    DEFAULT_EVALUATION_RUN_CONFIG_PATH,
    EvaluationRunConfigError,
    load_evaluation_run_config,
)
from financebench_eval_harness.report import (
    DEFAULT_REPORT_OUTPUT_DIR,
    BaselineReportError,
    generate_baseline_report,
)


DEFAULT_BASELINE_RUN_CONFIG_PATH = Path("configs/baseline_closed_book.yaml")
DEFAULT_BASELINE_REPORT_OUTPUT_DIR = Path("reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financebench-harness",
        description="Utilities for the FinanceBench evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-data",
        help="Validate the expected local FinanceBench data layout.",
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_PATH,
        help="Dataset config YAML path.",
    )
    validate_parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing questions.jsonl and documents/.",
    )

    dataset_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate the FinanceBench question schema.",
    )
    _add_dataset_path_arguments(dataset_parser)

    documents_parser = subparsers.add_parser(
        "validate-documents",
        help="Validate FinanceBench evidence document file coverage.",
    )
    _add_dataset_path_arguments(documents_parser)

    extract_parser = subparsers.add_parser(
        "extract-documents",
        help="Extract local FinanceBench PDF text page by page.",
    )
    _add_dataset_path_arguments(extract_parser)

    evidence_pages_parser = subparsers.add_parser(
        "validate-evidence-pages",
        help="Validate FinanceBench evidence text against extracted document pages.",
    )
    _add_dataset_path_arguments(evidence_pages_parser)

    build_examples_parser = subparsers.add_parser(
        "build-examples",
        help="Build canonical processed FinanceBench examples JSONL.",
    )
    _add_dataset_path_arguments(build_examples_parser)

    run_eval_parser = subparsers.add_parser(
        "run-eval",
        help="Run a configured baseline evaluation.",
    )
    run_eval_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EVALUATION_RUN_CONFIG_PATH,
        help="Evaluation run config YAML path.",
    )
    run_eval_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional deterministic run directory name.",
    )
    run_eval_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Override the configured example limit for smoke tests.",
    )

    run_baseline_parser = subparsers.add_parser(
        "run-baseline",
        help="Run a baseline evaluation and generate its Markdown report.",
    )
    run_baseline_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASELINE_RUN_CONFIG_PATH,
        help="Baseline run config YAML path.",
    )
    run_baseline_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional deterministic run directory name.",
    )
    run_baseline_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Override the configured example limit for smoke tests.",
    )
    run_baseline_parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_BASELINE_REPORT_OUTPUT_DIR,
        help="Directory where the generated Markdown report should be written.",
    )

    report_parser = subparsers.add_parser(
        "report-baseline",
        help="Generate a Markdown baseline report for an evaluation run.",
    )
    report_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Evaluation run directory containing run_metadata.json, predictions.jsonl, and scores.jsonl.",
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT_DIR,
        help="Directory where the generated Markdown report should be written.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-data":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            layout = validate_financebench_data_layout(dataset_config)
            examples = load_financebench_examples(dataset_config)
        except (
            DatasetConfigError,
            FinanceBenchQuestionLoadError,
            MissingFinanceBenchDataError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"FinanceBench data layout is valid: {layout.root}")
        print(f"Loaded {len(examples)} FinanceBench examples.")
        return 0

    if args.command == "validate-dataset":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = validate_financebench_dataset(dataset_config)
        except (DatasetConfigError, FinanceBenchQuestionLoadError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Valid examples: {result.valid_count}")
        print(f"Invalid examples: {result.invalid_count}")
        if result.invalid_count:
            print("Dataset schema validation failed.")
            for issue in result.issues:
                print(issue.format())
            return 1

        print("Dataset schema validation passed.")
        return 0

    if args.command == "validate-documents":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = validate_financebench_document_registry(dataset_config)
        except (DatasetConfigError, FinanceBenchQuestionLoadError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Resolved documents: {len(result.resolved_documents)}")
        print(f"Missing documents: {len(result.missing_documents)}")
        print(f"Unused documents: {len(result.unused_documents)}")

        if result.missing_documents:
            print("Document registry validation failed.")
            for filename in result.missing_documents:
                print(f"missing {filename}")
            return 1

        if result.unused_documents:
            print("Unused local documents:")
            for filename in result.unused_documents:
                print(f"unused {filename}")

        print("Document registry validation passed.")
        return 0

    if args.command == "extract-documents":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = extract_financebench_documents(
                dataset_config,
                on_document_start=lambda path: print(f"Extracting {path.name}"),
            )
        except (DatasetConfigError, DocumentExtractionError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Extracted {result.document_count} documents.")
        print(f"Wrote {result.page_count} pages to {result.output_path}.")
        print(f"Extraction failures: {result.failure_count}")
        if result.failure_count:
            print(f"Failure details written to {result.failures_path}.")
        return 0

    if args.command == "validate-evidence-pages":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = validate_financebench_evidence_pages(dataset_config)
        except (
            DatasetConfigError,
            DocumentPageLoadError,
            FinanceBenchQuestionLoadError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        for check in result.checks:
            print(_format_evidence_page_check(check))

        print(f"Total evidence checks: {result.total_count}")
        print(f"Matches: {result.matched_count}")
        print(f"Mismatches: {result.mismatch_count}")

        if result.is_valid:
            print("Evidence page validation passed.")
            return 0

        print("Evidence page validation failed.")
        return 1

    if args.command == "build-examples":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            result = build_processed_financebench_examples(dataset_config)
        except (
            DatasetConfigError,
            DocumentPageLoadError,
            FinanceBenchQuestionLoadError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Accepted examples: {result.accepted_count}")
        print(f"Rejected examples: {result.rejected_count}")
        for reason, count in result.skip_reason_counts.items():
            print(f"{reason}: {count}")
        print(f"Wrote accepted examples to {result.output_path.resolve()}.")
        print(f"Wrote rejected examples to {result.rejected_path.resolve()}.")
        return 0

    if args.command == "run-eval":
        try:
            result = _run_configured_evaluation(
                config_path=args.config,
                run_id=args.run_id,
                limit=args.limit,
            )
        except (
            EvaluationRunConfigError,
            EvaluationRunError,
            LLMConfigError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Evaluation run output: {result.output_dir}")
        print(f"Wrote config snapshot to {result.config_path}")
        print(f"Wrote {result.example_count} predictions to {result.predictions_path}")
        print(f"Wrote {result.example_count} score rows to {result.scores_path}")
        print(f"Wrote run metadata to {result.run_metadata_path}")
        print(f"Attempted: {result.attempted_count}")
        print(f"Succeeded: {result.success_count}")
        print(f"Errors: {result.error_count}")
        run_metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
        judge_summary = run_metadata["judge_summary"]
        print(f"Judge attempted: {judge_summary['attempted_count']}")
        print(f"Judge succeeded: {judge_summary['success_count']}")
        print(f"Judge errors: {judge_summary['error_count']}")
        return 0

    if args.command == "run-baseline":
        try:
            result = _run_configured_evaluation(
                config_path=args.config,
                run_id=args.run_id,
                limit=args.limit,
            )
            report = generate_baseline_report(
                result.output_dir,
                output_dir=args.report_dir,
                report_filename=f"baseline_{result.output_dir.name}.md",
            )
            _record_report_path(result.run_metadata_path, report.report_path)
            run_metadata = json.loads(
                result.run_metadata_path.read_text(encoding="utf-8")
            )
            judge_summary = run_metadata["judge_summary"]
        except (
            BaselineReportError,
            EvaluationRunConfigError,
            EvaluationRunError,
            LLMConfigError,
            OSError,
            KeyError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Baseline run output: {result.output_dir}")
        print(f"Wrote config snapshot to {result.config_path}")
        print(f"Wrote {result.example_count} predictions to {result.predictions_path}")
        print(f"Wrote {result.example_count} score rows to {result.scores_path}")
        print(f"Wrote run metadata to {result.run_metadata_path}")
        print(f"Baseline report: {report.report_path}")
        print(f"Attempted: {result.attempted_count}")
        print(f"Succeeded: {result.success_count}")
        print(f"Errors: {result.error_count}")
        print(f"Correct: {judge_summary['correct_count']}")
        print(f"Partially correct: {judge_summary['partially_correct_count']}")
        print(f"Incorrect: {judge_summary['incorrect_count']}")
        return 0

    if args.command == "report-baseline":
        try:
            result = generate_baseline_report(
                args.run_dir,
                output_dir=args.output_dir,
            )
            run_metadata = json.loads(
                (args.run_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            judge_summary = run_metadata["judge_summary"]
        except (BaselineReportError, OSError, KeyError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"Baseline report: {result.report_path}")
        print(f"Questions evaluated: {result.evaluated_count}")
        print(f"Correct: {judge_summary['correct_count']}")
        print(f"Partially correct: {judge_summary['partially_correct_count']}")
        print(f"Incorrect: {judge_summary['incorrect_count']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_dataset_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_PATH,
        help="Dataset config YAML path.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing questions.jsonl and documents/.",
    )


def _resolve_dataset_config(config_path: Path, data_root: Path | None) -> DatasetConfig:
    if data_root is not None:
        return DatasetConfig.from_data_root(data_root)
    return load_dataset_config(config_path)


def _run_configured_evaluation(
    *,
    config_path: Path,
    run_id: str | None,
    limit: int | None,
):
    run_config = load_evaluation_run_config(config_path)
    if limit is not None:
        run_config = replace(
            run_config,
            settings=replace(run_config.settings, limit=limit),
        )
    llm_client = _build_llm_client(run_config)
    judge_client = _build_judge_client(run_config)
    return run_evaluation_from_config(
        run_config,
        llm_client,
        judge_client=judge_client,
        run_id=run_id,
    )


def _record_report_path(run_metadata_path: Path, report_path: Path) -> None:
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise BaselineReportError(
            f"Run metadata must be a JSON object: {run_metadata_path}"
        )
    metadata["report_path"] = str(report_path)
    run_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_llm_client(run_config) -> LLMClient:
    if run_config.model.provider == "mock":
        return MockLLMClient(
            run_config.model,
            responses=["mock response"] * run_config.settings.limit,
        )
    if run_config.model.provider == "ollama":
        return OllamaClient(run_config.model)
    raise LLMConfigError(f"Unsupported LLM provider: {run_config.model.provider}")


def _build_judge_client(run_config) -> LLMClient | None:
    if run_config.judge is None:
        return None
    if run_config.judge.model.provider == "mock":
        return MockLLMClient(
            run_config.judge.model,
            responses=[
                '{"verdict": "incorrect", "reason": "Mock judge response."}'
            ]
            * run_config.settings.limit,
        )
    if run_config.judge.model.provider == "ollama":
        return OllamaClient(run_config.judge.model)
    raise LLMConfigError(
        f"Unsupported judge LLM provider: {run_config.judge.model.provider}"
    )


def _positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def _format_evidence_page_check(check: EvidencePageCheck) -> str:
    status = "MATCH" if check.is_match else "MISMATCH"
    document_path = str(check.document_path) if check.document_path is not None else "missing"
    return (
        f"{status} {check.question_id} evidence[{check.evidence_index}] "
        f"doc={check.document_filename} "
        f"path={document_path} "
        f"evidence_page={check.evidence_page_num} "
        f"extracted_page={check.extracted_page_num} "
        f"reason={check.reason} "
        f"method={check.match_method} "
        f"evidence={_quote_excerpt(check.evidence_excerpt)} "
        f"page={_quote_excerpt(check.page_excerpt)}"
    )


def _quote_excerpt(text: str) -> str:
    safe_text = text.replace('"', "'")
    return f'"{safe_text}"'


if __name__ == "__main__":
    raise SystemExit(main())
