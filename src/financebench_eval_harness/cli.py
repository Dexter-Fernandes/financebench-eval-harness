from __future__ import annotations

import argparse
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
    FinanceBenchQuestionLoadError,
    MissingFinanceBenchDataError,
    extract_financebench_documents,
    load_financebench_examples,
    validate_financebench_document_registry,
    validate_financebench_dataset,
    validate_financebench_data_layout,
)


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


if __name__ == "__main__":
    raise SystemExit(main())
