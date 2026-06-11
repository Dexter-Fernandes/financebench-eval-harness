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
    MissingFinanceBenchDataError,
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-data":
        try:
            dataset_config = _resolve_dataset_config(args.config, args.data_root)
            layout = validate_financebench_data_layout(dataset_config)
        except (DatasetConfigError, MissingFinanceBenchDataError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"FinanceBench data layout is valid: {layout.root}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _resolve_dataset_config(config_path: Path, data_root: Path | None) -> DatasetConfig:
    if data_root is not None:
        return DatasetConfig.from_data_root(data_root)
    return load_dataset_config(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
