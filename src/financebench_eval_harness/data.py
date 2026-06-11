from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from financebench_eval_harness.config import DatasetConfig


DEFAULT_DATA_ROOT = Path("data/raw/financebench")
QUESTIONS_FILENAME = "questions.jsonl"
DOCUMENTS_DIRNAME = "documents"


@dataclass(frozen=True)
class FinanceBenchDataLayout:
    """Expected local FinanceBench data paths."""

    config: DatasetConfig

    @property
    def questions_path(self) -> Path:
        return self.config.questions_path

    @property
    def documents_path(self) -> Path:
        return self.config.documents_dir

    @property
    def processed_dir(self) -> Path:
        return self.config.processed_dir

    @property
    def root(self) -> Path:
        return self.questions_path.parent

    def expected_layout_message(self) -> str:
        return self.config.expected_layout_message()


class MissingFinanceBenchDataError(FileNotFoundError):
    """Raised when the expected local FinanceBench data layout is missing."""

    def __init__(self, layout: FinanceBenchDataLayout, missing_paths: list[Path]) -> None:
        self.layout = layout
        self.missing_paths = missing_paths
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        super().__init__(
            "FinanceBench data is missing.\n\n"
            f"{layout.expected_layout_message()}\n\n"
            "Missing:\n"
            f"{missing}\n\n"
            "Place the public FinanceBench sample question file at questions.jsonl "
            "and source documents under documents/, or pass --config PATH or "
            "--data-root PATH."
        )


def validate_financebench_data_layout(
    dataset_config: DatasetConfig | str | Path | None = None,
) -> FinanceBenchDataLayout:
    """Validate the expected local FinanceBench dataset layout."""

    if dataset_config is None:
        config = DatasetConfig.from_data_root(DEFAULT_DATA_ROOT)
    elif isinstance(dataset_config, DatasetConfig):
        config = dataset_config
    else:
        config = DatasetConfig.from_data_root(dataset_config)

    layout = FinanceBenchDataLayout(config)
    missing_paths: list[Path] = []

    if not config.questions_path.is_file():
        missing_paths.append(config.questions_path)

    if not config.documents_dir.is_dir():
        missing_paths.append(config.documents_dir)

    if missing_paths:
        raise MissingFinanceBenchDataError(layout, missing_paths)

    return layout
