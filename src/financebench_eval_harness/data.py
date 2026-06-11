from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data/raw/financebench")
QUESTIONS_FILENAME = "questions.jsonl"
DOCUMENTS_DIRNAME = "documents"


@dataclass(frozen=True)
class FinanceBenchDataLayout:
    """Expected local FinanceBench data paths."""

    root: Path = DEFAULT_DATA_ROOT

    @property
    def questions_path(self) -> Path:
        return self.root / QUESTIONS_FILENAME

    @property
    def documents_path(self) -> Path:
        return self.root / DOCUMENTS_DIRNAME

    def expected_layout_message(self) -> str:
        return (
            "Expected:\n"
            f"  {self.questions_path}\n"
            f"  {self.documents_path}/"
        )


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
            "and source documents under documents/, or pass --data-root PATH."
        )


def validate_financebench_data_layout(
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> FinanceBenchDataLayout:
    """Validate the expected local FinanceBench dataset layout."""

    layout = FinanceBenchDataLayout(Path(data_root))
    missing_paths: list[Path] = []

    if not layout.questions_path.is_file():
        missing_paths.append(layout.questions_path)

    if not layout.documents_path.is_dir():
        missing_paths.append(layout.documents_path)

    if missing_paths:
        raise MissingFinanceBenchDataError(layout, missing_paths)

    return layout
