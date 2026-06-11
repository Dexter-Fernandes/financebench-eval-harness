from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from financebench_eval_harness.config import DatasetConfig


DEFAULT_DATA_ROOT = Path("data/raw/financebench")
QUESTIONS_FILENAME = "questions.jsonl"
DOCUMENTS_DIRNAME = "documents"
PROCESSED_EXAMPLES_FILENAME = "examples.jsonl"
REJECTED_EXAMPLES_FILENAME = "examples.rejected.jsonl"
TEXT_NORMALIZER_ID = "financebench_text_v1"
PAGE_NUMBER_MISMATCH_WARNING = "matched_page_num_differs_from_gold_page_num"

SKIP_MISSING_EVIDENCE = "missing_evidence"
SKIP_MALFORMED_EVIDENCE = "malformed_evidence"
SKIP_MISSING_EXTRACTED_DOCUMENT = "missing_extracted_document"
SKIP_MISSING_EXTRACTED_PAGE = "missing_extracted_page"
SKIP_EVIDENCE_TEXT_NOT_FOUND = "evidence_text_not_found_in_page_text"


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


@dataclass(frozen=True)
class FinanceBenchEvidence:
    """Normalized evidence metadata for one FinanceBench example."""

    doc_name: str
    page_num: int
    text: str


@dataclass(frozen=True)
class FinanceBenchExample:
    """Normalized FinanceBench question/answer record."""

    question_id: str
    company: str
    doc_name: str
    question: str
    gold_answer: str
    evidence_doc_name: str
    evidence_page_num: int
    evidence_text: str
    evidence: tuple[FinanceBenchEvidence, ...]


@dataclass(frozen=True)
class DatasetValidationIssue:
    """One schema validation issue found in a FinanceBench dataset."""

    line_number: int
    question_id: str | None
    field: str
    message: str

    def format(self) -> str:
        question_id = self.question_id or "unknown"
        return f"line {self.line_number} [{question_id}] {self.message}"


@dataclass(frozen=True)
class DatasetValidationResult:
    """Aggregate schema validation result for a FinanceBench dataset."""

    valid_count: int
    invalid_count: int
    issues: tuple[DatasetValidationIssue, ...]


@dataclass(frozen=True)
class DocumentRegistryValidationResult:
    """Document registry coverage for FinanceBench evidence documents."""

    registry: dict[str, Path]
    resolved_documents: dict[str, Path]
    missing_documents: tuple[str, ...]
    unused_documents: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_documents


@dataclass(frozen=True)
class DocumentPage:
    """Extracted text for one document page."""

    doc_name: str
    page_num: int
    text: str


@dataclass(frozen=True)
class DocumentExtractionFailure:
    """A document or page extraction failure."""

    doc_name: str
    page_num: int | None
    error: str


@dataclass(frozen=True)
class DocumentExtractionResult:
    """Summary of an extraction run."""

    page_count: int
    document_count: int
    failure_count: int
    output_path: Path
    failures_path: Path


@dataclass(frozen=True)
class EvidencePageCheck:
    """Validation outcome for one FinanceBench evidence item."""

    question_id: str
    evidence_index: int
    document_filename: str
    document_path: Path | None
    evidence_page_num: int
    extracted_page_num: int
    is_match: bool
    reason: str
    evidence_excerpt: str
    page_excerpt: str
    match_method: str = "alphanumeric_substring"


@dataclass(frozen=True)
class EvidencePageValidationResult:
    """Aggregate validation result for FinanceBench evidence-page links."""

    checks: tuple[EvidencePageCheck, ...]

    @property
    def total_count(self) -> int:
        return len(self.checks)

    @property
    def matched_count(self) -> int:
        return sum(1 for check in self.checks if check.is_match)

    @property
    def mismatch_count(self) -> int:
        return self.total_count - self.matched_count

    @property
    def is_valid(self) -> bool:
        return self.mismatch_count == 0


@dataclass(frozen=True)
class ProcessedExamplesBuildResult:
    """Summary of a canonical processed examples build."""

    accepted_count: int
    rejected_count: int
    output_path: Path
    rejected_path: Path
    skip_reason_counts: dict[str, int]


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


class FinanceBenchQuestionLoadError(ValueError):
    """Raised when FinanceBench question records cannot be loaded."""


class DocumentExtractionError(ValueError):
    """Raised when document extraction cannot be started."""


class DocumentPageLoadError(ValueError):
    """Raised when extracted document pages cannot be loaded."""


class ProcessedFinanceBenchExampleLoadError(ValueError):
    """Raised when canonical processed examples cannot be loaded."""
