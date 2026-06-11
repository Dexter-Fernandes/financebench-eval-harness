from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable

from financebench_eval_harness.config import DatasetConfig, load_dataset_config


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


def load_financebench_examples(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> list[FinanceBenchExample]:
    """Load and normalize FinanceBench question records from configured JSONL."""

    config = _resolve_dataset_config(dataset_config_or_path)
    questions_path = config.questions_path

    if not questions_path.is_file():
        raise FinanceBenchQuestionLoadError(
            f"FinanceBench questions file not found: {questions_path}"
        )

    examples: list[FinanceBenchExample] = []
    try:
        with questions_path.open(encoding="utf-8") as questions_file:
            for line_number, line in enumerate(questions_file, start=1):
                if not line.strip():
                    continue
                examples.append(_parse_financebench_example(line, line_number))
    except OSError as exc:
        raise FinanceBenchQuestionLoadError(
            f"Could not read FinanceBench questions file: {questions_path}"
        ) from exc

    return examples


def validate_financebench_dataset(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> DatasetValidationResult:
    """Validate configured FinanceBench question records without raising per-row errors."""

    config = _resolve_dataset_config(dataset_config_or_path)
    questions_path = config.questions_path

    if not questions_path.is_file():
        raise FinanceBenchQuestionLoadError(
            f"FinanceBench questions file not found: {questions_path}"
        )

    valid_count = 0
    invalid_count = 0
    issues: list[DatasetValidationIssue] = []
    seen_question_ids: set[str] = set()

    try:
        with questions_path.open(encoding="utf-8") as questions_file:
            for line_number, line in enumerate(questions_file, start=1):
                if not line.strip():
                    continue

                try:
                    example = _parse_financebench_example(line, line_number)
                except FinanceBenchQuestionLoadError as exc:
                    invalid_count += 1
                    issues.append(_issue_from_load_error(line, line_number, exc))
                    continue

                if example.question_id in seen_question_ids:
                    invalid_count += 1
                    issues.append(
                        DatasetValidationIssue(
                            line_number=line_number,
                            question_id=example.question_id,
                            field="financebench_id",
                            message="duplicate question_id",
                        )
                    )
                    continue

                seen_question_ids.add(example.question_id)
                valid_count += 1
    except OSError as exc:
        raise FinanceBenchQuestionLoadError(
            f"Could not read FinanceBench questions file: {questions_path}"
        ) from exc

    return DatasetValidationResult(
        valid_count=valid_count,
        invalid_count=invalid_count,
        issues=tuple(issues),
    )


def build_document_registry(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> dict[str, Path]:
    """Map local document filenames to their paths."""

    config = _resolve_dataset_config(dataset_config_or_path)
    documents_dir = config.documents_dir

    if not documents_dir.is_dir():
        raise FinanceBenchQuestionLoadError(
            f"FinanceBench documents directory not found: {documents_dir}"
        )

    return {
        path.name: path
        for path in sorted(documents_dir.iterdir())
        if path.is_file()
    }


def validate_financebench_document_registry(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> DocumentRegistryValidationResult:
    """Validate evidence document references against local document files."""

    config = _resolve_dataset_config(dataset_config_or_path)
    examples = load_financebench_examples(config)
    registry = build_document_registry(config)
    required_filenames = {
        _document_filename(evidence.doc_name)
        for example in examples
        for evidence in example.evidence
    }
    missing_documents = tuple(
        sorted(filename for filename in required_filenames if filename not in registry)
    )
    unused_documents = tuple(
        sorted(filename for filename in registry if filename not in required_filenames)
    )
    resolved_documents = {
        filename: registry[filename]
        for filename in sorted(required_filenames)
        if filename in registry
    }

    return DocumentRegistryValidationResult(
        registry=registry,
        resolved_documents=resolved_documents,
        missing_documents=missing_documents,
        unused_documents=unused_documents,
    )


def validate_financebench_evidence_pages(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> EvidencePageValidationResult:
    """Validate evidence document and page references against extracted page text."""

    config = _resolve_dataset_config(dataset_config_or_path)
    examples = load_financebench_examples(config)
    registry = build_document_registry(config)
    page_index = _load_document_page_index(config.processed_dir / "pages.jsonl")

    checks: list[EvidencePageCheck] = []
    for example in examples:
        for evidence_index, evidence in enumerate(example.evidence, start=1):
            document_filename = _document_filename(evidence.doc_name)
            document_path = registry.get(document_filename)
            extracted_page_num = evidence.page_num + 1
            evidence_excerpt = _text_excerpt(evidence.text)

            if document_path is None:
                checks.append(
                    EvidencePageCheck(
                        question_id=example.question_id,
                        evidence_index=evidence_index,
                        document_filename=document_filename,
                        document_path=None,
                        evidence_page_num=evidence.page_num,
                        extracted_page_num=extracted_page_num,
                        is_match=False,
                        reason="missing_document",
                        evidence_excerpt=evidence_excerpt,
                        page_excerpt="",
                    )
                )
                continue

            page_text = page_index.get((document_filename, extracted_page_num))
            if page_text is None:
                checks.append(
                    EvidencePageCheck(
                        question_id=example.question_id,
                        evidence_index=evidence_index,
                        document_filename=document_filename,
                        document_path=document_path,
                        evidence_page_num=evidence.page_num,
                        extracted_page_num=extracted_page_num,
                        is_match=False,
                        reason="missing_page",
                        evidence_excerpt=evidence_excerpt,
                        page_excerpt="",
                    )
                )
                continue

            is_match = _alphanumeric_substring_match(evidence.text, page_text)
            checks.append(
                EvidencePageCheck(
                    question_id=example.question_id,
                    evidence_index=evidence_index,
                    document_filename=document_filename,
                    document_path=document_path,
                    evidence_page_num=evidence.page_num,
                    extracted_page_num=extracted_page_num,
                    is_match=is_match,
                    reason="matched" if is_match else "text_mismatch",
                    evidence_excerpt=evidence_excerpt,
                    page_excerpt=_text_excerpt(page_text),
                )
            )

    return EvidencePageValidationResult(checks=tuple(checks))


def canonical_pdf_name(doc_name: str) -> str:
    """Return the canonical PDF filename for a FinanceBench document name."""

    if doc_name.endswith(".pdf"):
        return doc_name
    return f"{doc_name}.pdf"


def build_processed_financebench_examples(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> ProcessedExamplesBuildResult:
    """Build canonical accepted and rejected FinanceBench example JSONL files."""

    config = _resolve_dataset_config(dataset_config_or_path)
    page_index = _load_document_page_index(config.processed_dir / "pages.jsonl")
    pages_by_document = _group_pages_by_document(page_index)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    skip_reason_counts: dict[str, int] = {}

    for line_number, row in _read_raw_question_rows(config.questions_path):
        metadata = _processed_metadata_from_raw_row(row, line_number)
        raw_evidence = row.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            skip_reason = SKIP_MISSING_EVIDENCE
            rejected_rows.append(
                _processed_rejected_row(
                    metadata=metadata,
                    skip_reason=skip_reason,
                    evidence_rows=[],
                )
            )
            _increment_count(skip_reason_counts, skip_reason)
            continue

        evidence_rows: list[dict[str, Any]] = []
        evidence_skip_reasons: list[str] = []
        for raw_evidence_index, evidence_item in enumerate(raw_evidence):
            processed_evidence, skip_reason = _processed_evidence_from_raw_item(
                item=evidence_item,
                raw_evidence_index=raw_evidence_index,
                page_index=page_index,
                pages_by_document=pages_by_document,
            )
            evidence_rows.append(processed_evidence)
            if skip_reason is not None:
                evidence_skip_reasons.append(skip_reason)

        if evidence_skip_reasons:
            skip_reason = evidence_skip_reasons[0]
            rejected_rows.append(
                _processed_rejected_row(
                    metadata=metadata,
                    skip_reason=skip_reason,
                    evidence_rows=evidence_rows,
                )
            )
            _increment_count(skip_reason_counts, skip_reason)
            continue

        accepted_rows.append(_processed_accepted_row(metadata, evidence_rows))

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.processed_dir / PROCESSED_EXAMPLES_FILENAME
    rejected_path = config.processed_dir / REJECTED_EXAMPLES_FILENAME
    _write_jsonl_records(output_path, accepted_rows)
    _write_jsonl_records(rejected_path, rejected_rows)

    return ProcessedExamplesBuildResult(
        accepted_count=len(accepted_rows),
        rejected_count=len(rejected_rows),
        output_path=output_path,
        rejected_path=rejected_path,
        skip_reason_counts=skip_reason_counts,
    )


def load_processed_financebench_examples(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load canonical FinanceBench examples from processed JSONL only."""

    config = _resolve_dataset_config(dataset_config_or_path)
    examples_path = config.processed_dir / PROCESSED_EXAMPLES_FILENAME
    if not examples_path.is_file():
        raise ProcessedFinanceBenchExampleLoadError(
            f"Processed FinanceBench examples file not found: {examples_path}. "
            "Run `financebench-harness build-examples` first."
        )

    examples: list[dict[str, Any]] = []
    try:
        with examples_path.open(encoding="utf-8") as examples_file:
            for line_number, line in enumerate(examples_file, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except JSONDecodeError as exc:
                    raise ProcessedFinanceBenchExampleLoadError(
                        f"Invalid processed examples JSONL at line {line_number}: {exc.msg}"
                    ) from exc

                _validate_processed_example_row(row, line_number)
                examples.append(row)
    except OSError as exc:
        raise ProcessedFinanceBenchExampleLoadError(
            f"Could not read processed examples file: {examples_path}"
        ) from exc

    return examples


def extract_document_pages(
    document_path: Path,
    pdf_reader_factory: Any | None = None,
) -> tuple[list[DocumentPage], list[DocumentExtractionFailure]]:
    """Extract page text from one PDF document."""

    reader_factory = pdf_reader_factory or _pdf_reader_class()
    doc_name = document_path.name

    try:
        reader = reader_factory(str(document_path))
    except Exception as exc:
        return [], [
            DocumentExtractionFailure(
                doc_name=doc_name,
                page_num=None,
                error=str(exc),
            )
        ]

    pages: list[DocumentPage] = []
    failures: list[DocumentExtractionFailure] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            failures.append(
                DocumentExtractionFailure(
                    doc_name=doc_name,
                    page_num=page_index,
                    error=str(exc),
                )
            )
            continue

        pages.append(DocumentPage(doc_name=doc_name, page_num=page_index, text=text))

    return pages, failures


def extract_financebench_documents(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
    pdf_reader_factory: Any | None = None,
    on_document_start: Callable[[Path], None] | None = None,
) -> DocumentExtractionResult:
    """Extract local FinanceBench PDFs into processed page JSONL."""

    config = _resolve_dataset_config(dataset_config_or_path)
    if not config.documents_dir.is_dir():
        raise DocumentExtractionError(
            f"FinanceBench documents directory not found: {config.documents_dir}"
        )

    document_paths = sorted(config.documents_dir.glob("*.pdf"))
    if not document_paths:
        raise DocumentExtractionError(
            f"No PDF documents found in FinanceBench documents directory: {config.documents_dir}"
        )

    pages: list[DocumentPage] = []
    failures: list[DocumentExtractionFailure] = []
    for document_path in document_paths:
        if on_document_start is not None:
            on_document_start(document_path)
        document_pages, document_failures = extract_document_pages(
            document_path,
            pdf_reader_factory=pdf_reader_factory,
        )
        pages.extend(document_pages)
        failures.extend(document_failures)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.processed_dir / "pages.jsonl"
    failures_path = config.processed_dir / "extraction_failures.jsonl"
    _write_document_pages(output_path, pages)
    _write_document_failures(failures_path, failures)

    return DocumentExtractionResult(
        page_count=len(pages),
        document_count=len(document_paths),
        failure_count=len(failures),
        output_path=output_path,
        failures_path=failures_path,
    )


def _resolve_dataset_config(
    dataset_config_or_path: DatasetConfig | str | Path | None,
) -> DatasetConfig:
    if dataset_config_or_path is None:
        return load_dataset_config()
    if isinstance(dataset_config_or_path, DatasetConfig):
        return dataset_config_or_path
    return load_dataset_config(dataset_config_or_path)


def _document_filename(doc_name: str) -> str:
    return canonical_pdf_name(doc_name)


def _read_raw_question_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FinanceBenchQuestionLoadError(
            f"FinanceBench questions file not found: {path}"
        )

    rows: list[tuple[int, dict[str, Any]]] = []
    try:
        with path.open(encoding="utf-8") as questions_file:
            for line_number, line in enumerate(questions_file, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except JSONDecodeError as exc:
                    raise FinanceBenchQuestionLoadError(
                        f"Invalid FinanceBench JSONL at line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(row, dict):
                    raise FinanceBenchQuestionLoadError(
                        f"Invalid FinanceBench row at line {line_number}: expected object"
                    )

                rows.append((line_number, row))
    except OSError as exc:
        raise FinanceBenchQuestionLoadError(
            f"Could not read FinanceBench questions file: {path}"
        ) from exc

    return rows


def _processed_metadata_from_raw_row(
    row: dict[str, Any],
    line_number: int,
) -> dict[str, str]:
    return {
        "question_id": _required_string(row, "financebench_id", line_number),
        "question": _required_string(row, "question", line_number),
        "gold_answer": _required_string(row, "answer", line_number),
        "company": _required_string(row, "company", line_number),
        "doc_name": _required_string(row, "doc_name", line_number),
    }


def _group_pages_by_document(
    page_index: dict[tuple[str, int], str],
) -> dict[str, list[int]]:
    pages_by_document: dict[str, list[int]] = {}
    for doc_name, page_num in page_index:
        pages_by_document.setdefault(doc_name, []).append(page_num)

    return {
        doc_name: sorted(page_nums)
        for doc_name, page_nums in pages_by_document.items()
    }


def _processed_evidence_from_raw_item(
    *,
    item: Any,
    raw_evidence_index: int,
    page_index: dict[tuple[str, int], str],
    pages_by_document: dict[str, list[int]],
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(item, dict):
        return (
            _processed_evidence_row(
                raw_evidence_index=raw_evidence_index,
                doc_name="",
                gold_page_num=None,
                matched_page_num=None,
                evidence_text="",
                page_text="",
                match_status=SKIP_MALFORMED_EVIDENCE,
            ),
            SKIP_MALFORMED_EVIDENCE,
        )

    doc_name = item.get("doc_name")
    evidence_text = item.get("evidence_text")
    gold_page_num = item.get("evidence_page_num")
    if (
        not isinstance(doc_name, str)
        or not doc_name.strip()
        or not isinstance(evidence_text, str)
        or not evidence_text.strip()
        or type(gold_page_num) is not int
    ):
        return (
            _processed_evidence_row(
                raw_evidence_index=raw_evidence_index,
                doc_name=doc_name if isinstance(doc_name, str) else "",
                gold_page_num=gold_page_num if type(gold_page_num) is int else None,
                matched_page_num=None,
                evidence_text=evidence_text if isinstance(evidence_text, str) else "",
                page_text="",
                match_status=SKIP_MALFORMED_EVIDENCE,
            ),
            SKIP_MALFORMED_EVIDENCE,
        )

    canonical_doc_name = canonical_pdf_name(doc_name)
    if canonical_doc_name not in pages_by_document:
        return (
            _processed_evidence_row(
                raw_evidence_index=raw_evidence_index,
                doc_name=doc_name,
                gold_page_num=gold_page_num,
                matched_page_num=None,
                evidence_text=evidence_text,
                page_text="",
                match_status=SKIP_MISSING_EXTRACTED_DOCUMENT,
            ),
            SKIP_MISSING_EXTRACTED_DOCUMENT,
        )

    candidate_page_nums = _candidate_page_nums(gold_page_num)
    available_candidate_page_nums = [
        page_num
        for page_num in candidate_page_nums
        if (canonical_doc_name, page_num) in page_index
    ]
    if not available_candidate_page_nums:
        return (
            _processed_evidence_row(
                raw_evidence_index=raw_evidence_index,
                doc_name=doc_name,
                gold_page_num=gold_page_num,
                matched_page_num=None,
                evidence_text=evidence_text,
                page_text="",
                match_status=SKIP_MISSING_EXTRACTED_PAGE,
            ),
            SKIP_MISSING_EXTRACTED_PAGE,
        )

    first_unmatched_page_num = available_candidate_page_nums[0]
    first_unmatched_page_text = page_index[(canonical_doc_name, first_unmatched_page_num)]
    for page_num in available_candidate_page_nums:
        page_text = page_index[(canonical_doc_name, page_num)]
        if _alphanumeric_substring_match(evidence_text, page_text):
            return (
                _processed_evidence_row(
                    raw_evidence_index=raw_evidence_index,
                    doc_name=doc_name,
                    gold_page_num=gold_page_num,
                    matched_page_num=page_num,
                    evidence_text=evidence_text,
                    page_text=page_text,
                    match_status="exact_match",
                ),
                None,
            )

    return (
        _processed_evidence_row(
            raw_evidence_index=raw_evidence_index,
            doc_name=doc_name,
            gold_page_num=gold_page_num,
            matched_page_num=first_unmatched_page_num,
            evidence_text=evidence_text,
            page_text=first_unmatched_page_text,
            match_status=SKIP_EVIDENCE_TEXT_NOT_FOUND,
        ),
        SKIP_EVIDENCE_TEXT_NOT_FOUND,
    )


def _candidate_page_nums(gold_page_num: int) -> list[int]:
    candidate_page_nums = [gold_page_num, gold_page_num + 1]
    return list(dict.fromkeys(candidate_page_nums))


def _processed_evidence_row(
    *,
    raw_evidence_index: int,
    doc_name: str,
    gold_page_num: int | None,
    matched_page_num: int | None,
    evidence_text: str,
    page_text: str,
    match_status: str,
) -> dict[str, Any]:
    return {
        "raw_evidence_index": raw_evidence_index,
        "doc_name": doc_name,
        "gold_page_num": gold_page_num,
        "matched_page_num": matched_page_num,
        "evidence_text": evidence_text,
        "page_text": page_text,
        "evidence_quality": _evidence_quality(
            evidence_text=evidence_text,
            page_text=page_text,
            match_status=match_status,
            gold_page_num=gold_page_num,
            matched_page_num=matched_page_num,
        ),
    }


def _evidence_quality(
    *,
    evidence_text: str,
    page_text: str,
    match_status: str,
    gold_page_num: int | None,
    matched_page_num: int | None,
) -> dict[str, Any]:
    normalized_evidence = _normalize_alphanumeric(evidence_text)
    normalized_page = _normalize_alphanumeric(page_text)
    normalized_substring_match = bool(normalized_evidence) and (
        normalized_evidence in normalized_page
    )
    normalized_full_page_match = bool(normalized_evidence) and (
        normalized_evidence == normalized_page
    )
    warnings: list[str] = []
    if (
        gold_page_num is not None
        and matched_page_num is not None
        and gold_page_num != matched_page_num
    ):
        warnings.append(PAGE_NUMBER_MISMATCH_WARNING)

    return {
        "normalizer": TEXT_NORMALIZER_ID,
        "match_status": match_status,
        "normalized_substring_match": normalized_substring_match,
        "normalized_full_page_match": normalized_full_page_match,
        "evidence_char_count": len(evidence_text),
        "page_char_count": len(page_text),
        "line_coverage_ratio": _line_coverage_ratio(evidence_text, page_text),
        "warnings": warnings,
    }


def _line_coverage_ratio(evidence_text: str, page_text: str) -> float:
    normalized_page = _normalize_alphanumeric(page_text)
    normalized_lines = [
        _normalize_alphanumeric(line)
        for line in evidence_text.splitlines()
        if _normalize_alphanumeric(line)
    ]
    if not normalized_lines:
        return 0.0

    matched_count = sum(1 for line in normalized_lines if line in normalized_page)
    return round(matched_count / len(normalized_lines), 6)


def _processed_accepted_row(
    metadata: dict[str, str],
    evidence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question_id": metadata["question_id"],
        "company": metadata["company"],
        "doc_name": metadata["doc_name"],
        "question": metadata["question"],
        "gold_answer": metadata["gold_answer"],
        "evidence": evidence_rows,
    }


def _processed_rejected_row(
    *,
    metadata: dict[str, str],
    skip_reason: str,
    evidence_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row = _processed_accepted_row(metadata, evidence_rows)
    row["skip_reason"] = skip_reason
    return row


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_processed_example_row(row: Any, line_number: int) -> None:
    if not isinstance(row, dict):
        raise ProcessedFinanceBenchExampleLoadError(
            f"Invalid processed example row at line {line_number}: expected object"
        )

    for field in ("question_id", "company", "doc_name", "question", "gold_answer"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ProcessedFinanceBenchExampleLoadError(
                f"Invalid processed example row at line {line_number}: "
                f"missing or empty '{field}'"
            )

    evidence = row.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ProcessedFinanceBenchExampleLoadError(
            f"Invalid processed example row at line {line_number}: "
            "missing or empty 'evidence'"
        )


def _pdf_reader_class() -> Any:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentExtractionError(
            "pypdf is required for PDF extraction. Install the project dependencies "
            "with `python -m pip install -e .`."
        ) from exc
    return PdfReader


def _load_document_page_index(path: Path) -> dict[tuple[str, int], str]:
    if not path.is_file():
        raise DocumentPageLoadError(
            f"Extracted document pages file not found: {path}. "
            "Run `financebench-harness extract-documents` first."
        )

    page_index: dict[tuple[str, int], str] = {}
    try:
        with path.open(encoding="utf-8") as pages_file:
            for line_number, line in enumerate(pages_file, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except JSONDecodeError as exc:
                    raise DocumentPageLoadError(
                        f"Invalid extracted page JSONL at line {line_number}: {exc.msg}"
                    ) from exc

                if not isinstance(row, dict):
                    raise DocumentPageLoadError(
                        f"Invalid extracted page row at line {line_number}: expected object"
                    )

                doc_name = row.get("doc_name")
                page_num = row.get("page_num")
                text = row.get("text")
                if not isinstance(doc_name, str) or not doc_name.strip():
                    raise DocumentPageLoadError(
                        f"Invalid extracted page row at line {line_number}: "
                        "missing or empty 'doc_name'"
                    )
                if not isinstance(page_num, int) or isinstance(page_num, bool):
                    raise DocumentPageLoadError(
                        f"Invalid extracted page row at line {line_number}: "
                        "missing or invalid 'page_num'"
                    )
                if not isinstance(text, str):
                    raise DocumentPageLoadError(
                        f"Invalid extracted page row at line {line_number}: "
                        "missing or invalid 'text'"
                    )

                page_index[(doc_name, page_num)] = text
    except OSError as exc:
        raise DocumentPageLoadError(
            f"Could not read extracted document pages file: {path}"
        ) from exc

    return page_index


def _alphanumeric_substring_match(evidence_text: str, page_text: str) -> bool:
    evidence = _normalize_alphanumeric(evidence_text)
    page = _normalize_alphanumeric(page_text)
    return bool(evidence) and evidence in page


def _normalize_alphanumeric(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalnum())


def _text_excerpt(text: str, limit: int = 160) -> str:
    excerpt = " ".join(text.split())
    if len(excerpt) <= limit:
        return excerpt
    return f"{excerpt[: limit - 3]}..."


def _write_document_pages(path: Path, pages: list[DocumentPage]) -> None:
    with path.open("w", encoding="utf-8") as pages_file:
        for page in pages:
            pages_file.write(
                json.dumps(
                    {
                        "doc_name": page.doc_name,
                        "page_num": page.page_num,
                        "text": page.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_document_failures(
    path: Path,
    failures: list[DocumentExtractionFailure],
) -> None:
    with path.open("w", encoding="utf-8") as failures_file:
        for failure in failures:
            failures_file.write(
                json.dumps(
                    {
                        "doc_name": failure.doc_name,
                        "page_num": failure.page_num,
                        "error": failure.error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _issue_from_load_error(
    line: str,
    line_number: int,
    exc: FinanceBenchQuestionLoadError,
) -> DatasetValidationIssue:
    return DatasetValidationIssue(
        line_number=line_number,
        question_id=_question_id_from_line(line),
        field=_field_from_error_message(str(exc)),
        message=_message_from_error(str(exc), line_number),
    )


def _question_id_from_line(line: str) -> str | None:
    try:
        row = json.loads(line)
    except JSONDecodeError:
        return None

    if not isinstance(row, dict):
        return None

    question_id = row.get("financebench_id")
    if not isinstance(question_id, str) or not question_id.strip():
        return None
    return question_id


def _field_from_error_message(message: str) -> str:
    if "'" in message:
        return message.split("'", maxsplit=2)[1]
    if "JSONL" in message:
        return "json"
    return "row"


def _message_from_error(message: str, line_number: int) -> str:
    prefix = f"Invalid FinanceBench row at line {line_number}: "
    if message.startswith(prefix):
        return message.removeprefix(prefix)

    json_prefix = f"Invalid FinanceBench JSONL at line {line_number}: "
    if message.startswith(json_prefix):
        return message.removeprefix(json_prefix)

    return message


def _parse_financebench_example(line: str, line_number: int) -> FinanceBenchExample:
    try:
        row = json.loads(line)
    except JSONDecodeError as exc:
        raise FinanceBenchQuestionLoadError(
            f"Invalid FinanceBench JSONL at line {line_number}: {exc.msg}"
        ) from exc

    if not isinstance(row, dict):
        raise FinanceBenchQuestionLoadError(
            f"Invalid FinanceBench row at line {line_number}: expected object"
        )

    question_id = _required_string(row, "financebench_id", line_number)
    question = _required_string(row, "question", line_number)
    gold_answer = _required_string(row, "answer", line_number)
    company = _required_string(row, "company", line_number)
    doc_name = _required_string(row, "doc_name", line_number)
    evidence = _required_evidence(row, line_number)
    primary_evidence = evidence[0]

    return FinanceBenchExample(
        question_id=question_id,
        company=company,
        doc_name=doc_name,
        question=question,
        gold_answer=gold_answer,
        evidence_doc_name=primary_evidence.doc_name,
        evidence_page_num=primary_evidence.page_num,
        evidence_text=primary_evidence.text,
        evidence=evidence,
    )


def _required_string(row: dict[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FinanceBenchQuestionLoadError(
            f"Invalid FinanceBench row at line {line_number}: missing or empty '{field}'"
        )
    return value


def _required_evidence(
    row: dict[str, Any],
    line_number: int,
) -> tuple[FinanceBenchEvidence, ...]:
    raw_evidence = row.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise FinanceBenchQuestionLoadError(
            f"Invalid FinanceBench row at line {line_number}: missing or empty 'evidence'"
        )

    evidence_items = [
        _parse_evidence_item(item, line_number, index)
        for index, item in enumerate(raw_evidence, start=1)
    ]
    return tuple(evidence_items)


def _parse_evidence_item(
    item: Any,
    line_number: int,
    index: int,
) -> FinanceBenchEvidence:
    if not isinstance(item, dict):
        raise FinanceBenchQuestionLoadError(
            "Invalid FinanceBench row at line "
            f"{line_number}: evidence[{index}] must be an object"
        )

    doc_name = _required_evidence_string(item, "doc_name", line_number, index)
    text = _required_evidence_string(item, "evidence_text", line_number, index)
    page_num = item.get("evidence_page_num")
    if type(page_num) is not int:
        raise FinanceBenchQuestionLoadError(
            "Invalid FinanceBench row at line "
            f"{line_number}: missing or invalid 'evidence[{index}].evidence_page_num'"
        )

    return FinanceBenchEvidence(doc_name=doc_name, page_num=page_num, text=text)


def _required_evidence_string(
    item: dict[str, Any],
    field: str,
    line_number: int,
    index: int,
) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FinanceBenchQuestionLoadError(
            "Invalid FinanceBench row at line "
            f"{line_number}: missing or empty 'evidence[{index}].{field}'"
        )
    return value
