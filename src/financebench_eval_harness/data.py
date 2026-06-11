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
    if doc_name.endswith(".pdf"):
        return doc_name
    return f"{doc_name}.pdf"


def _pdf_reader_class() -> Any:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentExtractionError(
            "pypdf is required for PDF extraction. Install the project dependencies "
            "with `python -m pip install -e .`."
        ) from exc
    return PdfReader


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
    evidence = _required_evidence(row, line_number)
    primary_evidence = evidence[0]

    return FinanceBenchExample(
        question_id=question_id,
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
