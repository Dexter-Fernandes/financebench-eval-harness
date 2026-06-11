from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data_common import canonical_pdf_name
from financebench_eval_harness.data_loading import (
    load_financebench_examples,
    validate_financebench_data_layout,
    validate_financebench_dataset,
)
from financebench_eval_harness.data_types import (
    DEFAULT_DATA_ROOT,
    DOCUMENTS_DIRNAME,
    PAGE_NUMBER_MISMATCH_WARNING,
    PROCESSED_EXAMPLES_FILENAME,
    QUESTIONS_FILENAME,
    REJECTED_EXAMPLES_FILENAME,
    SKIP_EVIDENCE_TEXT_NOT_FOUND,
    SKIP_MALFORMED_EVIDENCE,
    SKIP_MISSING_EVIDENCE,
    SKIP_MISSING_EXTRACTED_DOCUMENT,
    SKIP_MISSING_EXTRACTED_PAGE,
    TEXT_NORMALIZER_ID,
    DatasetValidationIssue,
    DatasetValidationResult,
    DocumentExtractionError,
    DocumentExtractionFailure,
    DocumentExtractionResult,
    DocumentPage,
    DocumentPageLoadError,
    DocumentRegistryValidationResult,
    EvidencePageCheck,
    EvidencePageValidationResult,
    FinanceBenchDataLayout,
    FinanceBenchEvidence,
    FinanceBenchExample,
    FinanceBenchQuestionLoadError,
    MissingFinanceBenchDataError,
    ProcessedExamplesBuildResult,
    ProcessedFinanceBenchExampleLoadError,
)
from financebench_eval_harness.documents import (
    _pdf_reader_class,
    build_document_registry,
    validate_financebench_document_registry,
)
from financebench_eval_harness.documents import (
    extract_document_pages as _extract_document_pages,
)
from financebench_eval_harness.documents import (
    extract_financebench_documents as _extract_financebench_documents,
)
from financebench_eval_harness.evidence import validate_financebench_evidence_pages
from financebench_eval_harness.processed_examples import (
    build_processed_financebench_examples,
    load_processed_financebench_examples,
)


def extract_document_pages(
    document_path: Path,
    pdf_reader_factory: Any | None = None,
) -> tuple[list[DocumentPage], list[DocumentExtractionFailure]]:
    return _extract_document_pages(
        document_path,
        pdf_reader_factory=pdf_reader_factory or _lazy_pdf_reader_factory,
    )


def extract_financebench_documents(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
    pdf_reader_factory: Any | None = None,
    on_document_start: Callable[[Path], None] | None = None,
) -> DocumentExtractionResult:
    return _extract_financebench_documents(
        dataset_config_or_path,
        pdf_reader_factory=pdf_reader_factory or _lazy_pdf_reader_factory,
        on_document_start=on_document_start,
    )


def _lazy_pdf_reader_factory(path: str) -> Any:
    return _pdf_reader_class()(path)


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DOCUMENTS_DIRNAME",
    "DatasetValidationIssue",
    "DatasetValidationResult",
    "DocumentExtractionError",
    "DocumentExtractionFailure",
    "DocumentExtractionResult",
    "DocumentPage",
    "DocumentPageLoadError",
    "DocumentRegistryValidationResult",
    "EvidencePageCheck",
    "EvidencePageValidationResult",
    "FinanceBenchDataLayout",
    "FinanceBenchEvidence",
    "FinanceBenchExample",
    "FinanceBenchQuestionLoadError",
    "MissingFinanceBenchDataError",
    "PAGE_NUMBER_MISMATCH_WARNING",
    "PROCESSED_EXAMPLES_FILENAME",
    "ProcessedExamplesBuildResult",
    "ProcessedFinanceBenchExampleLoadError",
    "QUESTIONS_FILENAME",
    "REJECTED_EXAMPLES_FILENAME",
    "SKIP_EVIDENCE_TEXT_NOT_FOUND",
    "SKIP_MALFORMED_EVIDENCE",
    "SKIP_MISSING_EVIDENCE",
    "SKIP_MISSING_EXTRACTED_DOCUMENT",
    "SKIP_MISSING_EXTRACTED_PAGE",
    "TEXT_NORMALIZER_ID",
    "build_document_registry",
    "build_processed_financebench_examples",
    "canonical_pdf_name",
    "extract_document_pages",
    "extract_financebench_documents",
    "load_financebench_examples",
    "load_processed_financebench_examples",
    "validate_financebench_data_layout",
    "validate_financebench_dataset",
    "validate_financebench_document_registry",
    "validate_financebench_evidence_pages",
]
