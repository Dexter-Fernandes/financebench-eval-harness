from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data_common import canonical_pdf_name, resolve_dataset_config
from financebench_eval_harness.data_loading import load_financebench_examples
from financebench_eval_harness.data_types import (
    DocumentExtractionError,
    DocumentExtractionFailure,
    DocumentExtractionResult,
    DocumentPage,
    DocumentRegistryValidationResult,
    FinanceBenchQuestionLoadError,
)


def build_document_registry(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> dict[str, Path]:
    """Map local document filenames to their paths."""

    config = resolve_dataset_config(dataset_config_or_path)
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

    config = resolve_dataset_config(dataset_config_or_path)
    examples = load_financebench_examples(config)
    registry = build_document_registry(config)
    required_filenames = {
        canonical_pdf_name(evidence.doc_name)
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

    config = resolve_dataset_config(dataset_config_or_path)
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
