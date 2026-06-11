from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path

from financebench_eval_harness.config import DatasetConfig, load_dataset_config
from financebench_eval_harness.data_types import DocumentPageLoadError


def resolve_dataset_config(
    dataset_config_or_path: DatasetConfig | str | Path | None,
) -> DatasetConfig:
    if dataset_config_or_path is None:
        return load_dataset_config()
    if isinstance(dataset_config_or_path, DatasetConfig):
        return dataset_config_or_path
    return load_dataset_config(dataset_config_or_path)


def canonical_pdf_name(doc_name: str) -> str:
    """Return the canonical PDF filename for a FinanceBench document name."""

    if doc_name.endswith(".pdf"):
        return doc_name
    return f"{doc_name}.pdf"


def load_document_page_index(path: Path) -> dict[tuple[str, int], str]:
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


def alphanumeric_substring_match(evidence_text: str, page_text: str) -> bool:
    evidence = normalize_alphanumeric(evidence_text)
    page = normalize_alphanumeric(page_text)
    return bool(evidence) and evidence in page


def normalize_alphanumeric(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalnum())


def text_excerpt(text: str, limit: int = 160) -> str:
    excerpt = " ".join(text.split())
    if len(excerpt) <= limit:
        return excerpt
    return f"{excerpt[: limit - 3]}..."


def write_jsonl_records(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
