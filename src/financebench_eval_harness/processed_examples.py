from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data_common import (
    alphanumeric_substring_match,
    canonical_pdf_name,
    load_document_page_index,
    normalize_alphanumeric,
    resolve_dataset_config,
    write_jsonl_records,
)
from financebench_eval_harness.data_loading import (
    processed_metadata_from_raw_row,
    read_raw_question_rows,
)
from financebench_eval_harness.data_types import (
    PAGE_NUMBER_MISMATCH_WARNING,
    PROCESSED_EXAMPLES_FILENAME,
    REJECTED_EXAMPLES_FILENAME,
    SKIP_EVIDENCE_TEXT_NOT_FOUND,
    SKIP_MALFORMED_EVIDENCE,
    SKIP_MISSING_EVIDENCE,
    SKIP_MISSING_EXTRACTED_DOCUMENT,
    SKIP_MISSING_EXTRACTED_PAGE,
    TEXT_NORMALIZER_ID,
    ProcessedExamplesBuildResult,
    ProcessedFinanceBenchExampleLoadError,
)


def build_processed_financebench_examples(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> ProcessedExamplesBuildResult:
    """Build canonical accepted and rejected FinanceBench example JSONL files."""

    config = resolve_dataset_config(dataset_config_or_path)
    page_index = load_document_page_index(config.processed_dir / "pages.jsonl")
    pages_by_document = _group_pages_by_document(page_index)

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    skip_reason_counts: dict[str, int] = {}

    for line_number, row in read_raw_question_rows(config.questions_path):
        metadata = processed_metadata_from_raw_row(row, line_number)
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
    write_jsonl_records(output_path, accepted_rows)
    write_jsonl_records(rejected_path, rejected_rows)

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

    config = resolve_dataset_config(dataset_config_or_path)
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
        if alphanumeric_substring_match(evidence_text, page_text):
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
    normalized_evidence = normalize_alphanumeric(evidence_text)
    normalized_page = normalize_alphanumeric(page_text)
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
    normalized_page = normalize_alphanumeric(page_text)
    normalized_lines = [
        normalize_alphanumeric(line)
        for line in evidence_text.splitlines()
        if normalize_alphanumeric(line)
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
