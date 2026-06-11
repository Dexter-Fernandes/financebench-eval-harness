from __future__ import annotations

from pathlib import Path

from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data_common import (
    alphanumeric_substring_match,
    canonical_pdf_name,
    load_document_page_index,
    resolve_dataset_config,
    text_excerpt,
)
from financebench_eval_harness.data_loading import load_financebench_examples
from financebench_eval_harness.data_types import EvidencePageCheck, EvidencePageValidationResult
from financebench_eval_harness.documents import build_document_registry


def validate_financebench_evidence_pages(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> EvidencePageValidationResult:
    """Validate evidence document and page references against extracted page text."""

    config = resolve_dataset_config(dataset_config_or_path)
    examples = load_financebench_examples(config)
    registry = build_document_registry(config)
    page_index = load_document_page_index(config.processed_dir / "pages.jsonl")

    checks: list[EvidencePageCheck] = []
    for example in examples:
        for evidence_index, evidence in enumerate(example.evidence, start=1):
            document_filename = canonical_pdf_name(evidence.doc_name)
            document_path = registry.get(document_filename)
            extracted_page_num = evidence.page_num + 1
            evidence_excerpt = text_excerpt(evidence.text)

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

            is_match = alphanumeric_substring_match(evidence.text, page_text)
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
                    page_excerpt=text_excerpt(page_text),
                )
            )

    return EvidencePageValidationResult(checks=tuple(checks))
