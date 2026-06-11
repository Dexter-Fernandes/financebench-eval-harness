from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from financebench_eval_harness.config import DatasetConfig
from financebench_eval_harness.data_common import resolve_dataset_config
from financebench_eval_harness.data_types import (
    DEFAULT_DATA_ROOT,
    DatasetValidationIssue,
    DatasetValidationResult,
    FinanceBenchDataLayout,
    FinanceBenchEvidence,
    FinanceBenchExample,
    FinanceBenchQuestionLoadError,
    MissingFinanceBenchDataError,
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


def load_financebench_examples(
    dataset_config_or_path: DatasetConfig | str | Path | None = None,
) -> list[FinanceBenchExample]:
    """Load and normalize FinanceBench question records from configured JSONL."""

    config = resolve_dataset_config(dataset_config_or_path)
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

    config = resolve_dataset_config(dataset_config_or_path)
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


def read_raw_question_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
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


def processed_metadata_from_raw_row(
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
