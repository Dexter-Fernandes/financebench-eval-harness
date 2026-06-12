from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.inspection import (
    InspectedEvidence,
    InspectedQuestion,
    InspectionError,
    format_inspection,
    load_inspection,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_RETRIEVED = [
    {"rank": 1, "chunk_id": "DOC_p001_c000", "doc_name": "DOC.pdf",
     "page_num": 1, "score": 0.82, "text": "Revenue was $4.2 billion for the fiscal year."},
    {"rank": 2, "chunk_id": "DOC_p002_c000", "doc_name": "DOC.pdf",
     "page_num": 2, "score": 0.71, "text": "Operating income decreased 5% year-over-year."},
]

SAMPLE_RESULTS = [
    {"question_id": "q001", "query": "What was revenue?", "retrieved": SAMPLE_RETRIEVED},
    {"question_id": "q002", "query": "What was net income?", "retrieved": []},
]

SAMPLE_EXAMPLES = [
    {
        "question_id": "q001",
        "question": "What was total revenue?",
        "evidence": [
            {
                "doc_name": "DOC.pdf",
                "matched_page_num": 1,
                "evidence_text": "Revenue was $4.2 billion for the fiscal year.",
            }
        ],
    },
    {
        "question_id": "q002",
        "question": "What was net income?",
        "evidence": [
            {
                "doc_name": "DOC.pdf",
                "matched_page_num": 3,
                "evidence_text": "Net income was $540 million.",
            }
        ],
    },
]


def _write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _write_examples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_run_dir(
    tmp_path: Path,
    *,
    results: list[dict] | None = None,
    metadata: dict | None = None,
) -> Path:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    _write_results(run_dir / "retrieval_results.jsonl", results or SAMPLE_RESULTS)
    if metadata is not None:
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    return run_dir


# ---------------------------------------------------------------------------
# InspectedEvidence
# ---------------------------------------------------------------------------


class TestInspectedEvidence:
    def test_stores_fields(self) -> None:
        ev = InspectedEvidence(doc_name="DOC.pdf", page_num=42, text="Revenue was high.")
        assert ev.doc_name == "DOC.pdf"
        assert ev.page_num == 42
        assert ev.text == "Revenue was high."

    def test_page_num_can_be_none(self) -> None:
        ev = InspectedEvidence(doc_name="DOC.pdf", page_num=None, text="")
        assert ev.page_num is None

    def test_is_immutable(self) -> None:
        ev = InspectedEvidence(doc_name="DOC.pdf", page_num=1, text="t")
        with pytest.raises(Exception):
            ev.doc_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InspectedQuestion
# ---------------------------------------------------------------------------


class TestInspectedQuestion:
    def test_stores_all_fields(self) -> None:
        iq = InspectedQuestion(
            question_id="q001",
            question_text="What was revenue?",
            gold_evidence=[],
            query="What was revenue?",
            retrieved=SAMPLE_RETRIEVED,
        )
        assert iq.question_id == "q001"
        assert iq.question_text == "What was revenue?"
        assert iq.gold_evidence == []
        assert iq.query == "What was revenue?"
        assert iq.retrieved == SAMPLE_RETRIEVED

    def test_is_immutable(self) -> None:
        iq = InspectedQuestion(
            question_id="q001", question_text="", gold_evidence=[],
            query="", retrieved=[],
        )
        with pytest.raises(Exception):
            iq.question_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# load_inspection — basic retrieval parsing
# ---------------------------------------------------------------------------


class TestLoadInspectionBasic:
    def test_returns_inspected_question(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        result = load_inspection("q001", run_dir)
        assert isinstance(result, InspectedQuestion)

    def test_question_id_matches(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        result = load_inspection("q001", run_dir)
        assert result.question_id == "q001"

    def test_query_comes_from_results(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        result = load_inspection("q001", run_dir)
        assert result.query == "What was revenue?"

    def test_retrieved_chunks_are_loaded(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        result = load_inspection("q001", run_dir)
        assert len(result.retrieved) == 2
        assert result.retrieved[0]["rank"] == 1
        assert result.retrieved[0]["score"] == 0.82

    def test_question_text_empty_without_examples(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        result = load_inspection("q001", run_dir)
        assert result.question_text == ""

    def test_gold_evidence_empty_without_examples(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        result = load_inspection("q001", run_dir)
        assert result.gold_evidence == []


# ---------------------------------------------------------------------------
# load_inspection — error cases
# ---------------------------------------------------------------------------


class TestLoadInspectionErrors:
    def test_raises_when_question_id_not_found(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        with pytest.raises(InspectionError, match="q999"):
            load_inspection("q999", run_dir)

    def test_raises_when_results_file_missing(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(InspectionError):
            load_inspection("q001", run_dir)

    def test_raises_when_run_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(InspectionError):
            load_inspection("q001", tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# load_inspection — gold evidence via explicit examples_path
# ---------------------------------------------------------------------------


class TestLoadInspectionWithExamples:
    def test_populates_question_text(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        examples_path = tmp_path / "examples.jsonl"
        _write_examples(examples_path, SAMPLE_EXAMPLES)
        result = load_inspection("q001", run_dir, examples_path=examples_path)
        assert result.question_text == "What was total revenue?"

    def test_populates_gold_evidence_doc_name(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        examples_path = tmp_path / "examples.jsonl"
        _write_examples(examples_path, SAMPLE_EXAMPLES)
        result = load_inspection("q001", run_dir, examples_path=examples_path)
        assert len(result.gold_evidence) == 1
        assert result.gold_evidence[0].doc_name == "DOC.pdf"

    def test_populates_gold_evidence_page_num(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        examples_path = tmp_path / "examples.jsonl"
        _write_examples(examples_path, SAMPLE_EXAMPLES)
        result = load_inspection("q001", run_dir, examples_path=examples_path)
        assert result.gold_evidence[0].page_num == 1

    def test_populates_gold_evidence_text(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)
        examples_path = tmp_path / "examples.jsonl"
        _write_examples(examples_path, SAMPLE_EXAMPLES)
        result = load_inspection("q001", run_dir, examples_path=examples_path)
        assert "4.2 billion" in result.gold_evidence[0].text

    def test_question_text_empty_when_question_id_not_in_examples(
        self, tmp_path: Path
    ) -> None:
        run_dir = _make_run_dir(tmp_path)
        examples_path = tmp_path / "examples.jsonl"
        _write_examples(examples_path, [SAMPLE_EXAMPLES[1]])  # only q002
        result = load_inspection("q001", run_dir, examples_path=examples_path)
        assert result.question_text == ""
        assert result.gold_evidence == []


# ---------------------------------------------------------------------------
# load_inspection — auto-reads dataset_path from run_metadata.json
# ---------------------------------------------------------------------------


class TestLoadInspectionMetadataFallback:
    def test_reads_examples_from_run_metadata(self, tmp_path: Path) -> None:
        examples_path = tmp_path / "examples.jsonl"
        _write_examples(examples_path, SAMPLE_EXAMPLES)
        run_dir = _make_run_dir(
            tmp_path,
            metadata={"dataset_path": str(examples_path), "run_id": "r"},
        )
        result = load_inspection("q001", run_dir)
        assert result.question_text == "What was total revenue?"

    def test_no_metadata_file_is_ok(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path)   # no metadata kwarg
        result = load_inspection("q001", run_dir)
        assert result.question_text == ""


# ---------------------------------------------------------------------------
# format_inspection
# ---------------------------------------------------------------------------


def _make_inspection(**kwargs) -> InspectedQuestion:
    defaults = dict(
        question_id="q001",
        question_text="What was total revenue?",
        gold_evidence=[
            InspectedEvidence(doc_name="DOC.pdf", page_num=1,
                              text="Revenue was $4.2 billion."),
        ],
        query="What was revenue?",
        retrieved=SAMPLE_RETRIEVED,
    )
    return InspectedQuestion(**{**defaults, **kwargs})


class TestFormatInspection:
    def test_includes_question_id(self) -> None:
        out = format_inspection(_make_inspection())
        assert "q001" in out

    def test_includes_question_text(self) -> None:
        out = format_inspection(_make_inspection())
        assert "What was total revenue?" in out

    def test_includes_gold_doc_name(self) -> None:
        out = format_inspection(_make_inspection())
        assert "DOC.pdf" in out

    def test_includes_gold_page_num(self) -> None:
        out = format_inspection(_make_inspection())
        assert "1" in out

    def test_includes_retrieved_rank(self) -> None:
        out = format_inspection(_make_inspection())
        assert "#1" in out
        assert "#2" in out

    def test_includes_retrieved_score(self) -> None:
        out = format_inspection(_make_inspection())
        assert "0.82" in out

    def test_includes_retrieved_doc_name(self) -> None:
        out = format_inspection(_make_inspection())
        assert "DOC.pdf" in out

    def test_includes_retrieved_page_num(self) -> None:
        out = format_inspection(_make_inspection())
        assert "p.1" in out

    def test_truncates_chunk_text_to_preview_chars(self) -> None:
        long_text = "x" * 500
        retrieved = [{"rank": 1, "chunk_id": "c", "doc_name": "D.pdf",
                      "page_num": 1, "score": 0.5, "text": long_text}]
        out = format_inspection(_make_inspection(retrieved=retrieved), preview_chars=50)
        assert "x" * 51 not in out
        assert "x" * 50 in out

    def test_shows_no_gold_evidence_section_when_empty(self) -> None:
        out = format_inspection(_make_inspection(gold_evidence=[]))
        assert "Gold Evidence" not in out

    def test_returns_string(self) -> None:
        assert isinstance(format_inspection(_make_inspection()), str)
