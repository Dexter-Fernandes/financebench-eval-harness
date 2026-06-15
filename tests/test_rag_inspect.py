from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.rag_inspect import (
    RagInspectError,
    RagInspectionResult,
    format_rag_inspection,
    load_rag_inspection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_rag_run_dir(tmp_path: Path, *, retrieval_run_id: str | None = "run_001") -> Path:
    rag_dir = tmp_path / "rag_run"
    rag_dir.mkdir(exist_ok=True)

    predictions = [
        {
            "question_id": "q1",
            "question": "What was Apple revenue in FY2022?",
            "gold_answer": "$394,328 million",
            "prediction": "Apple revenue was $394.3 billion.",
            "retrieved_chunk_ids": ["chunk_abc", "chunk_def"],
            "model_provider": "ollama",
            "model_name": "llama3.2:3b",
        }
    ]
    (rag_dir / "rag_predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8"
    )

    scores = [
        {
            "question_id": "q1",
            "scores": {
                "exact_match": False,
                "numeric_match": True,
                "unit_match": True,
                "normalized_string_match": True,
                "gold_numeric_values": [394328.0],
            },
            "judge": {
                "verdict": "correct",
                "numeric_error": False,
                "unsupported_claims": False,
                "reason": "Numeric value matches within tolerance.",
            },
            "status": "success",
        }
    ]
    (rag_dir / "scores.jsonl").write_text(
        "\n".join(json.dumps(s) for s in scores) + "\n", encoding="utf-8"
    )

    metadata = {
        "model_provider": "ollama",
        "model_name": "llama3.2:3b",
        "top_k": 5,
        "retrieval_run_id": retrieval_run_id,
    }
    (rag_dir / "rag_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    return rag_dir


def _make_retrieval_run_dir(tmp_path: Path) -> Path:
    ret_dir = tmp_path / "run_001"
    ret_dir.mkdir(exist_ok=True)

    results = [
        {
            "question_id": "q1",
            "query": "Apple FY2022 revenue",
            "retrieved": [
                {
                    "rank": 1,
                    "chunk_id": "chunk_abc",
                    "doc_name": "AAPL_FY2022_10K.pdf",
                    "page_num": 33,
                    "score": 0.923,
                    "text": "Apple reported net sales of $394,328 million for fiscal year 2022.",
                },
                {
                    "rank": 2,
                    "chunk_id": "chunk_def",
                    "doc_name": "AAPL_FY2022_10K.pdf",
                    "page_num": 12,
                    "score": 0.847,
                    "text": "Total net sales grew 8% year-over-year.",
                },
            ],
        }
    ]
    (ret_dir / "retrieval_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8"
    )

    scores = [
        {
            "question_id": "q1",
            "gold_doc_name": "AAPL_FY2022_10K.pdf",
            "gold_page_num": 33,
            "page_hit@5": True,
            "doc_hit@5": True,
            "best_evidence_overlap": 0.82,
            "k": 5,
        }
    ]
    (ret_dir / "retrieval_scores.jsonl").write_text(
        "\n".join(json.dumps(s) for s in scores) + "\n", encoding="utf-8"
    )

    return ret_dir


def _make_joined_dir(tmp_path: Path) -> Path:
    joined_dir = tmp_path / "analysis"
    joined_dir.mkdir(exist_ok=True)

    rows = [
        {
            "question_id": "q1",
            "retrieval": {"page_hit@5": True, "best_evidence_overlap": 0.82},
            "answer": {"numeric_match": True},
            "judge_verdict": "correct",
            "failure_labels": [],
            "category": "retrieval_hit_answer_correct",
        }
    ]
    (joined_dir / "joined_metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return joined_dir


# ---------------------------------------------------------------------------
# Cycle 1 — load returns question and prediction
# ---------------------------------------------------------------------------


def test_load_returns_question_and_prediction(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)

    result = load_rag_inspection(rag_dir, "q1")

    assert result.question_id == "q1"
    assert result.question == "What was Apple revenue in FY2022?"
    assert result.gold_answer == "$394,328 million"
    assert result.prediction == "Apple revenue was $394.3 billion."
    assert result.retrieved_chunk_ids == ["chunk_abc", "chunk_def"]


# ---------------------------------------------------------------------------
# Cycle 2 — load merges answer scores from scores.jsonl
# ---------------------------------------------------------------------------


def test_load_merges_answer_scores(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)

    result = load_rag_inspection(rag_dir, "q1")

    assert result.numeric_match is True
    assert result.exact_match is False
    assert result.judge_verdict == "correct"
    assert result.judge_numeric_error is False
    assert result.judge_unsupported_claims is False


# ---------------------------------------------------------------------------
# Cycle 3 — load is graceful without retrieval run
# ---------------------------------------------------------------------------


def test_load_graceful_without_retrieval_run(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)

    result = load_rag_inspection(rag_dir, "q1", retrieval_run_dir=None)

    assert result.retrieved_chunks == []
    assert result.gold_doc_name is None
    assert result.gold_page_num is None


# ---------------------------------------------------------------------------
# Cycle 4 — load includes retrieved chunks with text when retrieval run available
# ---------------------------------------------------------------------------


def test_load_includes_retrieved_chunks_with_text(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)
    ret_dir = _make_retrieval_run_dir(tmp_path)

    result = load_rag_inspection(rag_dir, "q1", retrieval_run_dir=ret_dir)

    assert len(result.retrieved_chunks) == 2
    first = result.retrieved_chunks[0]
    assert first["rank"] == 1
    assert first["doc_name"] == "AAPL_FY2022_10K.pdf"
    assert first["page_num"] == 33
    assert "394,328" in first["text"]
    assert result.gold_doc_name == "AAPL_FY2022_10K.pdf"
    assert result.gold_page_num == 33
    assert result.page_hit_at_k is True
    assert result.best_evidence_overlap == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# Cycle 4b — auto-resolves retrieval run from rag_run_metadata.json
# ---------------------------------------------------------------------------


def test_load_auto_resolves_retrieval_run(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id="run_001")
    _make_retrieval_run_dir(tmp_path)  # creates tmp_path/run_001

    result = load_rag_inspection(rag_dir, "q1")  # no explicit retrieval_run_dir

    assert len(result.retrieved_chunks) == 2


# ---------------------------------------------------------------------------
# Cycle 5 — load includes failure labels from joined_dir
# ---------------------------------------------------------------------------


def test_load_includes_failure_labels_from_joined_dir(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)
    # Override joined row with failure labels
    joined_dir = tmp_path / "analysis"
    joined_dir.mkdir(exist_ok=True)
    row = {
        "question_id": "q1",
        "retrieval": {"page_hit@5": False, "best_evidence_overlap": 0.1},
        "answer": {"numeric_match": False},
        "judge_verdict": "incorrect",
        "failure_labels": ["retrieval_miss", "numeric_error"],
        "category": "retrieval_miss_answer_wrong",
    }
    (joined_dir / "joined_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = load_rag_inspection(rag_dir, "q1", joined_dir=joined_dir)

    assert "retrieval_miss" in result.failure_labels
    assert "numeric_error" in result.failure_labels
    assert result.category == "retrieval_miss_answer_wrong"


# ---------------------------------------------------------------------------
# Cycle 6 — format shows all expected sections
# ---------------------------------------------------------------------------


def test_format_shows_all_sections(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)
    result = load_rag_inspection(rag_dir, "q1")
    output = format_rag_inspection(result)

    assert "QUESTION" in output
    assert "GOLD ANSWER" in output
    assert "GENERATED ANSWER" in output
    assert "ANSWER SCORES" in output
    assert "FAILURE LABELS" in output
    assert "q1" in output


# ---------------------------------------------------------------------------
# Cycle 7 — format marks [PAGE HIT] on matching chunk
# ---------------------------------------------------------------------------


def test_format_marks_page_hit_chunk(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)
    ret_dir = _make_retrieval_run_dir(tmp_path)

    result = load_rag_inspection(rag_dir, "q1", retrieval_run_dir=ret_dir)
    output = format_rag_inspection(result)

    assert "[PAGE HIT]" in output
    assert "RETRIEVED CHUNKS" in output


# ---------------------------------------------------------------------------
# Cycle 8 — CLI inspect-rag prints report
# ---------------------------------------------------------------------------


def test_cli_inspect_rag_prints_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from financebench_eval_harness.cli import main

    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)

    exit_code = main([
        "inspect-rag",
        "--run", str(rag_dir),
        "--question-id", "q1",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "QUESTION" in captured.out


# ---------------------------------------------------------------------------
# Error-path tests — fail loudly on bad inputs
# ---------------------------------------------------------------------------


def test_load_raises_on_missing_run_dir(tmp_path: Path) -> None:
    with pytest.raises(RagInspectError, match="not found"):
        load_rag_inspection(tmp_path / "nonexistent", "q1")


def test_load_raises_on_missing_predictions_file(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag_run"
    rag_dir.mkdir()
    # scores.jsonl present, but no rag_predictions.jsonl
    (rag_dir / "scores.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RagInspectError, match="rag_predictions.jsonl"):
        load_rag_inspection(rag_dir, "q1")


def test_load_raises_on_missing_scores_file(tmp_path: Path) -> None:
    rag_dir = tmp_path / "rag_run"
    rag_dir.mkdir()
    (rag_dir / "rag_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    # no scores.jsonl

    with pytest.raises(RagInspectError, match="scores.jsonl"):
        load_rag_inspection(rag_dir, "q1")


def test_load_raises_on_unknown_question_id(tmp_path: Path) -> None:
    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)

    with pytest.raises(RagInspectError, match="Question ID not found"):
        load_rag_inspection(rag_dir, "does_not_exist")


def test_cli_returns_1_on_missing_run_dir(tmp_path: Path) -> None:
    from financebench_eval_harness.cli import main

    exit_code = main([
        "inspect-rag",
        "--run", str(tmp_path / "nonexistent"),
        "--question-id", "q1",
    ])

    assert exit_code == 1


def test_cli_returns_1_on_unknown_question_id(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from financebench_eval_harness.cli import main

    rag_dir = _make_rag_run_dir(tmp_path, retrieval_run_id=None)

    exit_code = main([
        "inspect-rag",
        "--run", str(rag_dir),
        "--question-id", "not_a_real_id",
    ])

    assert exit_code == 1
    assert "not_a_real_id" in capsys.readouterr().err
