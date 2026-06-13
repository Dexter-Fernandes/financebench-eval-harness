"""M5.6–M5.7 — RAG answer runner (TDD)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.llm import (
    LLMGenerationConfig,
    LLMGenerationResult,
    LLMProviderError,
    MockLLMClient,
)
from financebench_eval_harness.rag_run import RAGRunResult, run_rag_from_config
from financebench_eval_harness.rag_run_config import RAGEvalSettings, RAGRunConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_EVAL_CONFIG_PATH = Path("configs/evaluation/rag_modes.yaml")

_EXAMPLES = [
    {
        "question_id": "q001",
        "question": "What was the revenue?",
        "gold_answer": "$10 million",
        "evidence": [{"evidence_text": "Revenue was $10 million."}],
    },
    {
        "question_id": "q002",
        "question": "What were the operating expenses?",
        "gold_answer": "$1.8 billion",
        "evidence": [{"evidence_text": "Operating expenses rose to $1.8 billion."}],
    },
]

_RETRIEVAL = [
    {
        "question_id": "q001",
        "retrieved": [
            {
                "rank": 1,
                "chunk_id": "doc_p1_c01",
                "doc_name": "doc.pdf",
                "page_num": 1,
                "text": "Revenue was $10 million.",
                "score": 0.9,
            },
            {
                "rank": 2,
                "chunk_id": "doc_p2_c01",
                "doc_name": "doc.pdf",
                "page_num": 2,
                "text": "Other info.",
                "score": 0.7,
            },
        ],
    },
    {
        "question_id": "q002",
        "retrieved": [
            {
                "rank": 1,
                "chunk_id": "doc_p3_c01",
                "doc_name": "doc.pdf",
                "page_num": 3,
                "text": "Operating expenses rose to $1.8 billion.",
                "score": 0.85,
            },
        ],
    },
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )


def _make_config(
    tmp_path: Path,
    *,
    top_k: int = 3,
    max_context_chars: int | None = None,
    mode: EvaluationMode = EvaluationMode.RAG_DENSE,
    retrieval_run_id: str | None = "test-retrieval-run",
) -> RAGRunConfig:
    examples_path = tmp_path / "examples.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    _write_jsonl(examples_path, _EXAMPLES)
    _write_jsonl(retrieval_path, _RETRIEVAL)
    return RAGRunConfig(
        settings=RAGEvalSettings(
            examples_path=examples_path,
            retrieval_results_path=retrieval_path,
            output_dir=tmp_path / "runs",
            mode=mode,
            top_k=top_k,
            retrieval_run_id=retrieval_run_id,
            max_context_chars=max_context_chars,
            eval_config_path=_EVAL_CONFIG_PATH,
        ),
        model=LLMGenerationConfig(
            provider="mock",
            model_name="mock-llm",
            temperature=0.0,
            max_tokens=512,
            timeout_seconds=30.0,
        ),
    )


def _mock_client(config: RAGRunConfig, responses: list[str] | None = None) -> MockLLMClient:
    return MockLLMClient(config.model, responses=responses or ["mock answer"] * 20)


# ---------------------------------------------------------------------------
# Cycle 1 — run_rag_from_config() returns RAGRunResult; predictions.jsonl exists
# ---------------------------------------------------------------------------


def test_run_rag_returns_rag_run_result(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    assert isinstance(result, RAGRunResult)
    assert result.predictions_path.exists()


# ---------------------------------------------------------------------------
# Cycle 2 — one prediction row per example
# ---------------------------------------------------------------------------


def test_predictions_has_one_row_per_example(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Cycle 3 — prediction row has standard fields
# ---------------------------------------------------------------------------


def test_prediction_row_has_standard_fields(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(
        config, _mock_client(config, ["$10 million"] * 20), run_id="test-run"
    )
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = rows[0]
    assert row["question_id"] == "q001"
    assert row["question"] == "What was the revenue?"
    assert row["gold_answer"] == "$10 million"
    assert row["prediction"] == "$10 million"
    assert row["mode"] == "rag_dense"
    assert row["model"] == "mock-llm"
    assert row["model_provider"] == "mock"
    assert row["model_name"] == "mock-llm"
    assert "prompt_id" in row
    assert "prompt_version" in row
    assert "latency_ms" in row
    assert row["status"] == "success"
    assert row["error"] is None


# ---------------------------------------------------------------------------
# Cycle 4 — prediction row has RAG-specific fields
# ---------------------------------------------------------------------------


def test_prediction_row_has_rag_fields(tmp_path: Path) -> None:
    config = _make_config(tmp_path, top_k=3, retrieval_run_id="run_test")
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = rows[0]
    assert row["top_k"] == 3
    assert isinstance(row["context_chunk_ids"], list)
    assert "doc_p1_c01" in row["context_chunk_ids"]
    assert row["retrieval_run_id"] == "run_test"


# ---------------------------------------------------------------------------
# Cycle 5 — config.yaml snapshot written to output_dir
# ---------------------------------------------------------------------------


def test_config_yaml_written_to_output_dir(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    assert result.config_path.exists()
    loaded = yaml.safe_load(result.config_path.read_text(encoding="utf-8"))
    assert "rag_eval" in loaded
    assert "model" in loaded


# ---------------------------------------------------------------------------
# Cycle 6 — scores.jsonl has one row per example
# ---------------------------------------------------------------------------


def test_scores_jsonl_has_one_row_per_example(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    rows = [
        json.loads(line)
        for line in result.scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert "question_id" in rows[0]
    assert "scores" in rows[0]


# ---------------------------------------------------------------------------
# Cycle 7 — run_metadata.json has correct counts and RAG fields
# ---------------------------------------------------------------------------


def test_run_metadata_has_correct_fields(tmp_path: Path) -> None:
    config = _make_config(tmp_path, top_k=5)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["attempted_count"] == 2
    assert metadata["success_count"] == 2
    assert metadata["error_count"] == 0
    assert "score_summary" in metadata
    assert metadata["top_k"] == 5
    assert metadata["run_id"] == "test-run"


# ---------------------------------------------------------------------------
# Cycle 8 — LLMProviderError → status=error, run continues
# ---------------------------------------------------------------------------


class _FailingLLMClient:
    def __init__(self, config: LLMGenerationConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> LLMGenerationResult:
        raise LLMProviderError("Connection refused")


def test_llm_failure_status_error_run_continues(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(config, _FailingLLMClient(config.model), run_id="test-run")
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(r["status"] == "error" for r in rows)
    assert result.error_count == 2
    assert result.success_count == 0


# ---------------------------------------------------------------------------
# Cycle 9 — limit slices examples
# ---------------------------------------------------------------------------


def test_limit_slices_examples(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run", limit=1)
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert result.attempted_count == 1


# ---------------------------------------------------------------------------
# Cycle 10 — judge absent → judge=None in score row, judge_failures.jsonl not created
# ---------------------------------------------------------------------------


def test_judge_absent_score_row_has_null_judge(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    assert config.judge is None
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    rows = [
        json.loads(line)
        for line in result.scores_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["judge"] is None
    assert not result.judge_failures_path.exists()


# ---------------------------------------------------------------------------
# Cycle 11 — top_k slices retrieved_context to at most top_k chunks
# ---------------------------------------------------------------------------


def test_top_k_slices_retrieved_context(tmp_path: Path) -> None:
    # q001 has 2 chunks; top_k=1 → only the first chunk used
    config = _make_config(tmp_path, top_k=1)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row_q001 = next(r for r in rows if r["question_id"] == "q001")
    assert row_q001["context_chunk_ids"] == ["doc_p1_c01"]


# ---------------------------------------------------------------------------
# Cycle 12 — output files use rag_ prefix (M5.7)
# ---------------------------------------------------------------------------


def test_output_files_use_rag_prefix(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    assert result.predictions_path.name == "rag_predictions.jsonl"
    assert result.run_metadata_path.name == "rag_run_metadata.json"


# ---------------------------------------------------------------------------
# Cycle 13 — run_metadata includes retrieval_run_id (M5.7)
# ---------------------------------------------------------------------------


def test_run_metadata_includes_retrieval_run_id(tmp_path: Path) -> None:
    config = _make_config(tmp_path, retrieval_run_id="run_abc")
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    metadata = json.loads(result.run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["retrieval_run_id"] == "run_abc"


# ---------------------------------------------------------------------------
# Cycle 14 — retrieval_run_id=None propagates to prediction row (M5.7)
# ---------------------------------------------------------------------------


def test_prediction_row_retrieval_run_id_none_when_unset(tmp_path: Path) -> None:
    config = _make_config(tmp_path, retrieval_run_id=None)
    result = run_rag_from_config(config, _mock_client(config), run_id="test-run")
    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["retrieval_run_id"] is None
