"""M5.1 — RAG evaluation mode definitions (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest

from financebench_eval_harness.evaluation import (
    EvaluationMode,
    load_evaluation_config,
    render_prompt,
)

# ---------------------------------------------------------------------------
# Cycle 1 — Enum values
# ---------------------------------------------------------------------------


def test_rag_dense_mode_round_trips() -> None:
    assert EvaluationMode("rag_dense") is EvaluationMode.RAG_DENSE
    assert EvaluationMode.RAG_DENSE.value == "rag_dense"


def test_rag_oracle_mode_round_trips() -> None:
    assert EvaluationMode("rag_oracle") is EvaluationMode.RAG_ORACLE
    assert EvaluationMode.RAG_ORACLE.value == "rag_oracle"


def test_rag_no_context_mode_round_trips() -> None:
    assert EvaluationMode("rag_no_context") is EvaluationMode.RAG_NO_CONTEXT
    assert EvaluationMode.RAG_NO_CONTEXT.value == "rag_no_context"


def test_future_rag_hybrid_and_reranked_enum_values_exist() -> None:
    assert EvaluationMode.RAG_HYBRID.value == "rag_hybrid"
    assert EvaluationMode.RAG_RERANKED.value == "rag_reranked"


# ---------------------------------------------------------------------------
# Cycle 2 — Config loading
# ---------------------------------------------------------------------------

RAG_CONFIG_PATH = Path("configs/evaluation/rag_modes.yaml")


def test_load_rag_modes_config_returns_three_modes() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    assert set(config.modes) == {
        EvaluationMode.RAG_DENSE,
        EvaluationMode.RAG_ORACLE,
        EvaluationMode.RAG_NO_CONTEXT,
    }


def test_load_rag_modes_config_has_correct_prompt_ids() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    assert config.prompt_for(EvaluationMode.RAG_DENSE).id == "rag_dense_v2"
    assert config.prompt_for(EvaluationMode.RAG_ORACLE).id == "rag_oracle_v1"
    assert config.prompt_for(EvaluationMode.RAG_NO_CONTEXT).id == "rag_no_context_v1"


# ---------------------------------------------------------------------------
# Cycle 3 — context_source field
# ---------------------------------------------------------------------------


def test_rag_dense_context_source_is_retrieved_chunks() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_DENSE).context_source == "retrieved_chunks"


def test_rag_oracle_context_source_is_gold_evidence() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_ORACLE).context_source == "gold_evidence"


def test_rag_no_context_context_source_is_none() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_NO_CONTEXT).context_source == "none"


# ---------------------------------------------------------------------------
# Cycle 4-6 — render_prompt dispatches on context_source
# ---------------------------------------------------------------------------


def test_render_rag_dense_injects_retrieved_chunks_into_prompt() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_DENSE,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million.", "Gross profit was $4 million."],
    )

    assert "What was FY2022 revenue?" in rendered.text
    assert "[Evidence 1]" in rendered.text
    assert "Revenue was $10 million." in rendered.text
    assert "[Evidence 2]" in rendered.text
    assert "Gross profit was $4 million." in rendered.text
    assert rendered.mode is EvaluationMode.RAG_DENSE


def test_render_rag_no_context_omits_evidence_even_when_supplied() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_NO_CONTEXT,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million."],
    )

    assert "What was FY2022 revenue?" in rendered.text
    assert "Revenue was $10 million." not in rendered.text
    assert rendered.mode is EvaluationMode.RAG_NO_CONTEXT


def test_render_rag_oracle_injects_gold_evidence_text() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_ORACLE,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million."],
    )

    assert "What was FY2022 revenue?" in rendered.text
    assert "Revenue was $10 million." in rendered.text
    assert rendered.mode is EvaluationMode.RAG_ORACLE


def test_render_rag_dense_raises_when_no_evidence_supplied() -> None:
    from financebench_eval_harness.evaluation import EvaluationConfigError

    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    with pytest.raises(EvaluationConfigError):
        render_prompt(
            config,
            EvaluationMode.RAG_DENSE,
            question="What was FY2022 revenue?",
            evidence_texts=None,
        )


# ---------------------------------------------------------------------------
# Cycle 7 — baselines still work after context_source is added to baselines.yaml
# ---------------------------------------------------------------------------


def test_baseline_closed_book_context_source_is_none() -> None:
    from financebench_eval_harness.evaluation import DEFAULT_EVALUATION_CONFIG_PATH

    config = load_evaluation_config(DEFAULT_EVALUATION_CONFIG_PATH)
    assert config.prompt_for(EvaluationMode.CLOSED_BOOK).context_source == "none"


def test_baseline_oracle_context_context_source_is_gold_evidence() -> None:
    from financebench_eval_harness.evaluation import DEFAULT_EVALUATION_CONFIG_PATH

    config = load_evaluation_config(DEFAULT_EVALUATION_CONFIG_PATH)
    assert config.prompt_for(EvaluationMode.ORACLE_CONTEXT).context_source == "gold_evidence"
