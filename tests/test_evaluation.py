from pathlib import Path

import pytest

from financebench_eval_harness.evaluation import (
    DEFAULT_EVALUATION_CONFIG_PATH,
    EvaluationConfigError,
    EvaluationMode,
    load_evaluation_config,
    render_prompt,
    render_prompt_for_processed_example,
)


def test_load_evaluation_config_reads_default_baseline_modes() -> None:
    config = load_evaluation_config()

    assert DEFAULT_EVALUATION_CONFIG_PATH == Path("configs/evaluation/baselines.yaml")
    assert set(config.modes) == {
        EvaluationMode.CLOSED_BOOK,
        EvaluationMode.ORACLE_CONTEXT,
    }
    assert config.prompt_for(EvaluationMode.CLOSED_BOOK).id == "closed_book_v1"
    assert config.prompt_for(EvaluationMode.ORACLE_CONTEXT).id == "oracle_context_v1"


def test_load_evaluation_config_reports_missing_required_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "baselines.yaml"
    config_path.write_text(
        "\n".join(
            [
                "evaluation:",
                "  modes:",
                "    closed_book:",
                "      prompt:",
                "        id: closed_book_v1",
                "        template: 'Question: {question}'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationConfigError) as exc_info:
        load_evaluation_config(config_path)

    assert "Evaluation config missing required mode(s): oracle_context" in str(
        exc_info.value
    )


def test_render_closed_book_prompt_includes_question_without_context() -> None:
    config = load_evaluation_config()

    prompt = render_prompt(
        config,
        EvaluationMode.CLOSED_BOOK,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million."],
    )

    assert "What was FY2022 revenue?" in prompt
    assert "Revenue was $10 million." not in prompt
    assert "Do not invent citations" in prompt


def test_render_oracle_context_prompt_includes_joined_gold_evidence() -> None:
    config = load_evaluation_config()

    prompt = render_prompt(
        config,
        EvaluationMode.ORACLE_CONTEXT,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million.", "Net income was $2 million."],
    )

    assert "What was FY2022 revenue?" in prompt
    assert "[Evidence 1]" in prompt
    assert "Revenue was $10 million." in prompt
    assert "[Evidence 2]" in prompt
    assert "Net income was $2 million." in prompt
    assert "Use only the evidence context" in prompt


def test_render_prompt_for_processed_example_uses_evidence_source_order() -> None:
    config = load_evaluation_config()
    processed_example = {
        "question": "What was FY2022 revenue?",
        "evidence": [
            {"evidence_text": "First source text."},
            {"evidence_text": "Second source text."},
        ],
    }

    prompt = render_prompt_for_processed_example(
        config,
        EvaluationMode.ORACLE_CONTEXT,
        processed_example,
    )

    assert prompt.index("First source text.") < prompt.index("Second source text.")
