from pathlib import Path

import pytest

from financebench_eval_harness.evaluation import (
    DEFAULT_EVALUATION_CONFIG_PATH,
    EvaluationConfigError,
    EvaluationMode,
    RenderedPrompt,
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
    assert config.prompt_for(EvaluationMode.CLOSED_BOOK).version == "v1"
    assert config.prompt_for(EvaluationMode.ORACLE_CONTEXT).version == "v1"
    assert config.prompt_for(EvaluationMode.CLOSED_BOOK).template_path == Path(
        "prompts/baselines/closed_book_v1.txt"
    )
    assert config.prompt_for(EvaluationMode.ORACLE_CONTEXT).template_path == Path(
        "prompts/baselines/oracle_context_v1.txt"
    )


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
                "        version: v1",
                "        template_path: prompts/baselines/closed_book_v1.txt",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationConfigError) as exc_info:
        load_evaluation_config(config_path)

    assert "Evaluation config missing required mode(s): oracle_context" in str(
        exc_info.value
    )


def test_load_evaluation_config_reports_missing_prompt_file(tmp_path: Path) -> None:
    config_path = tmp_path / "baselines.yaml"
    missing_prompt_path = tmp_path / "missing.txt"
    config_path.write_text(
        "\n".join(
            [
                "evaluation:",
                "  modes:",
                "    closed_book:",
                "      prompt:",
                "        id: closed_book_v1",
                "        version: v1",
                f"        template_path: {missing_prompt_path}",
                "    oracle_context:",
                "      prompt:",
                "        id: oracle_context_v1",
                "        version: v1",
                f"        template_path: {missing_prompt_path}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationConfigError) as exc_info:
        load_evaluation_config(config_path)

    assert f"Evaluation prompt template file not found: {missing_prompt_path}" in str(
        exc_info.value
    )


def test_render_closed_book_prompt_returns_text_and_run_metadata() -> None:
    config = load_evaluation_config()

    rendered_prompt = render_prompt(
        config,
        EvaluationMode.CLOSED_BOOK,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million."],
    )

    assert isinstance(rendered_prompt, RenderedPrompt)
    assert "What was FY2022 revenue?" in rendered_prompt.text
    assert "Revenue was $10 million." not in rendered_prompt.text
    assert "If you are unsure, say you do not know." in rendered_prompt.text
    assert rendered_prompt.mode == EvaluationMode.CLOSED_BOOK
    assert rendered_prompt.prompt_id == "closed_book_v1"
    assert rendered_prompt.prompt_version == "v1"
    assert rendered_prompt.template_path == Path("prompts/baselines/closed_book_v1.txt")
    assert rendered_prompt.run_metadata == {
        "evaluation_mode": "closed_book",
        "prompt_id": "closed_book_v1",
        "prompt_version": "v1",
        "prompt_template_path": "prompts/baselines/closed_book_v1.txt",
    }


def test_render_oracle_context_prompt_returns_joined_evidence_and_run_metadata() -> None:
    config = load_evaluation_config()

    rendered_prompt = render_prompt(
        config,
        EvaluationMode.ORACLE_CONTEXT,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million.", "Net income was $2 million."],
    )

    assert "What was FY2022 revenue?" in rendered_prompt.text
    assert "[Evidence 1]" in rendered_prompt.text
    assert "Revenue was $10 million." in rendered_prompt.text
    assert "[Evidence 2]" in rendered_prompt.text
    assert "Net income was $2 million." in rendered_prompt.text
    assert "Answer using only the provided evidence." in rendered_prompt.text
    assert rendered_prompt.mode == EvaluationMode.ORACLE_CONTEXT
    assert rendered_prompt.prompt_id == "oracle_context_v1"
    assert rendered_prompt.prompt_version == "v1"
    assert rendered_prompt.template_path == Path(
        "prompts/baselines/oracle_context_v1.txt"
    )
    assert rendered_prompt.run_metadata == {
        "evaluation_mode": "oracle_context",
        "prompt_id": "oracle_context_v1",
        "prompt_version": "v1",
        "prompt_template_path": "prompts/baselines/oracle_context_v1.txt",
    }


def test_render_prompt_for_processed_example_uses_evidence_source_order() -> None:
    config = load_evaluation_config()
    processed_example = {
        "question": "What was FY2022 revenue?",
        "evidence": [
            {"evidence_text": "First source text."},
            {"evidence_text": "Second source text."},
        ],
    }

    rendered_prompt = render_prompt_for_processed_example(
        config,
        EvaluationMode.ORACLE_CONTEXT,
        processed_example,
    )

    assert rendered_prompt.text.index("First source text.") < rendered_prompt.text.index(
        "Second source text."
    )
