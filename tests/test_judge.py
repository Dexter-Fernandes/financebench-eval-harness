from pathlib import Path

import pytest

from financebench_eval_harness.judge import (
    JudgeError,
    parse_judge_response,
    render_judge_prompt_for_processed_example,
)
from financebench_eval_harness.run_config import JudgePromptConfig


def test_render_judge_prompt_includes_prediction_inputs_and_evidence(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "judge_prompt.txt"
    template_path.write_text(
        "\n".join(
            [
                "Question:",
                "{question}",
                "Gold:",
                "{gold_answer}",
                "Prediction:",
                "{prediction}",
                "Evidence:",
                "{evidence_text}",
            ]
        ),
        encoding="utf-8",
    )
    prompt_config = JudgePromptConfig(
        id="answer_correctness_v1",
        version="v1",
        template_path=template_path,
    )

    rendered = render_judge_prompt_for_processed_example(
        prompt_config,
        {
            "question": "What was revenue?",
            "gold_answer": "$123",
            "evidence": [
                {"evidence_text": "Revenue was $123."},
                {"evidence_text": "The fiscal year ended in 2024."},
            ],
        },
        prediction="$123",
    )

    assert rendered.text.startswith("Question:\nWhat was revenue?")
    assert "Gold:\n$123" in rendered.text
    assert "Prediction:\n$123" in rendered.text
    assert "[Evidence 1]\nRevenue was $123." in rendered.text
    assert "[Evidence 2]\nThe fiscal year ended in 2024." in rendered.text
    assert rendered.prompt_id == "answer_correctness_v1"
    assert rendered.prompt_version == "v1"
    assert rendered.template_path == template_path


def test_parse_judge_response_accepts_valid_json_verdict() -> None:
    result = parse_judge_response(
        '{"verdict": "partially_correct", "reason": "Right metric, wrong year."}'
    )

    assert result == {
        "verdict": "partially_correct",
        "reason": "Right metric, wrong year.",
        "numeric_error": None,
        "unsupported_claims": None,
    }


@pytest.mark.parametrize(
    ("raw_response", "message"),
    [
        ("not json", "Judge response was not valid JSON"),
        ('{"verdict": "correct"}', "Judge response missing string field: reason"),
        (
            '{"verdict": "mostly_correct", "reason": "Close."}',
            "Unsupported judge verdict: mostly_correct",
        ),
    ],
)
def test_parse_judge_response_rejects_invalid_outputs(
    raw_response: str,
    message: str,
) -> None:
    with pytest.raises(JudgeError) as exc_info:
        parse_judge_response(raw_response)

    assert message in str(exc_info.value)


# ---------------------------------------------------------------------------
# M5.9 — retrieved_context in render, extended parse output
# ---------------------------------------------------------------------------


def test_render_v2_prompt_includes_retrieved_context(tmp_path: Path) -> None:
    template_path = tmp_path / "v2.txt"
    template_path.write_text(
        "Q: {question}\nGold: {gold_answer}\nPred: {prediction}\n"
        "Context: {retrieved_context}\nEvidence: {evidence_text}",
        encoding="utf-8",
    )
    prompt_config = JudgePromptConfig(id="v2", version="v2", template_path=template_path)

    rendered = render_judge_prompt_for_processed_example(
        prompt_config,
        {"question": "What was revenue?", "gold_answer": "$123", "evidence": []},
        prediction="$123",
        retrieved_context="Revenue was $123 per the income statement.",
    )

    assert "Revenue was $123 per the income statement." in rendered.text


def test_render_v1_prompt_ignores_retrieved_context_kwarg(tmp_path: Path) -> None:
    template_path = tmp_path / "v1.txt"
    template_path.write_text(
        "Q: {question}\nGold: {gold_answer}\nPred: {prediction}\nEv: {evidence_text}",
        encoding="utf-8",
    )
    prompt_config = JudgePromptConfig(id="v1", version="v1", template_path=template_path)

    rendered = render_judge_prompt_for_processed_example(
        prompt_config,
        {"question": "Q?", "gold_answer": "A", "evidence": []},
        prediction="A",
        retrieved_context="Some context that v1 template doesn't use.",
    )

    assert "Q?" in rendered.text  # rendered without error


def test_render_prompt_uses_no_retrieved_context_placeholder_when_none(
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "v2.txt"
    template_path.write_text("Context: {retrieved_context}", encoding="utf-8")
    prompt_config = JudgePromptConfig(id="v2", version="v2", template_path=template_path)

    rendered = render_judge_prompt_for_processed_example(
        prompt_config,
        {"question": "Q?", "gold_answer": "A", "evidence": []},
        prediction="A",
        retrieved_context=None,
    )

    assert "(no retrieved context)" in rendered.text


def test_parse_judge_response_accepts_numeric_error_and_unsupported_claims() -> None:
    result = parse_judge_response(
        '{"verdict": "incorrect", "reason": "Wrong value.", "numeric_error": true, "unsupported_claims": false}'
    )

    assert result["verdict"] == "incorrect"
    assert result["numeric_error"] is True
    assert result["unsupported_claims"] is False


def test_parse_judge_response_returns_none_for_missing_v2_fields() -> None:
    result = parse_judge_response('{"verdict": "correct", "reason": "Matches."}')

    assert result["numeric_error"] is None
    assert result["unsupported_claims"] is None


def test_parse_judge_response_rejects_non_bool_numeric_error() -> None:
    with pytest.raises(JudgeError, match="numeric_error"):
        parse_judge_response(
            '{"verdict": "correct", "reason": "OK.", "numeric_error": "yes"}'
        )
