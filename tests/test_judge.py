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
