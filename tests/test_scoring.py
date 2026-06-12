from financebench_eval_harness.scoring import (
    extract_numeric_values,
    score_prediction,
    summarize_scores,
)


def test_score_prediction_reports_string_matching_signals() -> None:
    score = score_prediction(
        gold_answer="Revenue was $123.00.",
        prediction="The revenue was $123.00 for the period.",
    )

    assert score["exact_match"] is False
    assert score["normalized_string_match"] is False
    assert score["contains_gold_answer"] is True


def test_score_prediction_matches_normalized_strings() -> None:
    score = score_prediction(
        gold_answer="Gross profit: $456.00",
        prediction="gross profit $456.00",
    )

    assert score["exact_match"] is False
    assert score["normalized_string_match"] is True
    assert score["contains_gold_answer"] is True


def test_extract_numeric_values_handles_financial_number_formats() -> None:
    assert extract_numeric_values("Revenue was $1,234.50, up 12.5% in 2024.") == [
        1234.5,
        12.5,
        2024.0,
    ]


def test_extract_numeric_values_handles_common_financial_negative_formats() -> None:
    assert extract_numeric_values("Margin changes were (4.6)%, -$4.6, ($4.6), and -4.6%.") == [
        -4.6,
        -4.6,
        -4.6,
        -4.6,
    ]


def test_extract_numeric_values_handles_duplicate_minus_financial_output() -> None:
    assert extract_numeric_values("The table shows --4.6% for translation.") == [-4.6]


def test_extract_numeric_values_handles_mixed_table_summary_values() -> None:
    assert extract_numeric_values(
        "Organic sales were 1.2%, divestitures were (0.5)%, translation was --4.6%, and total sales change was (3.9)%."
    ) == [1.2, -0.5, -4.6, -3.9]


def test_score_prediction_matches_all_gold_numeric_values() -> None:
    score = score_prediction(
        gold_answer="$1,234.50 and 12.5%",
        prediction="The company reported 12.5% growth on revenue of $1,234.50.",
    )

    assert score["numeric_match"] is True
    assert score["gold_numeric_values"] == [1234.5, 12.5]
    assert score["prediction_numeric_values"] == [12.5, 1234.5]


def test_score_prediction_requires_gold_numbers_for_numeric_match() -> None:
    score = score_prediction(
        gold_answer="not disclosed",
        prediction="not disclosed",
    )

    assert score["numeric_match"] is False
    assert score["gold_numeric_values"] == []
    assert score["prediction_numeric_values"] == []


def test_summarize_scores_counts_and_rates() -> None:
    summary = summarize_scores(
        [
            {
                "exact_match": True,
                "normalized_string_match": True,
                "contains_gold_answer": True,
                "numeric_match": True,
                "gold_numeric_values": [1.0],
                "prediction_numeric_values": [1.0],
            },
            {
                "exact_match": False,
                "normalized_string_match": False,
                "contains_gold_answer": True,
                "numeric_match": False,
                "gold_numeric_values": [2.0],
                "prediction_numeric_values": [3.0],
            },
        ]
    )

    assert summary == {
        "example_count": 2,
        "exact_match_count": 1,
        "exact_match_rate": 0.5,
        "normalized_string_match_count": 1,
        "normalized_string_match_rate": 0.5,
        "contains_gold_answer_count": 2,
        "contains_gold_answer_rate": 1.0,
        "numeric_match_count": 1,
        "numeric_match_rate": 0.5,
    }
