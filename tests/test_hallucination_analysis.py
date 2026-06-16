"""Tests for M7: Hallucination and Grounding Analysis.

All tests use pure Python or MockEmbeddingClient — no real API calls.
Tests follow the TDD vertical-slice approach: each group corresponds to
one M7 module's public interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Group 0: Tracer bullet — taxonomy constants (M7.1, M7.2)
# ---------------------------------------------------------------------------


def test_grounding_labels_contains_all_seven():
    from financebench_eval_harness.grounding_types import GROUNDING_LABELS

    expected = {
        "grounded",
        "partially_grounded",
        "ungrounded",
        "contradicted",
        "insufficient_evidence",
        "over_refusal",
        "under_refusal",
    }
    assert set(GROUNDING_LABELS) == expected


def test_hallucination_failure_types_contains_all_eleven():
    from financebench_eval_harness.grounding_types import HALLUCINATION_FAILURE_TYPES

    expected = {
        "wrong_number",
        "wrong_unit",
        "wrong_period",
        "wrong_metric",
        "unsupported_claim",
        "contradicted_by_context",
        "bad_citation",
        "missing_citation",
        "retrieval_miss",
        "generation_error",
        "format_error",
    }
    assert set(HALLUCINATION_FAILURE_TYPES) == expected


def test_root_cause_labels_contains_all_six():
    from financebench_eval_harness.grounding_types import ROOT_CAUSE_LABELS

    expected = {
        "retrieval_failure",
        "generation_failure",
        "citation_failure",
        "hallucination_under_refusal",
        "over_refusal",
        "no_failure",
    }
    assert set(ROOT_CAUSE_LABELS) == expected


def test_failure_analysis_row_to_dict_round_trips():
    from financebench_eval_harness.grounding_types import FailureAnalysisRow

    row = FailureAnalysisRow(
        question_id="q1",
        evidence_hit=True,
        answer_verdict="correct",
        grounding_label="grounded",
        context_sufficiency="context_sufficient",
        citation_quality="supports_answer",
        failure_types=(),
        rule_flags=(),
        root_cause="no_failure",
        company="AAPL",
        question_type="numeric",
    )
    d = row.to_dict()
    assert d["question_id"] == "q1"
    assert d["root_cause"] == "no_failure"
    assert d["failure_types"] == []


# ---------------------------------------------------------------------------
# Group 1: claim_extraction.py — extract_claims(), extract_cited_chunk_ids()
# (M7.3, M7.5)
# ---------------------------------------------------------------------------


def test_extract_cited_chunk_ids_finds_inline_chunk_id_pattern():
    from financebench_eval_harness.claim_extraction import extract_cited_chunk_ids

    answer = "The FY2018 capex was $1,577 million. [chunk_id: doc1_p45_c02]"
    ids = extract_cited_chunk_ids(answer)
    assert "doc1_p45_c02" in ids


def test_extract_cited_chunk_ids_returns_empty_for_no_citation():
    from financebench_eval_harness.claim_extraction import extract_cited_chunk_ids

    answer = "Revenue was $1,577 million in fiscal year 2018."
    assert extract_cited_chunk_ids(answer) == []


def test_extract_cited_chunk_ids_finds_multiple_ids():
    from financebench_eval_harness.claim_extraction import extract_cited_chunk_ids

    answer = "See [chunk_id: doc1_p1_c01] and [chunk_id: doc1_p2_c03]."
    ids = extract_cited_chunk_ids(answer)
    assert "doc1_p1_c01" in ids
    assert "doc1_p2_c03" in ids
    assert len(ids) == 2


def test_extract_claims_numeric_claim_has_correct_type():
    from financebench_eval_harness.claim_extraction import extract_claims

    answer = "3M's FY2018 capex was $1,577 million."
    claims = extract_claims(answer)
    assert any(c["claim_type"] == "numeric" for c in claims)


def test_extract_claims_evidence_reference_has_correct_type():
    from financebench_eval_harness.claim_extraction import extract_claims

    answer = "The value comes from the cash flow statement. [chunk_id: doc1_p45_c02]"
    claims = extract_claims(answer)
    assert any(c["claim_type"] == "evidence_reference" for c in claims)


def test_extract_claims_assigns_sequential_claim_ids():
    from financebench_eval_harness.claim_extraction import extract_claims

    answer = "Revenue was $1,577 million. Operating income was $200 million."
    claims = extract_claims(answer)
    ids = [c["claim_id"] for c in claims]
    assert ids == sorted(ids)
    assert all(cid.startswith("claim_") for cid in ids)


def test_extract_claims_each_claim_has_required_keys():
    from financebench_eval_harness.claim_extraction import extract_claims

    answer = "Revenue was $500 million in FY2022."
    claims = extract_claims(answer)
    for claim in claims:
        assert "claim_id" in claim
        assert "claim_text" in claim
        assert "claim_type" in claim


# ---------------------------------------------------------------------------
# Group 2: scoring.py extension — extract_fiscal_periods() (M7.4)
# ---------------------------------------------------------------------------


def test_extract_fiscal_periods_finds_fy2022():
    from financebench_eval_harness.scoring import extract_fiscal_periods

    assert "FY2022" in extract_fiscal_periods("Revenue in FY2022 was high.")


def test_extract_fiscal_periods_finds_fy_with_space():
    from financebench_eval_harness.scoring import extract_fiscal_periods

    result = extract_fiscal_periods("FY 2021 results were strong.")
    assert any("2021" in p for p in result)


def test_extract_fiscal_periods_finds_quarter():
    from financebench_eval_harness.scoring import extract_fiscal_periods

    result = extract_fiscal_periods("Q3 2020 earnings exceeded expectations.")
    assert any("2020" in p for p in result)


def test_extract_fiscal_periods_returns_empty_for_no_period():
    from financebench_eval_harness.scoring import extract_fiscal_periods

    assert extract_fiscal_periods("Revenue was $500 million.") == []


def test_extract_fiscal_periods_does_not_return_duplicates():
    from financebench_eval_harness.scoring import extract_fiscal_periods

    result = extract_fiscal_periods("FY2022 revenue and FY2022 income.")
    assert result.count("FY2022") == 1


# ---------------------------------------------------------------------------
# Group 3: context_sufficiency.py (M7.7)
# ---------------------------------------------------------------------------


def test_context_sufficient_when_evidence_hit_true():
    from financebench_eval_harness.context_sufficiency import check_context_sufficiency

    result = check_context_sufficiency(
        gold_answer="$1,577 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "capex was $1,577 million"}],
        evidence_hit=True,
    )
    assert result == "context_sufficient"


def test_context_insufficient_when_evidence_hit_false_and_no_number_match():
    from financebench_eval_harness.context_sufficiency import check_context_sufficiency

    result = check_context_sufficiency(
        gold_answer="$1,577 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "operating income was $200 million"}],
        evidence_hit=False,
    )
    assert result == "context_insufficient"


def test_context_partially_sufficient_when_number_in_chunk():
    from financebench_eval_harness.context_sufficiency import check_context_sufficiency

    result = check_context_sufficiency(
        gold_answer="$1,577 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "purchases of property were 1577 million"}],
        evidence_hit=False,
    )
    assert result == "context_partially_sufficient"


def test_context_sufficient_when_evidence_hit_none_but_text_matches():
    from financebench_eval_harness.context_sufficiency import check_context_sufficiency

    result = check_context_sufficiency(
        gold_answer="$1,577 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "capex was $1,577 million in FY2018"}],
        evidence_hit=None,
    )
    assert result == "context_partially_sufficient"


# ---------------------------------------------------------------------------
# Group 4: citation_checker.py (M7.5, M7.6)
# ---------------------------------------------------------------------------


def _make_chunks(ids_and_texts: list[tuple[str, str]]) -> list[dict]:
    return [{"chunk_id": cid, "text": text} for cid, text in ids_and_texts]


def test_score_citations_no_citation_returns_missing():
    from financebench_eval_harness.citation_checker import score_citations

    chunks = _make_chunks([("c1", "revenue was $500 million")])
    result = score_citations(
        answer_text="Revenue was $500 million.",
        retrieved_chunks=chunks,
    )
    assert result["citation_quality"] == "citation_missing"
    assert result["missing_citation"] is True


def test_score_citations_invalid_chunk_id_returns_invalid():
    from financebench_eval_harness.citation_checker import score_citations

    chunks = _make_chunks([("c1", "revenue was $500 million")])
    result = score_citations(
        answer_text="Revenue was $500 million. [chunk_id: nonexistent_c99]",
        retrieved_chunks=chunks,
    )
    assert result["citation_quality"] == "citation_invalid"
    assert "nonexistent_c99" in result["invalid_chunk_ids"]


def test_score_citations_valid_citation_with_matching_number():
    from financebench_eval_harness.citation_checker import score_citations

    chunks = _make_chunks([("c1", "revenue was $500 million in 2022")])
    result = score_citations(
        answer_text="Revenue was $500 million. [chunk_id: c1]",
        retrieved_chunks=chunks,
    )
    assert result["citation_quality"] in ("supports_answer", "partially_supports_answer")
    assert result["missing_citation"] is False


def test_score_citations_valid_chunk_irrelevant_returns_does_not_support():
    from financebench_eval_harness.citation_checker import score_citations

    chunks = _make_chunks([("c1", "operating expenses were $300 million")])
    result = score_citations(
        answer_text="Revenue was $999 million. [chunk_id: c1]",
        retrieved_chunks=chunks,
    )
    assert result["citation_quality"] == "does_not_support_answer"


# ---------------------------------------------------------------------------
# Group 5: hallucination_checks.py (M7.9)
# ---------------------------------------------------------------------------


def test_rule_check_no_citation_flag():
    from financebench_eval_harness.hallucination_checks import run_rule_based_checks

    flags = run_rule_based_checks(
        prediction="Revenue was $500 million.",
        gold_answer="$500 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "revenue 500 million"}],
        cited_chunk_ids=[],
        all_chunk_ids=["c1"],
        gold_numeric_values=[500.0],
        prediction_numeric_values=[500.0],
    )
    assert "no_citation" in flags


def test_rule_check_predicted_number_not_in_context():
    from financebench_eval_harness.hallucination_checks import run_rule_based_checks

    flags = run_rule_based_checks(
        prediction="Revenue was $999 million.",
        gold_answer="$500 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "revenue was 500 million"}],
        cited_chunk_ids=[],
        all_chunk_ids=["c1"],
        gold_numeric_values=[500.0],
        prediction_numeric_values=[999.0],
    )
    assert "predicted_number_not_in_context" in flags


def test_rule_check_cited_chunk_not_retrieved():
    from financebench_eval_harness.hallucination_checks import run_rule_based_checks

    flags = run_rule_based_checks(
        prediction="Revenue was $500 million. [chunk_id: phantom_chunk]",
        gold_answer="$500 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "revenue 500 million"}],
        cited_chunk_ids=["phantom_chunk"],
        all_chunk_ids=["c1"],
        gold_numeric_values=[500.0],
        prediction_numeric_values=[500.0],
    )
    assert "cited_chunk_not_retrieved" in flags


def test_rule_check_refusal_with_evidence():
    from financebench_eval_harness.hallucination_checks import run_rule_based_checks

    flags = run_rule_based_checks(
        prediction="I cannot determine the answer from the provided context.",
        gold_answer="$500 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "revenue was 500 million in 2022"}],
        cited_chunk_ids=[],
        all_chunk_ids=["c1"],
        gold_numeric_values=[500.0],
        prediction_numeric_values=[],
    )
    assert "refusal_with_evidence" in flags


def test_rule_check_no_flags_for_grounded_answer():
    from financebench_eval_harness.hallucination_checks import run_rule_based_checks

    flags = run_rule_based_checks(
        prediction="Revenue was $500 million. [chunk_id: c1]",
        gold_answer="$500 million",
        retrieved_chunks=[{"chunk_id": "c1", "text": "revenue was 500 million in 2022"}],
        cited_chunk_ids=["c1"],
        all_chunk_ids=["c1"],
        gold_numeric_values=[500.0],
        prediction_numeric_values=[500.0],
    )
    assert "no_citation" not in flags
    assert "cited_chunk_not_retrieved" not in flags


# ---------------------------------------------------------------------------
# Group 6: analysis.py extension — classify_root_cause() (M7.11)
# ---------------------------------------------------------------------------


def _make_analysis_row(
    *,
    evidence_hit: bool | None = None,
    answer_verdict: str | None = None,
    citation_quality: str | None = None,
    context_sufficiency: str | None = None,
    grounding_label: str | None = None,
    k: int = 5,
) -> dict[str, Any]:
    return {
        f"evidence_hit@{k}": evidence_hit,
        "answer_verdict": answer_verdict,
        "citation_quality": citation_quality,
        "context_sufficiency": context_sufficiency,
        "grounding_label": grounding_label,
    }


def test_classify_root_cause_retrieval_failure():
    from financebench_eval_harness.analysis import classify_root_cause

    row = _make_analysis_row(evidence_hit=False, answer_verdict="incorrect")
    assert classify_root_cause(row, k=5) == "retrieval_failure"


def test_classify_root_cause_generation_failure():
    from financebench_eval_harness.analysis import classify_root_cause

    row = _make_analysis_row(
        evidence_hit=True, answer_verdict="incorrect",
        context_sufficiency="context_sufficient",
    )
    assert classify_root_cause(row, k=5) == "generation_failure"


def test_classify_root_cause_citation_failure():
    from financebench_eval_harness.analysis import classify_root_cause

    row = _make_analysis_row(
        evidence_hit=True,
        answer_verdict="correct",
        citation_quality="citation_invalid",
        context_sufficiency="context_sufficient",
    )
    assert classify_root_cause(row, k=5) == "citation_failure"


def test_classify_root_cause_hallucination_under_refusal():
    from financebench_eval_harness.analysis import classify_root_cause

    row = _make_analysis_row(
        evidence_hit=False,
        answer_verdict="incorrect",
        context_sufficiency="context_insufficient",
        grounding_label="under_refusal",
    )
    assert classify_root_cause(row, k=5) == "hallucination_under_refusal"


def test_classify_root_cause_over_refusal():
    from financebench_eval_harness.analysis import classify_root_cause

    row = _make_analysis_row(
        evidence_hit=True,
        answer_verdict="not_answered",
        context_sufficiency="context_sufficient",
        grounding_label="over_refusal",
    )
    assert classify_root_cause(row, k=5) == "over_refusal"


def test_classify_root_cause_no_failure():
    from financebench_eval_harness.analysis import classify_root_cause

    row = _make_analysis_row(
        evidence_hit=True,
        answer_verdict="correct",
        citation_quality="supports_answer",
        context_sufficiency="context_sufficient",
        grounding_label="grounded",
    )
    assert classify_root_cause(row, k=5) == "no_failure"


# ---------------------------------------------------------------------------
# Group 7: grounding_analysis.py — join_all_signals() (M7.10)
# ---------------------------------------------------------------------------


def _make_grounding_row(qid: str, grounding_label: str = "grounded",
                        context_sufficiency: str = "context_sufficient") -> dict:
    return {
        "question_id": qid,
        "grounding_label": grounding_label,
        "context_sufficiency": context_sufficiency,
        "rule_flags": [],
        "judge_failure_types": [],
        "cited_chunk_ids": [],
    }


def _make_citation_row(qid: str, citation_quality: str = "citation_missing") -> dict:
    return {
        "question_id": qid,
        "citation_quality": citation_quality,
        "cited_chunk_ids": [],
        "invalid_chunk_ids": [],
        "missing_citation": True,
    }


def _make_answer_score_row(qid: str, verdict: str = "correct") -> dict:
    return {
        "question_id": qid,
        "answer_verdict": verdict,
    }


def _make_retrieval_score_row(qid: str, evidence_hit: bool = True, k: int = 5) -> dict:
    return {
        "question_id": qid,
        f"evidence_text_hit@{k}": evidence_hit,
        f"page_hit@{k}": evidence_hit,
    }


def test_join_all_signals_produces_one_row():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1"),
        citation_row=_make_citation_row("q1"),
        answer_score_row=_make_answer_score_row("q1"),
        retrieval_score_row=_make_retrieval_score_row("q1"),
        k=5,
    )
    assert row["question_id"] == "q1"


def test_join_all_signals_includes_root_cause_field():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1"),
        citation_row=_make_citation_row("q1"),
        answer_score_row=_make_answer_score_row("q1"),
        retrieval_score_row=_make_retrieval_score_row("q1"),
        k=5,
    )
    assert "root_cause" in row


def test_join_correct_answer_supported_citation_is_no_failure():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1", "grounded", "context_sufficient"),
        citation_row=_make_citation_row("q1", "supports_answer") | {"missing_citation": False},
        answer_score_row=_make_answer_score_row("q1", "correct"),
        retrieval_score_row=_make_retrieval_score_row("q1", evidence_hit=True),
        k=5,
    )
    assert row["root_cause"] == "no_failure"


def test_join_wrong_answer_evidence_hit_is_generation_failure():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1", "ungrounded", "context_sufficient"),
        citation_row=_make_citation_row("q1", "citation_missing"),
        answer_score_row=_make_answer_score_row("q1", "incorrect"),
        retrieval_score_row=_make_retrieval_score_row("q1", evidence_hit=True),
        k=5,
    )
    assert row["root_cause"] == "generation_failure"


def test_join_wrong_answer_evidence_miss_is_retrieval_failure():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1", "insufficient_evidence", "context_insufficient"),
        citation_row=_make_citation_row("q1", "citation_missing"),
        answer_score_row=_make_answer_score_row("q1", "incorrect"),
        retrieval_score_row=_make_retrieval_score_row("q1", evidence_hit=False),
        k=5,
    )
    assert row["root_cause"] == "retrieval_failure"


def test_join_refusal_with_evidence_is_over_refusal():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1", "over_refusal", "context_sufficient"),
        citation_row=_make_citation_row("q1", "citation_missing"),
        answer_score_row=_make_answer_score_row("q1", "not_answered"),
        retrieval_score_row=_make_retrieval_score_row("q1", evidence_hit=True),
        k=5,
    )
    assert row["root_cause"] == "over_refusal"


def test_join_bad_citation_with_correct_answer_is_citation_failure():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1", "grounded", "context_sufficient"),
        citation_row=_make_citation_row("q1", "citation_invalid") | {"missing_citation": False},
        answer_score_row=_make_answer_score_row("q1", "correct"),
        retrieval_score_row=_make_retrieval_score_row("q1", evidence_hit=True),
        k=5,
    )
    assert row["root_cause"] == "citation_failure"


def test_join_all_signals_handles_none_answer_score():
    from financebench_eval_harness.grounding_analysis import join_all_signals

    row = join_all_signals(
        qid="q1",
        grounding_row=_make_grounding_row("q1"),
        citation_row=_make_citation_row("q1"),
        answer_score_row=None,
        retrieval_score_row=None,
        k=5,
    )
    assert row["question_id"] == "q1"
    assert "root_cause" in row


# ---------------------------------------------------------------------------
# Group 8: judge.py extension — parse_grounding_response_v2() (M7.8)
# ---------------------------------------------------------------------------


def _valid_v2_judge_json(
    grounding_label: str = "partially_grounded",
    failure_types: list[str] | None = None,
    context_sufficiency: str = "context_sufficient",
    citation_quality: str = "does_not_support_answer",
    reason: str = "The context contains the value but the answer is off.",
) -> str:
    return json.dumps({
        "grounding_label": grounding_label,
        "failure_types": failure_types or ["wrong_number"],
        "context_sufficiency": context_sufficiency,
        "citation_quality": citation_quality,
        "reason": reason,
    })


def test_parse_grounding_response_v2_parses_valid_json():
    from financebench_eval_harness.judge import parse_grounding_response_v2

    result = parse_grounding_response_v2(_valid_v2_judge_json())
    assert result["grounding_label"] == "partially_grounded"
    assert result["failure_types"] == ["wrong_number"]
    assert result["context_sufficiency"] == "context_sufficient"
    assert result["citation_quality"] == "does_not_support_answer"
    assert isinstance(result["reason"], str)


def test_parse_grounding_response_v2_rejects_unknown_grounding_label():
    from financebench_eval_harness.judge import JudgeError, parse_grounding_response_v2

    bad_json = _valid_v2_judge_json(grounding_label="completely_made_up")
    with pytest.raises(JudgeError):
        parse_grounding_response_v2(bad_json)


def test_parse_grounding_response_v2_rejects_unknown_failure_type():
    from financebench_eval_harness.judge import JudgeError, parse_grounding_response_v2

    bad_json = _valid_v2_judge_json(failure_types=["invented_type"])
    with pytest.raises(JudgeError):
        parse_grounding_response_v2(bad_json)


def test_parse_grounding_response_v2_allows_empty_failure_types():
    from financebench_eval_harness.judge import parse_grounding_response_v2

    result = parse_grounding_response_v2(_valid_v2_judge_json(failure_types=[]))
    assert result["failure_types"] == []


def test_render_grounding_prompt_v2_includes_question_and_gold_answer(tmp_path: Path):
    from financebench_eval_harness.judge import render_grounding_prompt_v2
    from financebench_eval_harness.run_config import JudgePromptConfig

    template_file = tmp_path / "grounding_v2.txt"
    template_file.write_text(
        "Q: {question}\nGold: {gold_answer}\nPred: {prediction}\n"
        "Context: {retrieved_context}\nCited: {cited_chunks}\nEvidence: {gold_evidence}",
        encoding="utf-8",
    )
    prompt_config = JudgePromptConfig(
        id="answer_grounding_v2",
        version="v2",
        template_path=template_file,
    )
    rendered = render_grounding_prompt_v2(
        prompt_config,
        question="What was the FY2018 capex?",
        gold_answer="$1,577 million",
        prediction="$1,577 million",
        retrieved_context="capex was $1,577 million",
        cited_chunks="doc1_p45_c02",
        gold_evidence="$1,577 million",
    )
    assert "FY2018 capex" in rendered.text
    assert "$1,577 million" in rendered.text


# ---------------------------------------------------------------------------
# Group 9: End-to-end analyze_grounding() (M7.10 deliverable)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_minimal_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "test_run"
    run_dir.mkdir(parents=True)

    predictions = [
        {
            "question_id": "q1",
            "question": "What was the FY2018 capex?",
            "gold_answer": "$1,577 million",
            "prediction": "$1,577 million",
            "retrieved_chunks": [
                {"chunk_id": "c1", "text": "capex was $1,577 million in FY2018"}
            ],
        }
    ]
    _write_jsonl(run_dir / "rag_predictions.jsonl", predictions)

    grounding_scores = [
        {
            "question_id": "q1",
            "verdict": "grounded",
            "reason": "Matches context.",
            "citation_correct": None,
            "unsupported_claims": None,
        }
    ]
    _write_jsonl(run_dir / "rag_grounding_scores.jsonl", grounding_scores)

    answer_scores = [
        {
            "question_id": "q1",
            "answer_verdict": "correct",
            "scores": {
                "numeric_match": True,
                "unit_match": True,
                "exact_match": False,
                "normalized_string_match": False,
                "gold_numeric_values": [1577.0],
            },
        }
    ]
    _write_jsonl(run_dir / "rag_answer_scores.jsonl", answer_scores)

    retrieval_scores = [
        {
            "question_id": "q1",
            "page_hit@5": True,
            "evidence_text_hit@5": True,
            "doc_hit@5": True,
        }
    ]
    _write_jsonl(run_dir / "retrieval_scores.jsonl", retrieval_scores)

    return run_dir


def test_analyze_grounding_writes_output_files(tmp_path: Path):
    from financebench_eval_harness.grounding_analysis import analyze_grounding
    from financebench_eval_harness.grounding_analysis_config import (
        GroundingAnalysisConfig,
        GroundingAnalysisSettings,
    )

    run_dir = _make_minimal_run_dir(tmp_path)
    config = GroundingAnalysisConfig(
        settings=GroundingAnalysisSettings(run_dir=run_dir, k=5),
        grounding_judge=None,
    )
    result = analyze_grounding(config)
    assert result.grounding_scores_path.is_file()
    assert result.citation_scores_path.is_file()
    assert result.failure_analysis_path.is_file()
    assert result.failure_summary_path.is_file()


def test_analyze_grounding_failure_analysis_has_root_cause(tmp_path: Path):
    from financebench_eval_harness.grounding_analysis import analyze_grounding
    from financebench_eval_harness.grounding_analysis_config import (
        GroundingAnalysisConfig,
        GroundingAnalysisSettings,
    )

    run_dir = _make_minimal_run_dir(tmp_path)
    config = GroundingAnalysisConfig(
        settings=GroundingAnalysisSettings(run_dir=run_dir, k=5),
        grounding_judge=None,
    )
    result = analyze_grounding(config)
    rows = [
        json.loads(line)
        for line in result.failure_analysis_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) >= 1
    assert all("root_cause" in row for row in rows)
