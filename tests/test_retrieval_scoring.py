"""Tests for retrieval_scoring.py — M4.1/M4.3/M4.4/M4.5 retrieval evaluation targets."""

from __future__ import annotations

import pytest

from financebench_eval_harness.retrieval_scoring import (
    answerable_hit,
    doc_hit,
    evidence_text_hit,
    normalize_doc_name,
    page_hit,
    score_hit_at_k,
    score_rank_metrics,
    score_retrieval_result,
    summarize_hit_at_k,
    summarize_rank_metrics,
    summarize_retrieval_scores,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_GOLD_3M = {
    "evidence": [
        {
            "doc_name": "3M_2018_10K",
            "gold_page_num": 60,
            "evidence_text": "purchases of property plant and equipment 1577",
        }
    ]
}

_RETRIEVED_3M_MATCH = {
    "retrieved": [
        {
            "rank": 1,
            "doc_name": "3M_2018_10K.pdf",
            "page_num": 60,
            "score": 0.9,
            "text": "Purchases of property, plant and equipment (PP&E) (1,577)",
        }
    ]
}

_RETRIEVED_AMAZON = {
    "retrieved": [
        {
            "rank": 1,
            "doc_name": "AMAZON_2019_10K.pdf",
            "page_num": 50,
            "score": 0.7,
            "text": "Net income including noncontrolling interest 5363",
        }
    ]
}

# ---------------------------------------------------------------------------
# Slice 1 — Tracer bullet: score_retrieval_result returns correct shape
# ---------------------------------------------------------------------------


def test_score_retrieval_result_returns_all_four_metric_keys() -> None:
    retrieval_result = {"question_id": "q0", "query": "...", "retrieved": []}
    gold_example = {"question_id": "q0", "evidence": []}
    score = score_retrieval_result(retrieval_result, gold_example)
    assert set(score.keys()) == {"doc_hit", "page_hit", "evidence_text_hit", "answerable_hit"}
    assert all(isinstance(v, bool) for v in score.values())


# ---------------------------------------------------------------------------
# Slice 2 — doc_hit
# ---------------------------------------------------------------------------


def test_doc_hit_true_when_retrieved_chunk_matches_gold_doc() -> None:
    assert doc_hit(_RETRIEVED_3M_MATCH, _GOLD_3M) is True


def test_doc_hit_false_when_no_chunk_matches_gold_doc() -> None:
    assert doc_hit(_RETRIEVED_AMAZON, _GOLD_3M) is False


def test_doc_hit_false_when_retrieved_list_is_empty() -> None:
    assert doc_hit({"retrieved": []}, _GOLD_3M) is False


def test_doc_hit_strips_pdf_suffix_case_insensitively() -> None:
    result = {"retrieved": [{"rank": 1, "doc_name": "3M_2018_10K.PDF", "page_num": 1, "score": 0.5, "text": ""}]}
    assert doc_hit(result, _GOLD_3M) is True


# ---------------------------------------------------------------------------
# Slice 3 — page_hit
# ---------------------------------------------------------------------------


def test_page_hit_true_when_doc_and_page_both_match() -> None:
    assert page_hit(_RETRIEVED_3M_MATCH, _GOLD_3M) is True


def test_page_hit_false_when_doc_matches_but_page_does_not() -> None:
    result = {"retrieved": [{"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 99, "score": 0.9, "text": ""}]}
    assert page_hit(result, _GOLD_3M) is False


def test_page_hit_false_when_doc_does_not_match() -> None:
    assert page_hit(_RETRIEVED_AMAZON, _GOLD_3M) is False


def test_page_hit_uses_gold_page_num_field() -> None:
    gold = {"evidence": [{"doc_name": "3M_2018_10K", "gold_page_num": 59, "evidence_text": "..."}]}
    result = {"retrieved": [{"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 59, "score": 0.9, "text": ""}]}
    assert page_hit(result, gold) is True


# ---------------------------------------------------------------------------
# Slice 4 — evidence_text_hit
# ---------------------------------------------------------------------------

_GOLD_WITH_TEXT = {
    "evidence": [
        {
            "doc_name": "3M_2018_10K",
            "gold_page_num": 60,
            "evidence_text": "purchases of property plant and equipment 1577",
        }
    ]
}


def test_evidence_text_hit_true_when_chunk_has_high_word_overlap() -> None:
    assert evidence_text_hit(_RETRIEVED_3M_MATCH, _GOLD_WITH_TEXT) is True


def test_evidence_text_hit_false_when_chunk_has_low_word_overlap() -> None:
    assert evidence_text_hit(_RETRIEVED_AMAZON, _GOLD_WITH_TEXT) is False


def test_evidence_text_hit_respects_custom_threshold() -> None:
    result = {
        "retrieved": [{"rank": 1, "doc_name": "other.pdf", "page_num": 1, "score": 0.5, "text": "property"}]
    }
    assert evidence_text_hit(result, _GOLD_WITH_TEXT, threshold=0.0) is True


def test_evidence_text_hit_false_when_retrieved_list_empty() -> None:
    assert evidence_text_hit({"retrieved": []}, _GOLD_WITH_TEXT) is False


# ---------------------------------------------------------------------------
# Slice 5 — answerable_hit
# ---------------------------------------------------------------------------


def test_answerable_hit_true_when_concatenated_context_overlaps_gold() -> None:
    result = {
        "retrieved": [
            {"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 60, "score": 0.9,
             "text": "Purchases of property plant and equipment"},
            {"rank": 2, "doc_name": "3M_2018_10K.pdf", "page_num": 60, "score": 0.8,
             "text": "1577 capital expenditure investing activities"},
        ]
    }
    gold = {"evidence": [{"doc_name": "3M_2018_10K", "gold_page_num": 60,
                          "evidence_text": "purchases of property plant and equipment 1577"}]}
    assert answerable_hit(result, gold) is True


def test_answerable_hit_top_k_limits_chunks_used() -> None:
    result = {
        "retrieved": [
            {"rank": 1, "doc_name": "AMAZON_2019_10K.pdf", "page_num": 1, "score": 0.9,
             "text": "completely unrelated text about amazon"},
            {"rank": 2, "doc_name": "3M_2018_10K.pdf", "page_num": 60, "score": 0.8,
             "text": "purchases of property plant and equipment 1577"},
        ]
    }
    gold = {"evidence": [{"doc_name": "3M_2018_10K", "gold_page_num": 60,
                          "evidence_text": "purchases of property plant and equipment 1577"}]}
    assert answerable_hit(result, gold, top_k=1) is False
    assert answerable_hit(result, gold, top_k=2) is True


def test_answerable_hit_false_when_retrieved_empty() -> None:
    assert answerable_hit({"retrieved": []}, _GOLD_WITH_TEXT) is False


# ---------------------------------------------------------------------------
# Slice 5 — summarize_retrieval_scores
# ---------------------------------------------------------------------------


def test_summarize_retrieval_scores_counts_and_rates() -> None:
    scores = [
        {"doc_hit": True, "page_hit": True, "evidence_text_hit": True, "answerable_hit": True},
        {"doc_hit": False, "page_hit": False, "evidence_text_hit": False, "answerable_hit": False},
    ]
    summary = summarize_retrieval_scores(scores)
    assert summary == {
        "example_count": 2,
        "doc_hit_count": 1,
        "doc_hit_rate": 0.5,
        "page_hit_count": 1,
        "page_hit_rate": 0.5,
        "evidence_text_hit_count": 1,
        "evidence_text_hit_rate": 0.5,
        "answerable_hit_count": 1,
        "answerable_hit_rate": 0.5,
    }


def test_summarize_retrieval_scores_empty_list() -> None:
    summary = summarize_retrieval_scores([])
    assert summary["example_count"] == 0
    assert summary["doc_hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# M4.3 — normalize_doc_name (public function, replaces _strip_pdf)
# ---------------------------------------------------------------------------


def test_normalize_doc_name_strips_pdf_suffix() -> None:
    assert normalize_doc_name("3M_2018_10K.pdf") == "3m_2018_10k"


def test_normalize_doc_name_strips_uppercase_pdf_suffix() -> None:
    assert normalize_doc_name("3M_2018_10K.PDF") == "3m_2018_10k"


def test_normalize_doc_name_lowercases_result() -> None:
    assert normalize_doc_name("AMAZON_2019_10K") == "amazon_2019_10k"


def test_normalize_doc_name_strips_surrounding_whitespace() -> None:
    assert normalize_doc_name("  3M_2018_10K.pdf  ") == "3m_2018_10k"


# ---------------------------------------------------------------------------
# M4.3 — page_hit uses matched_page_num (the real bug fix)
# ---------------------------------------------------------------------------

_GOLD_WITH_OFFSET = {
    "evidence": [
        {
            "doc_name": "3M_2018_10K",
            "gold_page_num": 59,
            "matched_page_num": 60,  # PDF extraction page (what the retriever sees)
            "evidence_text": "purchases of property plant and equipment 1577",
        }
    ]
}


def test_page_hit_true_when_chunk_matches_matched_page_num() -> None:
    # chunk page_num=60 (PDF index) matches matched_page_num=60, not gold_page_num=59
    result = {"retrieved": [{"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 60, "score": 0.9, "text": ""}]}
    assert page_hit(result, _GOLD_WITH_OFFSET) is True


def test_page_hit_true_when_chunk_matches_gold_page_num() -> None:
    # chunk page_num=59 matches gold_page_num=59 (accept both candidates)
    result = {"retrieved": [{"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 59, "score": 0.9, "text": ""}]}
    assert page_hit(result, _GOLD_WITH_OFFSET) is True


def test_page_hit_false_when_chunk_page_matches_neither_candidate() -> None:
    result = {"retrieved": [{"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 99, "score": 0.9, "text": ""}]}
    assert page_hit(result, _GOLD_WITH_OFFSET) is False


def test_page_hit_works_without_matched_page_num_key() -> None:
    # Evidence item with only gold_page_num (no matched_page_num key)
    gold = {"evidence": [{"doc_name": "3M_2018_10K", "gold_page_num": 59, "evidence_text": "..."}]}
    result = {"retrieved": [{"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 59, "score": 0.9, "text": ""}]}
    assert page_hit(result, gold) is True


# ---------------------------------------------------------------------------
# M4.4 — hit@k: shared fixture
# ---------------------------------------------------------------------------

# Correct chunk is at rank 7; ranks 1-6 are irrelevant docs.
_GOLD_3M_PAGE = {
    "evidence": [
        {
            "doc_name": "3M_2018_10K",
            "gold_page_num": 60,
            "matched_page_num": 60,
            "evidence_text": "purchases of property plant and equipment 1577",
        }
    ]
}

_RESULT_HIT_AT_RANK_7 = {
    "retrieved": [
        {"rank": i, "doc_name": "AMAZON_2019_10K.pdf", "page_num": i, "score": 0.9 - i * 0.01, "text": "unrelated"}
        for i in range(1, 7)
    ] + [
        {"rank": 7, "doc_name": "3M_2018_10K.pdf", "page_num": 60, "score": 0.3,
         "text": "Purchases of property plant and equipment 1577"}
    ]
}

# ---------------------------------------------------------------------------
# M4.4 — Slice 1: doc_hit with top_k
# ---------------------------------------------------------------------------


def test_doc_hit_top_k_false_when_correct_chunk_beyond_k() -> None:
    assert doc_hit(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, top_k=5) is False


def test_doc_hit_top_k_true_when_correct_chunk_within_k() -> None:
    assert doc_hit(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, top_k=10) is True


def test_doc_hit_top_k_none_uses_all_chunks() -> None:
    assert doc_hit(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, top_k=None) is True


# ---------------------------------------------------------------------------
# M4.4 — Slice 2: page_hit with top_k
# ---------------------------------------------------------------------------


def test_page_hit_top_k_false_when_correct_chunk_beyond_k() -> None:
    assert page_hit(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, top_k=5) is False


def test_page_hit_top_k_true_when_correct_chunk_within_k() -> None:
    assert page_hit(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, top_k=10) is True


# ---------------------------------------------------------------------------
# M4.4 — Slice 3: evidence_text_hit with top_k
# ---------------------------------------------------------------------------


def test_evidence_text_hit_top_k_false_when_correct_chunk_beyond_k() -> None:
    assert evidence_text_hit(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, top_k=5) is False


def test_evidence_text_hit_top_k_true_when_correct_chunk_within_k() -> None:
    assert evidence_text_hit(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, top_k=10) is True


# ---------------------------------------------------------------------------
# M4.4 — Slice 4: score_hit_at_k
# ---------------------------------------------------------------------------


def test_score_hit_at_k_returns_expected_keys() -> None:
    scores = score_hit_at_k(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, ks=(1, 5, 10))
    expected_keys = {
        "doc_hit@1", "doc_hit@5", "doc_hit@10",
        "page_hit@1", "page_hit@5", "page_hit@10",
        "evidence_text_hit@1", "evidence_text_hit@5", "evidence_text_hit@10",
    }
    assert set(scores.keys()) == expected_keys
    assert all(isinstance(v, bool) for v in scores.values())


def test_score_hit_at_k_correct_values_for_rank_7_hit() -> None:
    scores = score_hit_at_k(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE, ks=(5, 10))
    assert scores["page_hit@5"] is False
    assert scores["page_hit@10"] is True
    assert scores["doc_hit@5"] is False
    assert scores["doc_hit@10"] is True


def test_score_hit_at_k_default_ks_produces_four_k_values() -> None:
    scores = score_hit_at_k(_RESULT_HIT_AT_RANK_7, _GOLD_3M_PAGE)
    assert "doc_hit@1" in scores
    assert "doc_hit@20" in scores
    assert len(scores) == 3 * 4  # 3 metrics × 4 k values


# ---------------------------------------------------------------------------
# M4.4 — Slice 5: summarize_hit_at_k
# ---------------------------------------------------------------------------


def test_summarize_hit_at_k_counts_and_rates() -> None:
    scores = [
        {"doc_hit@1": True,  "doc_hit@5": True,  "page_hit@1": False, "page_hit@5": True,
         "evidence_text_hit@1": False, "evidence_text_hit@5": True},
        {"doc_hit@1": False, "doc_hit@5": True,  "page_hit@1": False, "page_hit@5": False,
         "evidence_text_hit@1": False, "evidence_text_hit@5": False},
    ]
    summary = summarize_hit_at_k(scores, ks=(1, 5))
    assert summary["example_count"] == 2
    assert summary["doc_hit@1_count"] == 1
    assert summary["doc_hit@1_rate"] == 0.5
    assert summary["doc_hit@5_count"] == 2
    assert summary["doc_hit@5_rate"] == 1.0
    assert summary["page_hit@5_count"] == 1
    assert summary["page_hit@5_rate"] == 0.5


def test_summarize_hit_at_k_empty_list() -> None:
    summary = summarize_hit_at_k([], ks=(1, 5, 10, 20))
    assert summary["example_count"] == 0
    assert summary["doc_hit@1_rate"] == 0.0


# ---------------------------------------------------------------------------
# M4.5 — Rank-based metrics shared fixture
# ---------------------------------------------------------------------------

# Correct chunk at rank 4; ranks 1-3 are wrong doc.
_RESULT_HIT_AT_RANK_4 = {
    "retrieved": [
        {"rank": i, "doc_name": "AMAZON_2019_10K.pdf", "page_num": i, "score": 0.9 - i * 0.05, "text": "unrelated"}
        for i in range(1, 4)
    ] + [
        {"rank": 4, "doc_name": "3M_2018_10K.pdf", "page_num": 60, "score": 0.5,
         "text": "Purchases of property plant and equipment 1577"}
    ]
}

_GOLD_3M_RANK = {
    "evidence": [
        {
            "doc_name": "3M_2018_10K",
            "gold_page_num": 59,
            "matched_page_num": 60,
            "evidence_text": "purchases of property plant and equipment 1577",
        }
    ]
}

# ---------------------------------------------------------------------------
# M4.5 — Slice 1: score_rank_metrics doc_first_hit_rank
# ---------------------------------------------------------------------------


def test_score_rank_metrics_returns_three_keys() -> None:
    result = score_rank_metrics(_RESULT_HIT_AT_RANK_4, _GOLD_3M_RANK)
    assert set(result.keys()) == {
        "doc_first_hit_rank", "page_first_hit_rank", "evidence_text_first_hit_rank"
    }


def test_score_rank_metrics_doc_first_hit_rank_correct() -> None:
    result = score_rank_metrics(_RESULT_HIT_AT_RANK_4, _GOLD_3M_RANK)
    assert result["doc_first_hit_rank"] == 4


# ---------------------------------------------------------------------------
# M4.5 — Slice 2: None when no hit
# ---------------------------------------------------------------------------


def test_score_rank_metrics_returns_none_when_no_doc_match() -> None:
    result_no_hit = {
        "retrieved": [
            {"rank": 1, "doc_name": "AMAZON_2019_10K.pdf", "page_num": 1, "score": 0.9, "text": "unrelated"},
        ]
    }
    result = score_rank_metrics(result_no_hit, _GOLD_3M_RANK)
    assert result["doc_first_hit_rank"] is None
    assert result["page_first_hit_rank"] is None
    assert result["evidence_text_first_hit_rank"] is None


def test_score_rank_metrics_returns_none_on_empty_retrieved() -> None:
    result = score_rank_metrics({"retrieved": []}, _GOLD_3M_RANK)
    assert result["doc_first_hit_rank"] is None


# ---------------------------------------------------------------------------
# M4.5 — Slice 3: page_first_hit_rank uses _page_candidates (matched_page_num)
# ---------------------------------------------------------------------------


def test_score_rank_metrics_page_first_hit_rank_uses_matched_page_num() -> None:
    # gold_page_num=59, matched_page_num=60; chunk has page_num=60 at rank 4
    result = score_rank_metrics(_RESULT_HIT_AT_RANK_4, _GOLD_3M_RANK)
    assert result["page_first_hit_rank"] == 4


def test_score_rank_metrics_page_hit_rank_none_when_page_wrong() -> None:
    result_wrong_page = {
        "retrieved": [
            {"rank": 1, "doc_name": "3M_2018_10K.pdf", "page_num": 99, "score": 0.9, "text": ""}
        ]
    }
    result = score_rank_metrics(result_wrong_page, _GOLD_3M_RANK)
    assert result["page_first_hit_rank"] is None


# ---------------------------------------------------------------------------
# M4.5 — Slice 4: summarize_rank_metrics MRR@k
# ---------------------------------------------------------------------------


def test_summarize_rank_metrics_mrr_with_hit_at_rank_4_and_miss() -> None:
    # Question 1: doc hit at rank 4. Question 2: miss.
    # MRR@5 = (1/4 + 0) / 2 = 0.125
    scores = [
        {"doc_first_hit_rank": 4, "page_first_hit_rank": 4, "evidence_text_first_hit_rank": None},
        {"doc_first_hit_rank": None, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None},
    ]
    summary = summarize_rank_metrics(scores, ks=(5,))
    assert summary["example_count"] == 2
    assert summary["doc_mrr@5"] == pytest.approx(0.125)


def test_summarize_rank_metrics_mrr_excludes_hit_beyond_k() -> None:
    # Hit at rank 7, k=5: reciprocal rank = 0
    scores = [{"doc_first_hit_rank": 7, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None}]
    summary = summarize_rank_metrics(scores, ks=(5,))
    assert summary["doc_mrr@5"] == pytest.approx(0.0)


def test_summarize_rank_metrics_mrr_two_hits() -> None:
    # Rank 4 and rank 2: MRR@5 = (0.25 + 0.5) / 2 = 0.375
    scores = [
        {"doc_first_hit_rank": 4, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None},
        {"doc_first_hit_rank": 2, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None},
    ]
    summary = summarize_rank_metrics(scores, ks=(5,))
    assert summary["doc_mrr@5"] == pytest.approx(0.375)


def test_summarize_rank_metrics_empty_list() -> None:
    summary = summarize_rank_metrics([], ks=(5, 10))
    assert summary["example_count"] == 0
    assert summary["doc_mrr@5"] == pytest.approx(0.0)
    assert summary["doc_median_first_hit_rank"] is None


# ---------------------------------------------------------------------------
# M4.5 — Slice 5: median_first_hit_rank
# ---------------------------------------------------------------------------


def test_summarize_rank_metrics_median_two_hits() -> None:
    scores = [
        {"doc_first_hit_rank": 3, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None},
        {"doc_first_hit_rank": 7, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None},
    ]
    summary = summarize_rank_metrics(scores, ks=(10,))
    assert summary["doc_median_first_hit_rank"] == pytest.approx(5.0)


def test_summarize_rank_metrics_median_all_misses_is_none() -> None:
    scores = [{"doc_first_hit_rank": None, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None}]
    summary = summarize_rank_metrics(scores, ks=(10,))
    assert summary["doc_median_first_hit_rank"] is None


def test_summarize_rank_metrics_median_single_hit() -> None:
    scores = [{"doc_first_hit_rank": 1, "page_first_hit_rank": None, "evidence_text_first_hit_rank": None}]
    summary = summarize_rank_metrics(scores, ks=(5,))
    assert summary["doc_median_first_hit_rank"] == pytest.approx(1.0)
