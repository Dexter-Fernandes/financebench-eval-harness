"""Tests for retrieval_scoring.py — M4.1/M4.3 retrieval evaluation targets."""

from __future__ import annotations

from financebench_eval_harness.retrieval_scoring import (
    answerable_hit,
    doc_hit,
    evidence_text_hit,
    normalize_doc_name,
    page_hit,
    score_retrieval_result,
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
