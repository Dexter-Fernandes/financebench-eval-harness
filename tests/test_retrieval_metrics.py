"""M4.14 — scenario-level tests for retrieval metric correctness.

These tests document expected metric behaviour at the pipeline level, complementing
the lower-level unit tests in test_retrieval_scoring.py.
"""

from __future__ import annotations

import pytest

from financebench_eval_harness.retrieval_scoring import (
    score_evidence_overlap,
    score_hit_at_k,
    score_rank_metrics,
    summarize_hit_at_k,
    summarize_rank_metrics,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _result(retrieved: list[dict]) -> dict:
    return {"retrieved": retrieved}


def _chunk(rank: int, doc_name: str, page_num: int, text: str = "some text", score: float = 0.9) -> dict:
    return {"rank": rank, "chunk_id": f"c{rank}", "doc_name": doc_name,
            "page_num": page_num, "score": score, "text": text}


def _gold(evidence: list[dict]) -> dict:
    return {"evidence": evidence}


def _evidence(
    doc_name: str,
    gold_page_num: int,
    evidence_text: str = "some text",
    matched_page_num: int | None = None,
) -> dict:
    ev: dict = {"doc_name": doc_name, "gold_page_num": gold_page_num, "evidence_text": evidence_text}
    if matched_page_num is not None:
        ev["matched_page_num"] = matched_page_num
    return ev


# ---------------------------------------------------------------------------
# Doc hit
# ---------------------------------------------------------------------------


def test_doc_hit_at_k_true_when_correct_document_at_rank_1() -> None:
    result = _result([_chunk(1, "3M_2022_10K.pdf", 12)])
    gold = _gold([_evidence("3M_2022_10K", 12)])
    assert score_hit_at_k(result, gold, ks=(5,))["doc_hit@5"] is True


def test_doc_hit_at_k_false_when_correct_document_beyond_top_k() -> None:
    chunks = [_chunk(r, "WRONG.pdf", 1) for r in range(1, 6)]
    chunks.append(_chunk(6, "3M_2022_10K.pdf", 12))
    result = _result(chunks)
    gold = _gold([_evidence("3M_2022_10K", 12)])
    assert score_hit_at_k(result, gold, ks=(5,))["doc_hit@5"] is False


# ---------------------------------------------------------------------------
# Page hit
# ---------------------------------------------------------------------------


def test_page_hit_at_k_true_when_correct_page_at_rank_1() -> None:
    result = _result([_chunk(1, "3M_2022_10K.pdf", 12)])
    gold = _gold([_evidence("3M_2022_10K", 12)])
    assert score_hit_at_k(result, gold, ks=(5,))["page_hit@5"] is True


def test_page_hit_at_k_false_when_correct_page_beyond_top_k() -> None:
    chunks = [_chunk(r, "3M_2022_10K.pdf", 99) for r in range(1, 6)]
    chunks.append(_chunk(6, "3M_2022_10K.pdf", 12))
    result = _result(chunks)
    gold = _gold([_evidence("3M_2022_10K", 12)])
    assert score_hit_at_k(result, gold, ks=(5,))["page_hit@5"] is False


def test_page_hit_uses_gold_page_num_from_evidence() -> None:
    result = _result([_chunk(1, "3M_2018_10K.pdf", 59)])
    gold = _gold([_evidence("3M_2018_10K", 59, matched_page_num=60)])
    assert score_hit_at_k(result, gold, ks=(5,))["page_hit@5"] is True


def test_page_hit_also_accepts_matched_page_num() -> None:
    result = _result([_chunk(1, "3M_2018_10K.pdf", 60)])
    gold = _gold([_evidence("3M_2018_10K", 59, matched_page_num=60)])
    assert score_hit_at_k(result, gold, ks=(5,))["page_hit@5"] is True


# ---------------------------------------------------------------------------
# Evidence text hit
# ---------------------------------------------------------------------------


def test_evidence_text_hit_false_when_evidence_missing() -> None:
    result = _result([_chunk(1, "3M_2022_10K.pdf", 12, text="revenues were high")])
    gold = _gold([])
    assert score_hit_at_k(result, gold, ks=(5,))["evidence_text_hit@5"] is False


def test_evidence_text_hit_true_when_overlap_meets_threshold() -> None:
    text = "net revenues were thirty four billion dollars"
    result = _result([_chunk(1, "3M_2022_10K.pdf", 12, text=text)])
    gold = _gold([_evidence("3M_2022_10K", 12, evidence_text=text)])
    assert score_hit_at_k(result, gold, ks=(5,), overlap_threshold=0.5)["evidence_text_hit@5"] is True


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


def test_mrr_correct_when_first_hit_is_at_rank_2() -> None:
    scores = [{"doc_first_hit_rank": 2, "page_first_hit_rank": None,
               "evidence_text_first_hit_rank": None}]
    summary = summarize_rank_metrics(scores, ks=(5,))
    assert summary["doc_mrr@5"] == pytest.approx(0.5)


def test_mrr_zero_when_no_hit_within_k() -> None:
    scores = [{"doc_first_hit_rank": None, "page_first_hit_rank": None,
               "evidence_text_first_hit_rank": None}]
    summary = summarize_rank_metrics(scores, ks=(5,))
    assert summary["doc_mrr@5"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Evidence overlap — normalization
# ---------------------------------------------------------------------------


def test_evidence_overlap_is_case_insensitive() -> None:
    result = _result([_chunk(1, "DOC.pdf", 1, text="Revenue Was High")])
    gold = _gold([_evidence("DOC", 1, evidence_text="revenue was high")])
    assert score_evidence_overlap(result, gold, threshold=0.5)["best_evidence_overlap"] == pytest.approx(1.0)


def test_evidence_overlap_handles_extra_whitespace() -> None:
    result = _result([_chunk(1, "DOC.pdf", 1, text="revenue   was   high")])
    gold = _gold([_evidence("DOC", 1, evidence_text="revenue was high")])
    assert score_evidence_overlap(result, gold, threshold=0.5)["best_evidence_overlap"] == pytest.approx(1.0)


def test_evidence_overlap_partial_token_coverage() -> None:
    result = _result([_chunk(1, "DOC.pdf", 1, text="alpha beta gamma extra")])
    gold = _gold([_evidence("DOC", 1, evidence_text="alpha beta gamma delta")])
    overlap = score_evidence_overlap(result, gold, threshold=0.9)
    assert 0.0 < overlap["best_evidence_overlap"] < 1.0
    assert overlap["evidence_text_hit"] is False


# ---------------------------------------------------------------------------
# Missing rank is null
# ---------------------------------------------------------------------------


def test_missing_rank_is_none_when_document_not_retrieved() -> None:
    result = _result([_chunk(1, "WRONG.pdf", 1)])
    gold = _gold([_evidence("3M_2022_10K", 12)])
    metrics = score_rank_metrics(result, gold)
    assert metrics["doc_first_hit_rank"] is None
    assert metrics["page_first_hit_rank"] is None


# ---------------------------------------------------------------------------
# Optional — multi-k evaluation
# ---------------------------------------------------------------------------


def test_doc_hit_false_at_k5_true_at_k10_when_correct_doc_at_rank_7() -> None:
    chunks = [_chunk(r, "WRONG.pdf", 1) for r in range(1, 7)]
    chunks.append(_chunk(7, "3M_2022_10K.pdf", 12))
    result = _result(chunks)
    gold = _gold([_evidence("3M_2022_10K", 12)])
    scores = score_hit_at_k(result, gold, ks=(5, 10))
    assert scores["doc_hit@5"] is False
    assert scores["doc_hit@10"] is True


def test_summarize_hit_at_k_rates_correct_for_multiple_ks() -> None:
    # q1: correct doc at rank 1 → doc_hit@1=True, doc_hit@5=True
    q1 = score_hit_at_k(_result([_chunk(1, "DOC.pdf", 1)]),
                         _gold([_evidence("DOC", 1)]), ks=(1, 5))
    # q2: wrong doc at rank 1, correct doc at rank 2 → doc_hit@1=False, doc_hit@5=True
    q2 = score_hit_at_k(
        _result([_chunk(1, "WRONG.pdf", 99), _chunk(2, "DOC.pdf", 1)]),
        _gold([_evidence("DOC", 1)]),
        ks=(1, 5),
    )
    summary = summarize_hit_at_k([q1, q2], ks=(1, 5))
    assert summary["doc_hit@1_rate"] == pytest.approx(0.5)
    assert summary["doc_hit@5_rate"] == pytest.approx(1.0)
