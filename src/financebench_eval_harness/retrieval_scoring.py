"""Retrieval evaluation scoring — M4.1/M4.3/M4.4.

Computes deterministic retrieval quality metrics directly from
retrieval_results.jsonl paired with gold evidence from examples.jsonl.
No LLM calls required.

Page number conventions
-----------------------
gold_page_num   — FinanceBench dataset annotation; may differ from PDF index by ±1.
matched_page_num — 1-based PDF page index where evidence text was actually found
                   (set by the page matcher in processed_examples.py).
chunk page_num  — 1-based PDF page index assigned by the document extractor;
                  same convention as matched_page_num.

page_hit compares chunk page_num against the union {gold_page_num, matched_page_num}
to avoid false negatives from the known ±1 annotation offset present in FinanceBench.
"""

from __future__ import annotations

from financebench_eval_harness.scoring import _normalize_text

__all__ = [
    "normalize_doc_name",
    "doc_hit",
    "page_hit",
    "evidence_text_hit",
    "answerable_hit",
    "score_retrieval_result",
    "score_hit_at_k",
    "summarize_retrieval_scores",
    "summarize_hit_at_k",
]

_SCORE_KEYS = ("doc_hit", "page_hit", "evidence_text_hit", "answerable_hit")
_HIT_AT_K_METRICS = ("doc_hit", "page_hit", "evidence_text_hit")
_DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)


# ---------------------------------------------------------------------------
# Public normalization helper
# ---------------------------------------------------------------------------


def normalize_doc_name(name: str) -> str:
    """Normalize a document name for comparison.

    Strips .pdf/.PDF suffix, lowercases, and strips surrounding whitespace.
    Handles the difference between retrieval results (always .pdf-suffixed)
    and gold evidence (no suffix).
    """
    return name.strip().removesuffix(".pdf").removesuffix(".PDF").lower()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _word_tokens(text: str) -> set[str]:
    return set(_normalize_text(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _page_candidates(ev: dict) -> set[int]:
    """Return the set of page numbers to accept as a hit for this evidence item.

    Includes both gold_page_num and matched_page_num (when present) so that
    chunks whose page_num aligns with either the FinanceBench annotation or
    the PDF-extraction index are counted as hits.
    """
    gold = ev["gold_page_num"]
    matched = ev.get("matched_page_num", gold)
    return {gold, matched}


def _top_k_chunks(retrieved: list[dict], top_k: int | None) -> list[dict]:
    """Return the top-k chunks sorted by rank ascending. top_k=None returns all."""
    if top_k is None:
        return retrieved
    return sorted(retrieved, key=lambda c: c.get("rank", 0))[:top_k]


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------


def doc_hit(
    retrieval_result: dict,
    gold_example: dict,
    top_k: int | None = None,
) -> bool:
    """Return True if any of the top-k retrieved chunks' doc_name matches any gold evidence doc_name.

    Comparison is case-insensitive and strips the .pdf suffix.
    top_k=None (default) considers all retrieved chunks.
    """
    gold_docs = {normalize_doc_name(ev["doc_name"]) for ev in gold_example.get("evidence", [])}
    chunks = _top_k_chunks(retrieval_result.get("retrieved", []), top_k)
    return any(normalize_doc_name(chunk["doc_name"]) in gold_docs for chunk in chunks)


def page_hit(
    retrieval_result: dict,
    gold_example: dict,
    top_k: int | None = None,
) -> bool:
    """Return True if any of the top-k retrieved chunks matches any gold evidence on both doc and page.

    Accepts chunk page_num matching either gold_page_num or matched_page_num from
    the gold evidence — see module docstring for page number conventions.
    top_k=None (default) considers all retrieved chunks.
    """
    gold_pages: set[tuple[str, int]] = set()
    for ev in gold_example.get("evidence", []):
        doc = normalize_doc_name(ev["doc_name"])
        for page in _page_candidates(ev):
            gold_pages.add((doc, page))
    chunks = _top_k_chunks(retrieval_result.get("retrieved", []), top_k)
    return any(
        (normalize_doc_name(chunk["doc_name"]), chunk["page_num"]) in gold_pages
        for chunk in chunks
    )


def evidence_text_hit(
    retrieval_result: dict,
    gold_example: dict,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> bool:
    """Return True if any of the top-k retrieved chunks has sufficient word-overlap with any gold evidence text.

    Uses Jaccard overlap of normalized word tokens. threshold=0.5 by default.
    top_k=None (default) considers all retrieved chunks.
    """
    gold_token_sets = [
        _word_tokens(ev["evidence_text"]) for ev in gold_example.get("evidence", [])
    ]
    chunks = _top_k_chunks(retrieval_result.get("retrieved", []), top_k)
    for chunk in chunks:
        chunk_tokens = _word_tokens(chunk.get("text", ""))
        for gold_tokens in gold_token_sets:
            if _jaccard(chunk_tokens, gold_tokens) >= threshold:
                return True
    return False


def answerable_hit(
    retrieval_result: dict,
    gold_example: dict,
    threshold: float = 0.5,
    top_k: int | None = None,
) -> bool:
    """Return True if concatenated top-k chunk texts have sufficient overlap with any gold evidence text.

    Concatenates retrieved chunks (sorted by rank, up to top_k) and computes
    Jaccard against each gold evidence text. top_k=None uses all chunks.
    """
    chunks = _top_k_chunks(retrieval_result.get("retrieved", []), top_k)
    if not chunks:
        return False
    combined_tokens = _word_tokens(" ".join(c.get("text", "") for c in chunks))
    return any(
        _jaccard(combined_tokens, _word_tokens(ev["evidence_text"])) >= threshold
        for ev in gold_example.get("evidence", [])
    )


def score_retrieval_result(
    retrieval_result: dict,
    gold_example: dict,
    overlap_threshold: float = 0.5,
    top_k: int | None = None,
) -> dict[str, bool]:
    """Compute all four retrieval metrics for one question's retrieval result."""
    return {
        "doc_hit": doc_hit(retrieval_result, gold_example, top_k=top_k),
        "page_hit": page_hit(retrieval_result, gold_example, top_k=top_k),
        "evidence_text_hit": evidence_text_hit(retrieval_result, gold_example, threshold=overlap_threshold, top_k=top_k),
        "answerable_hit": answerable_hit(retrieval_result, gold_example, threshold=overlap_threshold, top_k=top_k),
    }


def score_hit_at_k(
    retrieval_result: dict,
    gold_example: dict,
    ks: tuple[int, ...] = _DEFAULT_KS,
    overlap_threshold: float = 0.5,
) -> dict[str, bool]:
    """Compute hit@k metrics for doc, page, and evidence text at each value of k.

    Returns a flat dict with keys like "doc_hit@1", "page_hit@5", "evidence_text_hit@10".
    answerable_hit is excluded — it concatenates chunks (different semantic from per-chunk hit).
    """
    scores: dict[str, bool] = {}
    for k in ks:
        scores[f"doc_hit@{k}"] = doc_hit(retrieval_result, gold_example, top_k=k)
        scores[f"page_hit@{k}"] = page_hit(retrieval_result, gold_example, top_k=k)
        scores[f"evidence_text_hit@{k}"] = evidence_text_hit(
            retrieval_result, gold_example, threshold=overlap_threshold, top_k=k
        )
    return scores


def summarize_retrieval_scores(scores: list[dict[str, bool]]) -> dict[str, object]:
    """Aggregate per-question retrieval scores into counts and rates."""
    n = len(scores)
    summary: dict[str, object] = {"example_count": n}
    for key in _SCORE_KEYS:
        count = sum(1 for s in scores if s.get(key) is True)
        summary[f"{key}_count"] = count
        summary[f"{key}_rate"] = count / n if n else 0.0
    return summary


def summarize_hit_at_k(
    scores: list[dict[str, bool]],
    ks: tuple[int, ...] = _DEFAULT_KS,
) -> dict[str, object]:
    """Aggregate per-question hit@k scores into counts and rates.

    Expects scores produced by score_hit_at_k with the same ks.
    """
    n = len(scores)
    summary: dict[str, object] = {"example_count": n}
    for metric in _HIT_AT_K_METRICS:
        for k in ks:
            key = f"{metric}@{k}"
            count = sum(1 for s in scores if s.get(key) is True)
            summary[f"{key}_count"] = count
            summary[f"{key}_rate"] = count / n if n else 0.0
    return summary
