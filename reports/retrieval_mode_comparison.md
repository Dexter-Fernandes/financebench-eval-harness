# Run Comparison

Sorted by: `doc_mrr@10`

**Best run:** `m9_10_octen_dense`

| run_id | mode | model | accuracy | evidence_hit@10 | doc_hit@10 | mrr@10 |
| --- | --- | --- | --- | --- | --- | --- |
| m9_10_octen_dense ★ | dense | Octen/Octen-Embedding-0.6B | — | 0.217 | 0.978 | 0.797 |
| dw10lw1_octen_hybrid | hybrid | Octen/Octen-Embedding-0.6B | — | 0.217 | 0.957 | 0.756 |
| m9_10_granite_dense | dense | granite-embedding:278m | — | 0.196 | 0.978 | 0.713 |
| dw3lw1_granite_hybrid | hybrid | granite-embedding:278m | — | 0.217 | 0.978 | 0.709 |
| m9_10_octen_hybrid_reranked | hybrid_reranked | Octen/Octen-Embedding-0.6B | — | 0.043 | 0.957 | 0.695 |
| granite_hybrid_reranked_3to1 | hybrid_reranked | granite-embedding:278m | — | 0.022 | 0.957 | 0.673 |
| m9_10_hybrid_reranked | hybrid_reranked | nomic-embed-text | — | 0.000 | 0.717 | 0.452 |
| m9_10_hybrid | hybrid | nomic-embed-text | — | 0.022 | 0.652 | 0.249 |
| m9_10_dense | dense | nomic-embed-text | — | 0.000 | 0.326 | 0.119 |

## Three-model summary (all metrics)

| Embedding | Mode | doc_hit@1 | doc_hit@5 | doc_hit@10 | doc_hit@20 | doc_mrr@10 |
|---|---|---|---|---|---|---|
| nomic-embed-text | dense | 0.043 | 0.239 | 0.326 | 0.326 | 0.119 |
| nomic-embed-text | hybrid (1:1) | 0.087 | 0.478 | 0.652 | 0.652 | 0.249 |
| nomic-embed-text | hybrid_reranked | 0.370 | 0.565 | 0.717 | 0.804 | 0.452 |
| granite-embedding:278m | dense | 0.587 | 0.935 | 0.978 | 0.978 | 0.713 |
| granite-embedding:278m | hybrid (3:1) | 0.565 | 0.978 | 0.978 | 0.978 | 0.709 |
| granite-embedding:278m | hybrid_reranked (3:1) | 0.543 | 0.870 | 0.957 | 1.000 | 0.673 |
| **Octen-Embedding-0.6B** | **dense** | **0.696** | 0.957 | 0.978 | 0.978 | **0.797** |
| Octen-Embedding-0.6B | hybrid (10:1) | 0.630 | 0.957 | 0.957 | 0.957 | 0.756 |
| Octen-Embedding-0.6B | hybrid_reranked (10:1) | 0.587 | 0.826 | 0.957 | 1.000 | 0.695 |

## Octen-Embedding-0.6B: a third data point that breaks the pattern

Octen-Embedding-0.6B's dense-alone retrieval is the **strongest of the three
embedding models tested** — beating granite-embedding:278m despite being a
different (larger, ~0.6B vs 0.278B) model class, consistent with its
independently-measured RTEB leaderboard ranking.

Fusion weight calibration for this model followed the same sweep process as
granite (`docs/status.md` has the full method), but produced a **different
conclusion**: ratios 1:1 through 10:1 were tested, and `doc_hit@1` plateaued
at 0.630 from 5:1 onward (three ratios in a row, no further movement) while
`doc_mrr@10` gains kept shrinking (+0.015 at 5→7, +0.007 at 7→10) —
approaching an asymptote below dense-alone, not converging to it. **No
weight ratio tested fully recovers dense-alone performance for this model.**
This is the opposite of granite, where 3:1 nearly matched or exceeded
dense-alone on every metric. The calibrated default (`fusion_dense_weight:
10.0` in `configs/embedding/octen_0.6b_local.yaml`) reflects the best of
what was tested, but the honest takeaway is: **for Octen-0.6B on this
corpus, dense-alone retrieval is simply the better choice** — hybrid fusion
doesn't have a home here the way it does for nomic (weak dense baseline) or
granite (strong-but-improvable dense baseline).

`hybrid_reranked` shows the same mixed-result pattern already seen with
granite: `doc_hit@20` reaches a perfect 1.000 (best of any Octen variant on
that metric), but `doc_hit@1`/`doc_hit@5`/`doc_mrr@10` all regress relative
to plain `hybrid (10:1)`, let alone dense-alone. Reranking the reweighted
candidate pool doesn't recover what dense-alone already had.

**Takeaway across all three models:** the weaker the dense baseline, the
more hybrid/reranking help (nomic gains the most from both). The stronger
the dense baseline, the less headroom hybrid has to add value, and past some
point (Octen here) it has none at all in absolute terms — only relative
value among hybrid configurations (higher dense-weighting is always better
than lower), never beating the simpler dense-only mode.

## evidence_coverage_hit@k: fixing an artificially low evidence-span metric

`evidence_text_hit@k` (above) uses **symmetric Jaccard** between one chunk
and the full gold evidence span. FinanceBench's gold evidence for numeric
questions is usually a whole financial-statement table dumped as text
(median 1,389 chars, mean 1,502, vs. an 800-char chunk_size — 64% of
evidence spans are longer than one chunk), so no single chunk can contain
most of them, and any extra content in a chunk dilutes the Jaccard union
even when the chunk fully covers a shorter span. The result is an evidence
metric that's structurally capped well below what retrieval is actually
achieving.

`evidence_coverage_hit@k` (M9.11) uses **asymmetric coverage**
(`evidence tokens found / evidence tokens total`, with an exact-substring
shortcut — the same formula already used for `best_evidence_overlap`/
`mean_best_evidence_overlap`) over the **concatenated top-k chunks**, so it
isn't diluted by unrelated chunk content and gives credit when a table is
correctly assembled from several adjacent retrieved chunks. It's additive —
`evidence_text_hit@k` is unchanged and still reported.

**Spot-check (required before shipping this metric):** sampled all 25
questions (of 46) that newly "hit" under `evidence_coverage_hit@10` but not
`evidence_text_hit@10` on the Octen dense run, and checked whether each hit
holds up using *only* chunks from the correct gold document (i.e., with any
wrong-document chunks in the top-10 excluded).

- **23/25 (92%) are genuine** — coverage from the correct-document chunks
  alone already clears the 0.5 threshold. These are real cases of a
  financial statement table (e.g. AMD's or 3M's Cash Flow Statement, an
  Activision Blizzard Balance Sheet) being correctly assembled across 2+
  adjacent retrieved chunks from the right filing — exactly the fusion
  behavior the metric was designed to credit.
- **1/25 is a defensible edge case, not a clean pass**
  (`financebench_id_00941`, 3M's registered-securities cover-page table):
  correct-document chunks alone score 0.457 (just under threshold), but the
  hit is completed by chunks from *3M's own other filings* (2018/2022
  10-Ks) — which legitimately repeat the identical boilerplate registration
  table verbatim across filing periods. Not a cross-company false positive,
  but also not a clean single-document match.
- **1/25 (4%) is a confirmed false positive**
  (`financebench_id_01936`, Amcor's restructuring liability table): the
  correct document (`AMCOR_2023Q2_10Q`) has **zero** chunks in the top-10 at
  all (this question also fails `doc_hit@10`) — the coverage hit is entirely
  manufactured by `AMCOR_2023_10K` (a different filing, same company),
  whose restructuring-liability table happens to share the same
  labels/structure (`Employee Costs`, `Fixed Asset Related Costs`,
  `Liability balance`, etc.). This is a real, quantifiable failure mode:
  when a company reuses near-identical table structure/vocabulary across
  adjacent filings, coverage can cross threshold on the *wrong specific
  document*.

Net: the metric is a large, mostly-genuine improvement (~4x true-positive
rate) with one identified, low-frequency (~4% of newly-hit cases, ~2% of
the full sample) failure mode tied to companies reusing boilerplate table
structure across filing periods. Worth keeping in mind when reading
`evidence_coverage_hit@k` numbers as a ceiling rather than a guarantee that
retrieval found the *specific* correct filing.

| Model | Mode | txt@1 | txt@5 | txt@10 | txt@20 | cov@1 | cov@5 | cov@10 | cov@20 |
|---|---|---|---|---|---|---|---|---|---|
| nomic-embed-text | dense | 0.000 | 0.000 | 0.000 | 0.000 | 0.043 | 0.152 | 0.283 | 0.283 |
| nomic-embed-text | hybrid (1:1) | 0.000 | 0.022 | 0.022 | 0.022 | 0.043 | 0.239 | 0.326 | 0.326 |
| nomic-embed-text | hybrid_reranked | 0.000 | 0.000 | 0.000 | 0.000 | 0.043 | 0.196 | 0.304 | 0.587 |
| granite-embedding:278m | dense | 0.065 | 0.174 | 0.196 | 0.196 | 0.087 | 0.478 | 0.696 | 0.696 |
| granite-embedding:278m | hybrid (1:1, unweighted) | 0.022 | 0.109 | 0.174 | 0.174 | 0.043 | 0.283 | 0.609 | 0.609 |
| granite-embedding:278m | hybrid (3:1, calibrated) | 0.065 | 0.174 | 0.217 | 0.217 | 0.087 | 0.478 | 0.717 | 0.717 |
| granite-embedding:278m | hybrid_reranked (1:1, unweighted) | 0.000 | 0.000 | 0.043 | 0.152 | 0.043 | 0.261 | 0.565 | 0.848 |
| granite-embedding:278m | hybrid_reranked (3:1, calibrated) | 0.000 | 0.000 | 0.022 | 0.109 | 0.022 | 0.326 | 0.543 | 0.848 |
| **Octen-Embedding-0.6B** | **dense** | 0.065 | 0.109 | 0.217 | 0.217 | **0.174** | **0.543** | **0.761** | 0.761 |
| Octen-Embedding-0.6B | hybrid (10:1, calibrated) | 0.022 | 0.130 | 0.217 | 0.217 | 0.109 | 0.500 | **0.783** | **0.783** |
| Octen-Embedding-0.6B | hybrid_reranked (10:1, calibrated) | 0.000 | 0.022 | 0.043 | 0.130 | 0.065 | 0.283 | 0.522 | 0.804 |

Patterns that hold once the artificial chunk-size ceiling is lifted:

- **Reranking is the single biggest lever for evidence assembly at @20** —
  `hybrid_reranked`'s `cov@20` beats every other mode for both granite
  (0.848) and nomic (0.587 — its single best number anywhere in this
  table), even though reranking *hurts* `evidence_text_hit@k` for every
  model. The reranker is pulling in enough adjacent table fragments across
  20 slots to reassemble evidence that a stricter single-chunk metric can't
  see and that fewer slots (@10) can't yet fit.
- **Octen dense/hybrid still lead at @10** (0.761 / 0.783) — consistent
  with the doc-level story: Octen's dense retrieval is strong enough that
  it doesn't need reranking or many extra slots to assemble evidence.
- **nomic remains the weakest embedding on this metric too** — even its
  best mode (`hybrid_reranked` @20, 0.587) trails granite/Octen's weakest
  modes, confirming the coverage story tracks the doc_hit story rather than
  being a metric artifact of its own.
