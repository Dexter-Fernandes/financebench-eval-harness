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
