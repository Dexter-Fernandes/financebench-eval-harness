# Run Comparison

Sorted by: `doc_mrr@10`

**Best run:** `m9_10_granite_dense`

| run_id | mode | model | accuracy | evidence_hit@10 | doc_hit@10 | mrr@10 |
| --- | --- | --- | --- | --- | --- | --- |
| m9_10_granite_dense ★ | dense | granite-embedding:278m | — | 0.196 | 0.978 | 0.713 |
| dw3lw1_granite_hybrid | hybrid | granite-embedding:278m | — | 0.217 | 0.978 | 0.709 |
| granite_hybrid_reranked_3to1 | hybrid_reranked | granite-embedding:278m | — | 0.022 | 0.957 | 0.673 |
| m9_10_hybrid_reranked | hybrid_reranked | nomic-embed-text | — | 0.000 | 0.717 | 0.452 |
| m9_10_hybrid | hybrid | nomic-embed-text | — | 0.022 | 0.652 | 0.249 |
| m9_10_dense | dense | nomic-embed-text | — | 0.000 | 0.326 | 0.119 |

## Notes on the `hybrid` rows above

`hybrid` retrieval fuses dense + BM25 rankings via reciprocal rank fusion
(RRF), weighted by `dense_weight`/`lexical_weight`. These weights are now
**calibrated per embedding model** and applied automatically by the
`retrieve` CLI (`configs/embedding/ollama_*.yaml`'s `fusion_dense_weight`/
`fusion_lexical_weight`; override with `--dense-weight`/`--lexical-weight`):

- **nomic-embed-text: 1:1** (dense:lexical). nomic's dense signal alone is
  weak (`doc_hit@1`=0.043); lexical's full-strength vote is what makes
  hybrid help at all. `m9_10_hybrid` above already reflects this default.
- **granite-embedding:278m: 3:1**. granite's dense signal alone is already
  strong (`doc_hit@1`=0.587); equal weighting let BM25 noise outrank
  confident dense hits (`doc_hit@1` dropped to 0.457 at 1:1 — see
  `docs/status.md` for the full sweep). `dw3lw1_granite_hybrid` above is
  the 3:1-calibrated run — it nearly matches dense-alone on `doc_mrr@10`
  (0.709 vs 0.713) and slightly exceeds it on `doc_hit@5` (0.978 vs 0.935,
  not shown in this table — see `docs/status.md`).

## `hybrid_reranked` with calibrated weights — a mixed result

`granite_hybrid_reranked_3to1` above is the re-run with the 3:1 calibrated
weights (confirmed applied automatically via `retrieval_run_metadata.json`).
Unlike `fusion_k` (confirmed to have *zero* effect on `hybrid_reranked`'s
final metrics, since the cross-encoder reranker fully overrides the fused
order), `dense_weight`/`lexical_weight` **does** change the final results —
but not cleanly:

| Variant | doc_hit@1 | doc_hit@5 | doc_hit@10 | doc_hit@20 | doc_mrr@10 |
|---|---|---|---|---|---|
| dense-alone | 0.587 | 0.935 | 0.978 | 0.978 | 0.713 |
| hybrid_reranked, unweighted 1:1 (original M9.10 baseline) | 0.522 | 0.913 | 0.978 | 0.978 | 0.657 |
| hybrid_reranked, 3:1 (calibrated, this table's row) | 0.543 | 0.870 | 0.957 | **1.000** | 0.673 |

`doc_hit@1`/`doc_mrr@10` nudge up and `doc_hit@20` reaches a perfect 1.000
(now exceeding dense-alone), but `doc_hit@5`/`doc_hit@10` both get *worse*.
Reweighting fusion does shift which chunks land in the reranker's top-50
candidate pool as hypothesized, but the reranker's own behavior on that
reweighted pool doesn't uniformly track plain `hybrid`'s clean 3:1 win. The
3:1 ratio was tuned for plain `hybrid`, not for what happens once a reranker
sits downstream of it — `hybrid_reranked` likely needs its own calibration
sweep rather than inheriting `hybrid`'s ratio.
