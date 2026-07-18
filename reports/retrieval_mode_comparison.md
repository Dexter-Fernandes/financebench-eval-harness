# Run Comparison

Sorted by: `doc_mrr@10`

**Best run:** `m9_10_granite_dense`

| run_id | mode | model | accuracy | evidence_hit@10 | doc_hit@10 | mrr@10 |
| --- | --- | --- | --- | --- | --- | --- |
| m9_10_granite_dense ★ | dense | granite-embedding:278m | — | 0.196 | 0.978 | 0.713 |
| dw3lw1_granite_hybrid | hybrid | granite-embedding:278m | — | 0.217 | 0.978 | 0.709 |
| m9_10_granite_hybrid_reranked | hybrid_reranked | granite-embedding:278m | — | 0.043 | 0.978 | 0.657 |
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

**Known gap:** `hybrid_reranked` shares the same `HybridRetriever`
construction, so it also picks up these calibrated weights automatically
going forward. `m9_10_granite_hybrid_reranked` above predates the
calibration work and still reflects unweighted (1:1) internal fusion —
it has not been re-run with the 3:1 default. Separately, we confirmed
`fusion_k` (the RRF decay constant) has *zero* effect on `hybrid_reranked`'s
final metrics (the cross-encoder reranker fully overrides the fused order),
but we have not verified whether `dense_weight`/`lexical_weight` affects
`hybrid_reranked` the same way — weight changes are large enough to
plausibly shift *which* chunks land in the top-`candidate_k` pool the
reranker sees, unlike `fusion_k`. Worth checking in a follow-up.
