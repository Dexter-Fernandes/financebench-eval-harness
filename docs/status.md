# Project Status

## Current State

- M9 complete on branch `M9` — PR not yet raised.
- 1002 tests pass, 2 skipped (Ollama integration tests).
- Full pipeline operational through hybrid retrieval (BM25 + RRF + cross-encoder reranking) and nDCG scoring.
- Live comparison run recorded on the real FinanceBench sample corpus (46 questions, `nomic-embed-text` and `granite-embedding:278m`, `BAAI/bge-reranker-base` reranker) — see `reports/retrieval_mode_comparison.md`.
- Post-M9.10 follow-up: `HybridRetriever`'s RRF fusion now supports `dense_weight`/`lexical_weight`, calibrated per embedding model (see "Fusion Weighting Investigation" below), and `LocalRerankerClient` auto-detects CUDA (falls back to CPU) — a real GPU cut reranking time on 46 questions from ~11 min to ~1.6 min.

## Current Engineering Milestone

**M9 complete: Hybrid Retrieval — BM25 + RRF + Cross-Encoder Reranking + nDCG**

```bash
# Retrieve with a specific mode: dense | lexical | hybrid | hybrid_reranked
# Note: --mode/--candidate-k/--fusion-k/--reranker-config only exist on this
# individual-flags path, not the --config pipeline-config path (M9.8 scope decision).
python -m financebench_eval retrieve \
  --index-dir data/indexes/financebench \
  --questions data/processed/financebench/examples.jsonl \
  --embedding-config configs/embedding/ollama_nomic.yaml \
  --chunks-path data/processed/financebench/chunks.jsonl \
  --output runs/<run_id>/retrieval_results.jsonl \
  --run-id <run_id> \
  --mode hybrid_reranked \
  --reranker-config configs/reranker/bge_base_local.yaml

# Score (adds doc/page/evidence_text nDCG@k alongside hit@k, MRR@k)
python -m financebench_eval eval-retrieval --config configs/retrieval.yaml --run-id <run_id>

# Compare modes side by side
python -m financebench_eval compare-runs \
  --runs runs/<dense_id> runs/<hybrid_id> runs/<hybrid_reranked_id> \
  --metric evidence_text_ndcg@10 \
  --output reports/retrieval_mode_comparison.md
```

Output layout (additive to the existing retrieval schema):
```
runs/<run_id>/
  retrieval_results.jsonl       ← rows gain optional dense_rank/lexical_rank/rerank_score
  retrieval_run_metadata.json  ← gains retrieval_mode, fusion {k, candidate_k}, reranker {provider, model_name, adapter_path}
  retrieval_scores.jsonl       ← per-question rows gain *_ndcg@k keys
  retrieval_summary.json       ← aggregate gains *_ndcg@k keys

reports/
  retrieval_mode_comparison.md  ← committed dense vs hybrid vs hybrid_reranked comparison
```

New modules: `lexical_index.py` (BM25), `fusion.py` (reciprocal rank fusion),
`reranker.py` (mock/HTTP/local cross-encoder clients), `hybrid_retriever.py`
(Retriever compositions: DenseRetriever, LexicalRetriever, HybridRetriever,
RerankedRetriever). `retriever.run_retrieval` is now a thin wrapper over
`DenseRetriever` + `run_retrieval_with_retriever` — behavior-preserving.

`rag_hybrid` and `rag_reranked` RAG modes are wired end-to-end (config in
`configs/evaluation/rag_modes.yaml`, reusing the `rag_dense_v2` prompt).

## Real-data comparison (46 FinanceBench sample questions)

### nomic-embed-text (weak dense baseline)

| Mode | doc_hit@10 | MRR@10 (doc) | evidence_text_ndcg@10 |
|---|---|---|---|
| dense | 0.326 | 0.119 | 0.000 |
| hybrid (BM25 + RRF, 1:1 — calibrated default) | 0.652 | 0.249 | 0.011 |
| hybrid_reranked (bge-reranker-base) | 0.717 | 0.452 | 0.000 |

Hybrid retrieval roughly doubles doc-level hit rate over dense-only on this
sample; reranking further concentrates the correct document at a higher rank
(MRR nearly quadruples over dense).

### granite-embedding:278m (strong dense baseline)

| Mode | doc_hit@1 | doc_hit@10 | MRR@10 (doc) | evidence_text_ndcg@10 |
|---|---|---|---|---|
| dense | 0.587 | 0.978 | 0.713 | 0.131 |
| hybrid, unweighted 1:1 (original M9.10 baseline — no longer the default) | 0.457 | 0.978 | 0.620 | 0.079 |
| hybrid, 3:1 dense:lexical (calibrated default) | 0.565 | 0.978 | 0.709 | — |

With a strong dense baseline, unweighted RRF actively *hurt* — BM25's noisy
top picks (financial filings are full of repeated boilerplate: disclosure
headers, "Table of Contents," etc.) got an equal vote against a confident,
correct dense hit. See "Fusion Weighting Investigation" below for the full
sweep that found and fixed this. `evidence_text_ndcg@10` stays low across all
modes for both embedding models — exact evidence-text matching is rare in
this small sample, consistent with the known FinanceBench table-extraction
gap already tracked by this project's failure-label analysis.

Full tables with all metrics in `reports/retrieval_mode_comparison.md`.

## Fusion Weighting Investigation (post-M9.10)

**Trigger:** the M9.10 real-data comparison showed `hybrid` retrieval helping
nomic (doubling `doc_hit@10`) but *hurting* granite (`doc_hit@1` dropped from
0.587 dense-alone to 0.457 hybrid). Root cause: `reciprocal_rank_fusion` is
pure rank-based — it discards score magnitude entirely, so a confident
dense-rank-1 hit gets exactly the same vote as a marginal one, and an
equally-weighted noisy BM25 top pick can outrank it.

**Experiment 1 — sweep `fusion_k` (the RRF decay constant) down from the
default 60 to 20 and 10, on both embeddings, `hybrid` and `hybrid_reranked`
modes. Negative result — this was not the fix:**

| Embedding | Mode | fusion_k=10 | fusion_k=20 | fusion_k=60 (baseline) |
|---|---|---|---|---|
| granite | hybrid doc_hit@1 | 0.457 | 0.435 | **0.457** |
| granite | hybrid doc_mrr@10 | 0.614 | 0.605 | **0.620** |
| nomic | hybrid doc_mrr@10 | 0.259 | 0.258 | 0.249 |

`fusion_k=60` (the default) was tied-best or best for granite; lowering it
did not close the gap. For `hybrid_reranked`, changing `fusion_k` had **zero**
effect on any metric for either embedding — the cross-encoder reranker fully
overrides the RRF-fused order, so the decay constant only matters for plain
`hybrid` mode.

**Experiment 2 — sweep dense:lexical weight ratios {1:1, 2:1, 3:1, 5:1} via
the newly-added `dense_weight`/`lexical_weight` params (threading into
`reciprocal_rank_fusion`'s pre-existing but previously-unused `weights`
param), `hybrid` mode only. Positive result, opposite optima per model:**

| Embedding | Ratio | doc_hit@1 | doc_hit@5 | doc_mrr@10 |
|---|---|---|---|---|
| granite | 1:1 | 0.457 | 0.804 | 0.620 |
| granite | 2:1 | 0.543 | 0.935 | 0.695 |
| granite | **3:1** | **0.565** | **0.978** | **0.709** |
| granite | 5:1 | 0.543 | 0.978 | 0.698 |
| nomic | **1:1** | 0.087 | **0.478** | **0.249** |
| nomic | 2:1 | 0.087 | 0.370 | 0.207 |
| nomic | 3:1 | 0.087 | 0.326 | 0.184 |
| nomic | 5:1 | 0.109 | 0.304 | 0.185 |

For granite, upweighting dense monotonically helps through 3:1 (`doc_hit@5`
at 3:1 even *exceeds* dense-alone: 0.978 vs 0.935), then 5:1 overshoots
slightly. For nomic, every increase in `dense_weight` monotonically *hurts*
— nomic's dense signal is weak enough that lexical's full-strength vote is
exactly what's compensating. **No single global ratio works for both models**
— 3:1 would regress nomic's `doc_mrr@10` by ~26% (0.249 → 0.184).

**Fix shipped:** `EmbeddingConfig` gained `fusion_dense_weight`/
`fusion_lexical_weight` (default 1.0/1.0, backward compatible).
`configs/embedding/ollama_nomic.yaml` sets 1:1, `configs/embedding/ollama_granite.yaml`
sets 3:1. The `retrieve` CLI's `--dense-weight`/`--lexical-weight` flags now
default to `None` and fall back to the embedding config's calibrated values
when omitted — so `retrieve --mode hybrid` uses the right ratio automatically
per embedding model with zero extra flags, verified end-to-end (a bare
`--mode hybrid` run with the granite config reproduces the 3:1 sweep numbers
exactly). Explicit `--dense-weight`/`--lexical-weight` still override the
config when passed.

**Known follow-up:** `hybrid_reranked` shares `HybridRetriever`'s
construction, so it now also picks up calibrated weights automatically. We
confirmed `fusion_k` has zero effect on `hybrid_reranked`'s final metrics,
but we have **not** verified whether `dense_weight`/`lexical_weight` behaves
the same way — weight changes are large enough to plausibly shift which
chunks land in the reranker's candidate pool, unlike `fusion_k`. The granite
`hybrid_reranked` row in `reports/retrieval_mode_comparison.md` still
reflects the pre-calibration 1:1 internal fusion and hasn't been re-run.

## Next Steps

1. Raise M9 PR and merge to main.
2. M10 (fine-tuned reranker): train a LoRA adapter for `bge-reranker-base` and
   load it via `RerankerConfig.adapter_path` — the seam is already in place.
3. M13 (Snowflake embedding comparison) can reuse the same hybrid/reranked
   retrieval modes against a different base embedding model — will need its
   own calibrated `fusion_dense_weight`/`fusion_lexical_weight`, following the
   same sweep process documented above.
4. Re-run `hybrid_reranked` for granite with the 3:1 calibrated weights and
   confirm whether it changes the candidate pool the reranker sees (see
   "Known follow-up" above — `fusion_k` was confirmed to have zero effect,
   but weight changes haven't been checked).
5. Known gap (tracked separately, not fixed in M9): `rag_predictions.jsonl`
   writes `context_chunk_ids` but `analyze_grounding` reads `retrieved_chunks`,
   so citation/context-sufficiency checks silently see empty context on real
   `run-rag` output. Needs a fix before grounding analysis numbers on real runs
   can be trusted.

---

## How to Run the Full Pipeline

```bash
# Prerequisites
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Chunk + index (once)
python -m financebench_eval chunk-documents --config configs/retrieval.yaml
python -m financebench_eval build-index     --config configs/retrieval.yaml --progress

# Retrieve (mode defaults to dense; see M9 example above for hybrid/hybrid_reranked)
python -m financebench_eval retrieve \
  --config configs/retrieval.yaml --run-id run_005

# RAG generation (update retrieval_results_path in config first)
python -m financebench_eval run-rag \
  --config configs/evaluation/rag_dense_local.yaml

# Score + report
python -m financebench_eval score-rag \
  --config configs/evaluation/rag_score_local.yaml \
  --run-dir runs/<rag_run_id>

python -m financebench_eval join-metrics \
  --retrieval-scores runs/<retrieval_run_id>/retrieval_scores.jsonl \
  --answer-scores    runs/<rag_run_id>/rag_combined_scores.jsonl \
  --output-dir       runs/<rag_run_id>/joined --k 5

python -m financebench_eval report-rag \
  --joined-dir        runs/<rag_run_id>/joined \
  --rag-run-dir       runs/<rag_run_id> \
  --retrieval-summary runs/<retrieval_run_id>/retrieval_summary.json \
  --run-id            <rag_run_id>
```

## Current Boundaries

- Public FinanceBench sample data only.
- Local datasets, PDFs, vector stores, model caches, secrets, and run artefacts stay out of version control.
- `reports/*.md` gitignored by default; `reports/retrieval_mode_comparison.md` is an intentional exception — a small, curated, reviewable comparison table committed per the AGENTS.md rule allowing curated documentation.
- `rerank` is an optional dependency group (`pip install .[rerank]`) — torch/transformers/peft, only needed for the local cross-encoder reranker provider.
- Do not publish benchmark claims until backed by reproducible runs with committed configuration.
