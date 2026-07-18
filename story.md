# Project Story

## Current state

- M9 implementation complete on branch `M9` (many commits ahead of main, PR not yet raised).
- 1002 tests pass, 2 skipped (Ollama integration tests).
- `retrieve --mode {dense,lexical,hybrid,hybrid_reranked}` fully operational; `eval-retrieval` now reports nDCG@k alongside hit@k/MRR@k.
- Live comparison runs recorded on the real FinanceBench sample (46 questions) for **three** embedding models — `nomic-embed-text`, `granite-embedding:278m` (both Ollama), and `Octen-Embedding-0.6B` (local HuggingFace `sentence-transformers`, GPU) — with the real `BAAI/bge-reranker-base` reranker, also GPU-accelerated. See `reports/retrieval_mode_comparison.md`.
- Post-M9.10 follow-up work: diagnosed and fixed a real regression where hybrid retrieval was *hurting* a strong embedding model (see "Fusion weighting investigation" below), then extended the whole pipeline to local HuggingFace embedding models and found the calibration story has a third shape: for the strongest embedding model tested, hybrid never beats dense-alone at any weight (see "Local HuggingFace embedding models" below).

## Latest milestone

**M9: Hybrid Retrieval — BM25 + RRF + Cross-Encoder Reranking + nDCG** — all 10 sub-milestones complete, built test-first (one tracer-bullet-then-loop TDD cycle per sub-milestone, one commit each), followed by a post-M9.10 investigation and fix once the real comparison surfaced a problem.

Key deliverables (M9.1–M9.10):

- `retrieval_scoring.py` — `ndcg_at_k()`, `score_ndcg()`, `summarize_ndcg()` (M9.1); wired into `eval_retrieval.score_retrieval_run` output, the rich retrieval report, and `compare-runs --metric` (M9.2), all additive to the existing schema.
- `lexical_index.py` — `BM25Config` + `BM25Index` (Okapi BM25, Lucene-style non-negative IDF floor), built in-memory from `chunk_metadata.jsonl` (M9.3).
- `fusion.py` — `reciprocal_rank_fusion()` (M9.4).
- `reranker.py` — `RerankerConfig`/`RerankerClient` Protocol mirroring `EmbeddingClient`; `MockRerankerClient`, `HTTPRerankerClient` (M9.5); `LocalRerankerClient` running a real cross-encoder with lazy torch/transformers import and optional LoRA adapter loading via peft (M9.6, for M10).
- `hybrid_retriever.py` — `Retriever` Protocol + `DenseRetriever`/`LexicalRetriever`/`HybridRetriever`/`RerankedRetriever` compositions, `run_retrieval_with_retriever()` (M9.7). `retriever.run_retrieval` refactored into a thin wrapper over `DenseRetriever` + `run_retrieval_with_retriever` — behavior-preserving, existing tests pass unchanged.
- CLI: `retrieve --mode {dense,lexical,hybrid,hybrid_reranked} --candidate-k --fusion-k --reranker-config` (M9.8). Scope note: `PipelineConfig`/`RetrievalConfig` YAML schema changes for the `--config`-driven retrieve path were deferred — not covered by M9.8's acceptance criteria.
- `configs/evaluation/rag_modes.yaml` — added `rag_hybrid`/`rag_reranked` (context_source: retrieved_chunks, reusing the `rag_dense_v2` prompt); real end-to-end mock-LLM + mock-reranker pipeline tests replace the old placeholder enum test (M9.9).
- `pyproject.toml` — added the `rerank` optional dependency group (`torch`, `transformers`, `peft`) that M9.6's design called for but was missed until M9.10 exposed the gap.
- `reports/retrieval_mode_comparison.md` — real dense vs. hybrid vs. hybrid_reranked comparison (M9.10).

Post-M9.10 follow-up work:

- `LocalRerankerClient` now auto-detects CUDA and moves the model + input tensors to GPU, falling back to CPU (`_select_device()`, injectable/stubbable — tests never need real GPU hardware). Real speedup on this machine: bge-reranker-base over 46 questions × 50 candidates went from **~11 min (CPU) to ~1.6 min (GPU)** — verified numerically identical output on both.
- `HybridRetriever` gained `dense_weight`/`lexical_weight`, threaded into `reciprocal_rank_fusion`'s pre-existing (previously unused) `weights` param, plus CLI `--dense-weight`/`--lexical-weight` flags.
- `EmbeddingConfig` gained `fusion_dense_weight`/`fusion_lexical_weight` (default 1.0/1.0). `retrieve`'s weight flags now default to `None` and fall back to the embedding config's calibrated values when omitted — hybrid mode picks the right ratio automatically per embedding model.

## Fusion weighting investigation

**Trigger:** M9.10's real-data comparison showed `hybrid` helping nomic (roughly doubling `doc_hit@10`) but *hurting* granite (`doc_hit@1` dropped from 0.587 dense-alone to 0.457 hybrid). Root cause: `reciprocal_rank_fusion` is pure rank-based and discards score magnitude — a confident dense-rank-1 hit gets exactly the same vote as a marginal one, so an equally-weighted noisy BM25 top pick (financial filings are full of repeated boilerplate) can outrank it.

**`fusion_k` sweep (10, 20, vs. default 60) — negative result.** The default (60) was tied-best or best for granite `hybrid`; lowering it didn't close the gap. For `hybrid_reranked`, `fusion_k` had **zero** effect on any metric for either embedding — the cross-encoder reranker fully overrides the RRF-fused order.

**Weight ratio sweep (dense:lexical = 1:1, 2:1, 3:1, 5:1) — positive result, opposite optima per model.** Granite improves monotonically through 3:1 (`doc_hit@1` 0.457→0.565, `doc_mrr@10` 0.620→0.709, `doc_hit@5` at 3:1 even *exceeds* dense-alone: 0.978 vs 0.935), then 5:1 overshoots slightly. Nomic gets monotonically *worse* as `dense_weight` increases (`doc_mrr@10` 0.249→0.184 by 3:1) — its dense signal is weak enough that lexical's full-strength vote is what's helping. **No single global ratio works for both** — this has to be per-embedding-model.

**Fix shipped and verified end-to-end:** calibrated defaults now live in `configs/embedding/ollama_nomic.yaml` (1:1) and `configs/embedding/ollama_granite.yaml` (3:1); `retrieve --mode hybrid` picks the right ratio automatically with zero extra flags — confirmed a bare run against the granite config reproduces the 3:1 sweep numbers exactly (`doc_hit@1`=0.565, `doc_mrr@10`=0.709, both bit-for-bit matches).

**Follow-up resolved:** re-ran granite `hybrid_reranked` with the 3:1 calibrated weights (applied automatically via the embedding config, confirmed in `retrieval_run_metadata.json`). Unlike `fusion_k` (confirmed zero effect), `dense_weight`/`lexical_weight` **does** change `hybrid_reranked`'s final results — but the effect is mixed, not a clean win:

| Variant | doc_hit@1 | doc_hit@5 | doc_hit@10 | doc_hit@20 | doc_mrr@10 |
|---|---|---|---|---|---|
| dense-alone | 0.587 | 0.935 | 0.978 | 0.978 | 0.713 |
| hybrid_reranked, unweighted 1:1 | 0.522 | 0.913 | 0.978 | 0.978 | 0.657 |
| hybrid_reranked, 3:1 (calibrated) | 0.543 | 0.870 | 0.957 | **1.000** | 0.673 |

`doc_hit@1`/`doc_mrr@10` nudge up, `doc_hit@20` reaches a perfect 1.000 (exceeding dense-alone), but `doc_hit@5`/`doc_hit@10` get *worse*. Reweighting fusion does change which chunks reach the reranker's candidate pool as hypothesized, but the reranker's behavior on the reweighted pool doesn't uniformly track plain `hybrid`'s clean 3:1 win — worth a separate weight sweep for `hybrid_reranked` specifically rather than assuming `hybrid`'s tuned ratio transfers.

## Local HuggingFace embedding models

Added a third embedding model, `Octen/Octen-Embedding-0.6B`, this time local
HuggingFace rather than Ollama-served — to test whether the pipeline could
support that at all. Full narrative and VRAM math in `docs/status.md`'s
"Local HuggingFace Embedding Models" section; short version:

- **No new code needed.** `embedding.py`'s `SentenceTransformersEmbeddingClient`
  was already fully wired (provider, config loading, CLI) but never
  exercised against a real model. Added a `local-embed` optional dependency
  group (`sentence-transformers`, `bitsandbytes`) to `pyproject.toml`.
- **Original target (`Octen-Embedding-4B-INT8`) failed on hardware, not
  code**: `cuBLAS API failed with status 15` — the GTX 1660 Ti reports CUDA
  compute capability 7.5 but, unlike RTX 20-series cards with the same
  number, lacks real INT8 Tensor Cores, which `bitsandbytes`' INT8 kernels
  require. Fixed hardware limitation, not something batch size or retries
  could fix. Switched to the non-quantized `Octen-Embedding-0.6B` instead.
- **Batch size took three tries**: 32 → OOM (actual footprint ~4.9GB, far
  above a naive fp32 estimate), 4 → safe but ~9% CPU utilization (kernel
  launch overhead dominated, no completion after ~55 min), 24 → stable at
  the edge (~600MB free) for the full 46-minute build. A corrected two-point
  linear VRAM model (`1.73GB + batch_size × 0.142GB`, and correcting the
  bf16-not-fp32 weight-dtype assumption along the way) predicted the
  `batch_size=32` OOM and a ~27 theoretical ceiling, consistent with what
  was observed.
- **Weight sweep found a third shape.** Octen's dense-alone is the
  *strongest* baseline of all three models (`doc_hit@1`=0.696, beating even
  granite) — and unlike granite, **no ratio from 1:1 through 10:1 fully
  recovers dense-alone performance**. `doc_hit@1` plateaus at 0.630 from 5:1
  onward; `doc_mrr@10` gains shrink toward an asymptote below dense-alone
  (0.797), never reaching it. Calibrated at 10:1 (best tested), with an
  explicit note in `configs/embedding/octen_0.6b_local.yaml` that dense-alone
  is simply the better choice for this model on this corpus.
- **Queued next:** `nvidia/Nemotron-3-Embed-1B-BF16` at `batch_size=6`
  (extrapolated VRAM math in `docs/status.md`) — not started yet.

## Real-data comparison (46 FinanceBench sample questions)

**nomic-embed-text** (weakest dense baseline — hybrid/reranking help the most):

| Mode | doc_hit@10 | doc_mrr@10 | evidence_text_ndcg@10 |
|---|---|---|---|
| dense | 0.326 | 0.119 | 0.000 |
| hybrid (1:1, calibrated default) | 0.652 | 0.249 | 0.011 |
| hybrid_reranked | 0.717 | 0.452 | 0.000 |

**granite-embedding:278m** (strong dense baseline — hybrid nearly matches it at 3:1):

| Mode | doc_hit@1 | doc_hit@10 | doc_mrr@10 |
|---|---|---|---|
| dense | 0.587 | 0.978 | 0.713 |
| hybrid, unweighted 1:1 (original M9.10 baseline, no longer the default) | 0.457 | 0.978 | 0.620 |
| hybrid, 3:1 (calibrated default) | 0.565 | 0.978 | 0.709 |

**Octen-Embedding-0.6B** (strongest dense baseline — hybrid never catches up):

| Mode | doc_hit@1 | doc_hit@10 | doc_mrr@10 |
|---|---|---|---|
| dense | **0.696** | 0.978 | **0.797** |
| hybrid, 10:1 (calibrated default, best tested) | 0.630 | 0.957 | 0.756 |
| hybrid_reranked (10:1) | 0.587 | 0.957 | 0.695 |

`evidence_text_ndcg@10` stays low across all modes and all three embeddings
on this small sample — consistent with the project's already-known
FinanceBench table-extraction gap, not a new finding.

**Pattern across all three:** the weaker the dense baseline, the more
hybrid/reranking help. The stronger it gets, the less headroom hybrid has —
past some point (Octen), it adds none at all in absolute terms.

## Next step

1. **Raise and merge M9 PR** — branch is clean, all tests pass.
2. **Run `nvidia/Nemotron-3-Embed-1B-BF16` at `batch_size=6`** — queued, not started. Full extrapolated VRAM math in `docs/status.md`.
3. **Consider a separate weight sweep for `hybrid_reranked`** — 3:1 (granite) and 10:1 (Octen), both tuned for plain `hybrid`, give `hybrid_reranked` mixed results (small `doc_mrr@10` wins, `doc_hit@5`/`@10` regressions) for both models tested so far; may need its own calibrated ratio rather than inheriting `hybrid`'s.
4. **M10 (fine-tuned reranker)** — the seam is ready: train a LoRA adapter for `bge-reranker-base`, load it via `RerankerConfig.adapter_path` / `LocalRerankerClient`, compare against the pretrained baseline recorded here. GPU support means iteration on this will be much faster than the original CPU-only path.
5. **M13 (Snowflake embedding comparison)** — can reuse the same hybrid/reranked retrieval modes, but will need its own calibrated `fusion_dense_weight`/`fusion_lexical_weight` via the same sweep process, now proven across three very different models (Ollama-served and local HuggingFace).
6. **Fix the grounding-analysis context gap** (spawned as a separate task, not fixed in M9): `run_rag_from_config` writes `context_chunk_ids` (list of IDs) into `rag_predictions.jsonl`, but `analyze_grounding` reads `retrieved_chunks` (list of {chunk_id, text} dicts) — the key names don't match, so citation/context-sufficiency checks silently see empty context on every real `run-rag` → `analyze-grounding` run. Existing tests never caught this because they hand-build `rag_predictions.jsonl` fixtures with `retrieved_chunks` already present.

## Known risks or blockers

- The grounding-analysis field-name gap above predates M9 but was only noticed while wiring M9.9's end-to-end pipeline test; it means no `analyze-grounding` run on real (non-fixture) data has ever actually seen retrieved context.
- `reports/retrieval_mode_comparison.md` is committed as an intentional exception to the `reports/*.md` gitignore rule (curated, small, reviewable — per AGENTS.md's allowance for intentionally curated documentation).
- `evidence_text_ndcg@10` is a very strict, low-frequency metric on this 46-question sample (most runs saw 0 exact evidence-text matches); don't over-read any dense-vs-hybrid comparison on that single metric — `doc_hit@k` and `doc_mrr@10` are the more informative columns here.
- The granite `hybrid_reranked` row in `reports/retrieval_mode_comparison.md` reflects the 3:1 calibrated re-run (see "Follow-up resolved" above), not the original M9.10 1:1 baseline.
- `compare-runs`' table only surfaces `evidence_hit@10`/`doc_hit@10`/`mrr@10`/`accuracy` columns; for retrieval-only runs it now falls back to `retrieval_run_metadata.json` for `mode`/`model` and to `doc_mrr@10` for the `mrr@10` column (fixed post-M9.10), and best-run selection tie-breaks on `doc_hit@10` then `doc_mrr@10` after the requested `--metric`.
- This GPU (GTX 1660 Ti) cannot run `bitsandbytes` INT8-quantized models (no real Tensor Cores despite CC 7.5) — any `-INT8` HuggingFace model should be assumed incompatible unless verified otherwise.
- The `Octen-Embedding-0.6B` index build required real batch-size trial and error (32 OOM'd, 4 was too slow, 24 worked) — the naive VRAM estimate from param count alone was wrong twice (wrong dtype assumption). See `docs/status.md` for the corrected linear model.

## Safe next commands

```bash
# Confirm tests still pass
pytest -q

# Raise M9 PR
git push origin M9
gh pr create --title "feat(M9): hybrid retrieval (BM25 + RRF + reranking) and nDCG" --body "..."

# Re-run hybrid mode with calibrated defaults (no --dense-weight/--lexical-weight needed)
python -m financebench_eval retrieve \
  --index-dir "data/indexes/financebench/granite-embedding:278m" \
  --questions data/processed/financebench/examples.jsonl \
  --embedding-config configs/embedding/ollama_granite.yaml \
  --chunks-path data/processed/financebench/chunks.jsonl \
  --output runs/<run_id>/retrieval_results.jsonl \
  --run-id <run_id> --mode hybrid --top-k 10

# hybrid_reranked with calibrated 3:1 weights (already re-run once — see
# "Follow-up resolved" above; mixed result, doc_mrr@10 up, doc_hit@5/@10 down)
python -m financebench_eval retrieve \
  --index-dir "data/indexes/financebench/granite-embedding:278m" \
  --questions data/processed/financebench/examples.jsonl \
  --embedding-config configs/embedding/ollama_granite.yaml \
  --chunks-path data/processed/financebench/chunks.jsonl \
  --output runs/<run_id>/retrieval_results.jsonl \
  --run-id <run_id> --mode hybrid_reranked \
  --reranker-config configs/reranker/bge_base_local.yaml --top-k 10

# Score any granite run (configs/retrieval_granite.yaml points index_dir/embedding
# at the granite index so leaderboard/report metadata is accurate, unlike the
# nomic-tagged default configs/retrieval.yaml)
python -m financebench_eval eval-retrieval --config configs/retrieval_granite.yaml --run-id <run_id>

# Octen-Embedding-0.6B (local HuggingFace, GPU) — dense/hybrid modes only
# need the embedding client, safe/fast; hybrid_reranked also loads
# bge-reranker-base in the same process (confirmed no VRAM contention issue)
python -m financebench_eval retrieve \
  --index-dir "data/indexes/financebench/octen-embedding-0.6b" \
  --questions data/processed/financebench/examples.jsonl \
  --embedding-config configs/embedding/octen_0.6b_local.yaml \
  --chunks-path data/processed/financebench/chunks.jsonl \
  --output runs/<run_id>/retrieval_results.jsonl \
  --run-id <run_id> --mode hybrid --top-k 10
python -m financebench_eval eval-retrieval --config configs/retrieval_octen.yaml --run-id <run_id>

# Rebuild the Octen index from scratch if needed (~46 min at batch_size=24 on
# this GPU; use configs/retrieval.yaml for chunking settings, NOT the
# standalone configs/retrieval/recursive_text_800.yaml which is missing
# evidence_overlap_threshold and will error if passed to build-index directly)
python -m financebench_eval build-index \
  --pages data/processed/financebench/pages.jsonl \
  --retrieval-config configs/retrieval.yaml \
  --embedding-config configs/embedding/octen_0.6b_local.yaml \
  --index-dir "data/indexes/financebench/octen-embedding-0.6b" \
  --progress
```
