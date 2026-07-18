# Project Status

## Current State

- M9 complete on branch `M9` — PR not yet raised.
- 1007 tests pass, 2 skipped (Ollama integration tests).
- Full pipeline operational through hybrid retrieval (BM25 + RRF + cross-encoder reranking) and nDCG scoring.
- Live comparison run recorded on the real FinanceBench sample corpus (46 questions) across **three** embedding models — `nomic-embed-text`, `granite-embedding:278m` (both via Ollama), and `Octen-Embedding-0.6B` (local HuggingFace `sentence-transformers`) — with the real `BAAI/bge-reranker-base` reranker. See `reports/retrieval_mode_comparison.md`.
- Post-M9.10 follow-up: `HybridRetriever`'s RRF fusion now supports `dense_weight`/`lexical_weight`, calibrated per embedding model (see "Fusion Weighting Investigation" below), and `LocalRerankerClient` auto-detects CUDA (falls back to CPU) — a real GPU cut reranking time on 46 questions from ~11 min to ~1.6 min.
- Confirmed the existing `SentenceTransformersEmbeddingClient` provider (already in `embedding.py`, previously unused against a real model) supports local HuggingFace embedding models with zero code changes — see "Local HuggingFace Embedding Models" below for what it took to actually get one running on this GPU.
- **Default embedding model locked in: `Octen-Embedding-0.6B`, dense mode.** It's the strongest of the three models evaluated and gets no benefit from hybrid fusion at any tested weight ratio, so dense is both its best mode and now the pipeline default. `configs/retrieval.yaml` and the `build-index`/`embed-question`/`retrieve` CLI flag defaults all point at it (`data/indexes/financebench/octen-embedding-0.6b`, `configs/embedding/octen_0.6b_local.yaml`). `nomic-embed-text`/`granite-embedding:278m` remain available via `configs/retrieval_nomic.yaml`/`configs/retrieval_granite.yaml` for comparison runs.

## Current Engineering Milestone

**M9 complete: Hybrid Retrieval — BM25 + RRF + Cross-Encoder Reranking + nDCG**

```bash
# Retrieve with a specific mode: dense | lexical | hybrid | hybrid_reranked
# Note: --mode/--candidate-k/--fusion-k/--reranker-config only exist on this
# individual-flags path, not the --config pipeline-config path (M9.8 scope decision).
# Example below targets nomic explicitly (hybrid_reranked helps it the most,
# per the pattern below) — the pipeline default (no flags) is Octen dense-alone.
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

### Octen-Embedding-0.6B (strongest dense baseline — hybrid doesn't help here)

| Mode | doc_hit@1 | doc_hit@10 | MRR@10 (doc) |
|---|---|---|---|
| dense | **0.696** | 0.978 | **0.797** |
| hybrid, 10:1 dense:lexical (best of what was tested) | 0.630 | 0.957 | 0.756 |
| hybrid_reranked (10:1) | 0.587 | 0.957 | 0.695 |

The strongest dense baseline of the three models tested — beats granite
despite being a different model class. Unlike granite, **no fusion weight
ratio recovers dense-alone performance here** (tested 1:1 through 10:1; see
"Local HuggingFace Embedding Models" below for the full sweep). Dense-alone
is simply the better retrieval mode for this model on this corpus.

Full tables with all metrics in `reports/retrieval_mode_comparison.md`.

## Local HuggingFace Embedding Models (Octen-Embedding-0.6B)

**Goal:** add a third embedding model to the comparison, this time a local
HuggingFace model rather than an Ollama-served one, to test whether the
existing embedding pipeline could support that at all.

**Feasibility check — no new code needed.** `embedding.py` already had a
`SentenceTransformersEmbeddingClient` (lazy-imports `sentence_transformers`,
same pattern as `LocalRerankerClient`), already wired through
`EmbeddingConfig`, `load_embedding_config` (`provider: sentence_transformers`),
and `cli.py`'s `_build_embedding_client` — just never exercised against a
real model in this repo. GPU device placement is handled natively by the
`sentence-transformers` library itself (unlike the reranker, which needed
explicit `.to(device)` calls since it uses raw `transformers.AutoModel`).
Added a new `local-embed` optional dependency group
(`sentence-transformers`, `bitsandbytes`) to `pyproject.toml`.

**First attempt failed on hardware, not code.** The original target,
`Octen/Octen-Embedding-4B-INT8` (bitsandbytes INT8-quantized, ~4GB), loaded
fine but failed on the first real embedding call:
```
RuntimeError: cublasLt ran into an error!
cuBLAS API failed with status 15
```
Root cause: this GPU (GTX 1660 Ti) reports CUDA compute capability 7.5 (same
number as RTX 20-series "Turing" cards) but — unlike the RTX 20-series —
**lacks actual INT8 Tensor Cores**; Nvidia cut them from the GTX 16-series
for cost. `bitsandbytes`' INT8 matmul kernels require real Tensor Core
support this card doesn't have. Not fixable by batch size or retries — a
fixed hardware limitation. Confirmed via `torch.cuda.get_device_capability()`
and cross-referencing the GTX 1660 Ti's known lack of Tensor Cores.

**Switched to `Octen/Octen-Embedding-0.6B`** (same family, no quantization,
~1.2GB in bf16 — well within budget). No query-side instruction prefix
required per the model's own usage docs (plain symmetric `encode()` calls,
matching how this harness already treats all embedding providers).

**Batch size took three tries to get right** — worth recording since the
naive estimate was wrong twice:
- `batch_size=32` (matching nomic/granite's Ollama batch sizes): OOM
  immediately. Actual measured footprint (~4.9GB) was far higher than the
  naive fp32-weights estimate (~2.4GB) suggested.
- `batch_size=4`: safe (~2.3GB), but painfully slow — after ~55 minutes with
  no completion in sight, CPU time (a proxy for actual work done) was only
  ~9% of wall-clock time, meaning per-batch Python/CUDA kernel-launch
  overhead dominated over actual computation at that batch size.
- `batch_size=16`: safe (~4.0GB), meaningfully faster, but was interrupted
  mid-run for `batch_size=24` before finishing (see below).
- `batch_size=24`: ran at a tight but stable margin (~5.14GB used, ~600MB
  free) for the full 46-minute build with no OOM. **This is what actually
  built the index** (16,248 chunks, `data/indexes/financebench/octen-embedding-0.6b/`).

Corrected the two-point linear VRAM model along the way: `memory ≈ 1.73GB +
batch_size × 0.142GB`, derived from the `batch_size=4`/`batch_size=16` data
points, which correctly predicted the `batch_size=32` OOM (~6.27GB predicted
vs. ~5.61GB usable) and gave a theoretical ceiling of ~27 — consistent with
24 working and 32 failing. Also corrected an earlier wrong assumption: the
model loads in bf16/fp16 (2 bytes/param), not fp32 (4 bytes/param) as
initially guessed — the fp32 estimate alone exceeded the *entire* measured
`batch_size=4` footprint, which was impossible if that guess had been right.

**Weight sweep went further than granite's and found a different shape.**
Same process as granite (dense:lexical ratios via `--dense-weight`/
`--lexical-weight`), but the results kept improving well past 5:1, so the
sweep was extended to 7:1 and 10:1:

| Ratio | doc_hit@1 | doc_hit@5 | doc_mrr@10 |
|---|---|---|---|
| dense-alone | 0.696 | 0.957 | 0.797 |
| 1:1 | 0.391 | 0.696 | 0.537 |
| 2:1 | 0.435 | 0.826 | 0.592 |
| 3:1 | 0.565 | 0.870 | 0.683 |
| 5:1 | 0.630 | 0.870 | 0.734 |
| 7:1 | 0.630 | 0.935 | 0.749 |
| 10:1 | 0.630 | **0.957** | 0.756 |

`doc_hit@1` plateaued exactly at 0.630 from 5:1 onward (three ratios with no
further movement); `doc_mrr@10` gains shrank steadily (+0.015 at 5→7, +0.007
at 7→10) — approaching an asymptote *below* dense-alone (0.797), not
converging to it. Unlike granite (peaked and started declining by 5:1),
Octen's curve never turns over within the tested range, but it also never
reaches dense-alone. **Calibrated at 10:1** (best tested) in
`configs/embedding/octen_0.6b_local.yaml`, with an explicit note in that
config that dense-alone outperforms every hybrid ratio tried for this model.

`hybrid_reranked` (10:1) shows the same mixed pattern already seen with
granite: `doc_hit@20` reaches a perfect 1.000, but `doc_hit@1`/`doc_hit@5`/
`doc_mrr@10` all regress relative to plain `hybrid`, let alone dense-alone.

**Full three-model comparison in `reports/retrieval_mode_comparison.md`.**

**Queued next:** `nvidia/Nemotron-3-Embed-1B-BF16` at `batch_size=6` (math
extrapolated from Octen's corrected linear VRAM model, scaled for
Nemotron's larger size (1.14B params) and wider embeddings (2048 vs 1024
dim); ~9.8 theoretical max batch size, starting conservative at 6). Not yet
started as of this writeup.

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

**Follow-up resolved:** `hybrid_reranked` shares `HybridRetriever`'s
construction, so it now also picks up calibrated weights automatically. We
confirmed `fusion_k` has zero effect on `hybrid_reranked`'s final metrics —
but re-running granite `hybrid_reranked` with the 3:1 calibrated weights
shows `dense_weight`/`lexical_weight` behaves *differently*: it does change
the final results, but the effect is **mixed, not a clean win**:

| Variant | doc_hit@1 | doc_hit@5 | doc_hit@10 | doc_hit@20 | doc_mrr@10 |
|---|---|---|---|---|---|
| dense-alone | 0.587 | 0.935 | 0.978 | 0.978 | 0.713 |
| hybrid_reranked, unweighted 1:1 | 0.522 | 0.913 | 0.978 | 0.978 | 0.657 |
| hybrid_reranked, 3:1 (calibrated) | 0.543 | 0.870 | 0.957 | **1.000** | 0.673 |

`doc_hit@1` and `doc_mrr@10` both nudge up modestly, and `doc_hit@20` reaches
a perfect 1.000 (now exceeding dense-alone) — but `doc_hit@5` and
`doc_hit@10` both get *worse*. Reweighting the fusion stage does change which
chunks land in the reranker's top-`candidate_k` pool as hypothesized, but the
reranker's own behavior on the reweighted pool doesn't uniformly track the
plain-`hybrid` improvement — some correct-but-marginal dense candidates
apparently get displaced at some ranks before the reranker ever sees them.
Net: a small win on `doc_mrr@10`, but not the clean 3:1 win seen in plain
`hybrid` mode. `reports/retrieval_mode_comparison.md` now reflects the
re-run 3:1 numbers for granite `hybrid_reranked`.

## Evidence Coverage Metric (M9.11)

`evidence_text_hit@k` looked artificially low (~0.2 at best, vs. `doc_hit@10`
of ~0.98 for granite/Octen) because it uses symmetric Jaccard between one
chunk and the *entire* gold evidence span — and FinanceBench's gold evidence
for numeric questions is usually a whole financial-statement table (median
1,389 chars vs. an 800-char chunk_size; 64% of evidence spans exceed one
chunk). No single chunk can contain most of them, and any extra chunk
content dilutes the Jaccard union even when the chunk fully covers a shorter
span.

Added `evidence_coverage_hit@k` (additive — `evidence_text_hit@k` unchanged)
using asymmetric coverage (`evidence tokens found / evidence tokens total`,
the same formula as the existing `best_evidence_overlap`) over the
**concatenated top-k chunks**, crediting evidence correctly assembled across
adjacent chunks. Wired into `score_retrieval_run`/`retrieval_summary.json`/
`retrieval_leaderboard.json`/the rich report the same additive way nDCG was
wired in M9.2.

**Spot-check before shipping (25 questions that newly "hit" under the new
metric, Octen dense run):** 23/25 (92%) genuine — correct-document chunks
alone already clear threshold, real multi-chunk table assembly. 1/25
defensible-but-not-clean (3M's registered-securities table repeats verbatim
across its own 2018/2022/2023 filings). 1/25 confirmed false positive
(Amcor: the correct `2023Q2_10Q` filing has zero chunks in the top-10 at
all, but `AMCOR_2023_10K`'s structurally-identical restructuring-liability
table crosses threshold anyway) — a real, low-frequency failure mode when a
company reuses near-identical table structure across filing periods. Full
writeup and the complete `evidence_coverage_hit@k` table across all
9 model×mode runs in `reports/retrieval_mode_comparison.md`.

Headline result: `hybrid_reranked`'s `evidence_coverage_hit@20` is the best
score in the whole table for granite (0.848) and nomic (0.587, its best
result on any metric) — reranking's evidence-assembly benefit was
completely invisible under the strict metric, which showed reranking
*hurting* `evidence_text_hit@k` for every model.

## Next Steps

1. Raise M9 PR and merge to main.
2. **Queued:** `nvidia/Nemotron-3-Embed-1B-BF16` at `batch_size=6` as a fourth
   embedding model — see "Local HuggingFace Embedding Models" above for the
   VRAM math. Not yet started.
3. M10 (fine-tuned reranker): train a LoRA adapter for `bge-reranker-base` and
   load it via `RerankerConfig.adapter_path` — the seam is already in place.
4. M13 (Snowflake embedding comparison) can reuse the same hybrid/reranked
   retrieval modes against a different base embedding model — will need its
   own calibrated `fusion_dense_weight`/`fusion_lexical_weight`, following the
   same sweep process documented above (now proven across three models:
   nomic, granite, Octen).
5. Consider whether `hybrid_reranked`'s mixed result (see "Follow-up
   resolved" above) warrants its own weight calibration separate from plain
   `hybrid` — the ratio tuned for `hybrid` isn't obviously optimal once a
   reranker sits downstream of it; would need its own small sweep, for both
   granite and Octen.
6. Known gap (tracked separately, not fixed in M9): `rag_predictions.jsonl`
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
# Default embedding model (configs/retrieval.yaml) is Octen-Embedding-0.6B, a
# local HuggingFace sentence-transformers model — no ollama pull needed for
# it, but it does need `pip install -e ".[local-embed]"` and downloads
# weights from HF Hub on first use.

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
- `local-embed` is an optional dependency group (`pip install .[local-embed]`) — sentence-transformers/bitsandbytes, only needed for local HuggingFace embedding models (`provider: sentence_transformers`).
- Do not publish benchmark claims until backed by reproducible runs with committed configuration.
- This GPU (GTX 1660 Ti) cannot run `bitsandbytes` INT8-quantized models — it lacks real Tensor Cores despite reporting compute capability 7.5. Any `-INT8`-suffixed HuggingFace model should be assumed incompatible unless verified otherwise.
