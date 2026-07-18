# Project Story

## Current state

- M9 implementation complete on branch `M9` (11 commits ahead of main, PR not yet raised).
- 977 tests pass, 2 skipped (Ollama integration tests).
- `retrieve --mode {dense,lexical,hybrid,hybrid_reranked}` fully operational; `eval-retrieval` now reports nDCG@k alongside hit@k/MRR@k.
- Live comparison run recorded on the real FinanceBench sample (46 questions, `nomic-embed-text` embeddings, real `BAAI/bge-reranker-base` reranker) — see `reports/retrieval_mode_comparison.md`.

## Latest milestone

**M9: Hybrid Retrieval — BM25 + RRF + Cross-Encoder Reranking + nDCG** — all 10 sub-milestones complete, built test-first (one tracer-bullet-then-loop TDD cycle per sub-milestone, one commit each).

Key deliverables:

- `retrieval_scoring.py` — `ndcg_at_k()`, `score_ndcg()`, `summarize_ndcg()` (M9.1); wired into `eval_retrieval.score_retrieval_run` output, the rich retrieval report, and `compare-runs --metric` (M9.2), all additive to the existing schema.
- `lexical_index.py` — `BM25Config` + `BM25Index` (Okapi BM25, Lucene-style non-negative IDF floor), built in-memory from `chunk_metadata.jsonl` (M9.3).
- `fusion.py` — `reciprocal_rank_fusion()` (M9.4).
- `reranker.py` — `RerankerConfig`/`RerankerClient` Protocol mirroring `EmbeddingClient`; `MockRerankerClient`, `HTTPRerankerClient` (M9.5); `LocalRerankerClient` running a real cross-encoder with lazy torch/transformers import and optional LoRA adapter loading via peft (M9.6, for M10).
- `hybrid_retriever.py` — `Retriever` Protocol + `DenseRetriever`/`LexicalRetriever`/`HybridRetriever`/`RerankedRetriever` compositions, `run_retrieval_with_retriever()` (M9.7). `retriever.run_retrieval` refactored into a thin wrapper over `DenseRetriever` + `run_retrieval_with_retriever` — behavior-preserving, existing tests pass unchanged.
- CLI: `retrieve --mode {dense,lexical,hybrid,hybrid_reranked} --candidate-k --fusion-k --reranker-config` (M9.8). Scope note: `PipelineConfig`/`RetrievalConfig` YAML schema changes for the `--config`-driven retrieve path were deferred — not covered by M9.8's acceptance criteria.
- `configs/evaluation/rag_modes.yaml` — added `rag_hybrid`/`rag_reranked` (context_source: retrieved_chunks, reusing the `rag_dense_v2` prompt); real end-to-end mock-LLM + mock-reranker pipeline tests replace the old placeholder enum test (M9.9).
- `pyproject.toml` — added the `rerank` optional dependency group (`torch`, `transformers`, `peft`) that M9.6's design called for but was missed until M9.10 exposed the gap.
- `reports/retrieval_mode_comparison.md` — real dense vs. hybrid vs. hybrid_reranked comparison (M9.10, see table below).

## Real-data comparison (46 FinanceBench sample questions, nomic-embed-text)

| Mode | doc_hit@10 | MRR@10 (doc) | evidence_text_ndcg@10 |
|---|---|---|---|
| dense | 0.326 | 0.119 | 0.000 |
| hybrid (BM25 + RRF) | 0.652 | 0.249 | 0.011 |
| hybrid_reranked (bge-reranker-base) | 0.717 | 0.452 | 0.000 |

Hybrid retrieval roughly doubles doc-level hit rate over dense-only; reranking
further concentrates the correct document at a higher rank (MRR nearly
quadruples over dense). `evidence_text_ndcg@10` stays near zero across all
three modes on this small sample — consistent with the project's already-known
FinanceBench table-extraction gap, not a new finding.

## Next step

1. **Raise and merge M9 PR** — branch is clean, all tests pass.
2. **M10 (fine-tuned reranker)** — the seam is ready: train a LoRA adapter for `bge-reranker-base`, load it via `RerankerConfig.adapter_path` / `LocalRerankerClient`, compare against the pretrained baseline recorded here.
3. **M13 (Snowflake embedding comparison)** — can reuse the same hybrid/reranked retrieval modes against a different base embedding model without new plumbing.
4. **Fix the grounding-analysis context gap** (spawned as a separate task, not fixed in M9): `run_rag_from_config` writes `context_chunk_ids` (list of IDs) into `rag_predictions.jsonl`, but `analyze_grounding` reads `retrieved_chunks` (list of {chunk_id, text} dicts) — the key names don't match, so citation/context-sufficiency checks silently see empty context on every real `run-rag` → `analyze-grounding` run. Existing tests never caught this because they hand-build `rag_predictions.jsonl` fixtures with `retrieved_chunks` already present.

## Known risks or blockers

- The grounding-analysis field-name gap above predates M9 but was only noticed while wiring M9.9's end-to-end pipeline test; it means no `analyze-grounding` run on real (non-fixture) data has ever actually seen retrieved context.
- `reports/retrieval_mode_comparison.md` is committed as an intentional exception to the `reports/*.md` gitignore rule (curated, small, reviewable — per AGENTS.md's allowance for intentionally curated documentation).
- `evidence_text_ndcg@10` is a very strict, low-frequency metric on this 46-question sample (most runs saw 0 exact evidence-text matches); don't over-read the dense-vs-hybrid comparison on that single metric — `doc_hit@10` and `doc_mrr@10` are the more informative columns here.

**Fixed post-M9.10:** `compare-runs`' table used to show `—` for `mode`/`model`/`mrr@10` on retrieval-only runs (it only read `run_metadata.json`, a RAG-run file, and a nonexistent `mrr@10` key) and picked "best run" by the requested `--metric` alone with no tie-breaking. `report_loader.load_run_data` now also loads `retrieval_run_metadata.json`; `compare-runs` falls back to it for `mode`/`model`, falls back to `doc_mrr@10` for the `mrr@10` column, and tie-breaks best-run selection on `doc_hit@10` then `doc_mrr@10` after the requested metric. `reports/retrieval_mode_comparison.md` was regenerated with the fix.

## Safe next commands

```bash
# Confirm tests still pass
pytest -q

# Raise M9 PR
git push origin M9
gh pr create --title "feat(M9): hybrid retrieval (BM25 + RRF + reranking) and nDCG" --body "..."

# Re-run the three-mode comparison from scratch
# Note: --mode/--candidate-k/--fusion-k/--reranker-config only exist on the
# individual-flags path below, not the --config pipeline-config path (M9.8 scope decision).
COMMON="--index-dir data/indexes/financebench \
  --questions data/processed/financebench/examples.jsonl \
  --embedding-config configs/embedding/ollama_nomic.yaml \
  --chunks-path data/processed/financebench/chunks.jsonl --top-k 10"

python -m financebench_eval retrieve $COMMON --run-id dense_check \
  --output runs/dense_check/retrieval_results.jsonl --mode dense
python -m financebench_eval retrieve $COMMON --run-id hybrid_check \
  --output runs/hybrid_check/retrieval_results.jsonl --mode hybrid
python -m financebench_eval retrieve $COMMON --run-id reranked_check \
  --output runs/reranked_check/retrieval_results.jsonl \
  --mode hybrid_reranked --reranker-config configs/reranker/bge_base_local.yaml

python -m financebench_eval eval-retrieval --config configs/retrieval.yaml --run-id dense_check
python -m financebench_eval eval-retrieval --config configs/retrieval.yaml --run-id hybrid_check
python -m financebench_eval eval-retrieval --config configs/retrieval.yaml --run-id reranked_check

python -m financebench_eval compare-runs \
  --runs runs/dense_check runs/hybrid_check runs/reranked_check \
  --metric evidence_text_ndcg@10 --output reports/retrieval_mode_comparison.md
```
