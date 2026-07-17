# Project Status

## Current State

- M9 complete on branch `M9` — PR not yet raised.
- 977 tests pass, 2 skipped (Ollama integration tests).
- Full pipeline operational through hybrid retrieval (BM25 + RRF + cross-encoder reranking) and nDCG scoring.
- Live comparison run recorded on the real FinanceBench sample corpus (46 questions, `nomic-embed-text` embeddings, `BAAI/bge-reranker-base` reranker) — see `reports/retrieval_mode_comparison.md`.

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

## Real-data comparison (46 FinanceBench sample questions, nomic-embed-text)

| Mode | doc_hit@10 | MRR@10 (doc) | evidence_text_ndcg@10 |
|---|---|---|---|
| dense | 0.326 | 0.119 | 0.000 |
| hybrid (BM25 + RRF) | 0.652 | 0.249 | 0.011 |
| hybrid_reranked (bge-reranker-base) | 0.717 | 0.452 | 0.000 |

Hybrid retrieval roughly doubles doc-level hit rate over dense-only on this
sample; reranking further concentrates the correct document at a higher rank
(MRR nearly quadruples over dense). `evidence_text_ndcg@10` stays near zero
across all three modes — exact evidence-text matching is rare in this small
sample regardless of retrieval mode, consistent with the known FinanceBench
table-extraction gap already tracked by this project's failure-label analysis.
Full table with all metrics in `reports/retrieval_mode_comparison.md`.

## Next Steps

1. Raise M9 PR and merge to main.
2. M10 (fine-tuned reranker): train a LoRA adapter for `bge-reranker-base` and
   load it via `RerankerConfig.adapter_path` — the seam is already in place.
3. M13 (Snowflake embedding comparison) can reuse the same hybrid/reranked
   retrieval modes against a different base embedding model.
4. Known gap (tracked separately, not fixed in M9): `rag_predictions.jsonl`
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
