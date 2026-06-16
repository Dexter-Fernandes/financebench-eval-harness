"""M6 embedding model comparison runner.

Runs the same retrieval experiment across multiple embedding models under
identical chunking and retrieval settings, then collects results for
leaderboard and report generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from financebench_eval_harness.chunking import ChunkingConfig
from financebench_eval_harness.embedding import (
    EmbeddingClient,
    EmbeddingConfig,
    EmbeddingProviderError,
    MockEmbeddingClient,
    SentenceTransformersEmbeddingClient,
)
from financebench_eval_harness.embedding_cache import EmbeddingCache
from financebench_eval_harness.embedding_comparison_config import (
    EmbeddingComparisonConfig,
    EmbeddingModelSpec,
)
from financebench_eval_harness.eval_retrieval import score_retrieval_run
from financebench_eval_harness.index_builder import (
    IndexBuildError,
    IndexMetadata,
    build_index,
    load_index,
)
from financebench_eval_harness.pipeline_config import PipelineConfig
from financebench_eval_harness.retrieval_types import Chunk
from financebench_eval_harness.retriever import Question, run_retrieval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EmbeddingProviderUnavailableError(RuntimeError):
    """Raised when a required embedding provider/SDK is not installed."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ModelComparisonResult:
    """Result for a single embedding model in the comparison."""

    spec: EmbeddingModelSpec
    run_dir: Path
    summary: dict[str, Any] | None = None
    embedding_latency_s: float = 0.0
    retrieval_latency_s: float = 0.0
    index_size_mb: float = 0.0
    estimated_cost_usd: float | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class EmbeddingComparisonResult:
    """Aggregate result from comparing multiple embedding models."""

    run_dir: Path
    model_results: list[ModelComparisonResult] = field(default_factory=list)

    @property
    def succeeded_count(self) -> int:
        return sum(1 for r in self.model_results if r.succeeded)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.model_results if not r.succeeded)

    @property
    def failed_models(self) -> dict[str, str]:
        return {r.spec.name: r.error or "" for r in self.model_results if not r.succeeded}

    @property
    def successful_results(self) -> list[ModelComparisonResult]:
        return [r for r in self.model_results if r.succeeded]


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def build_client_for_model_spec(spec: EmbeddingModelSpec) -> EmbeddingClient:
    """Instantiate the appropriate EmbeddingClient for a model spec."""
    cfg = spec.to_embedding_config()

    if spec.provider == "mock":
        return MockEmbeddingClient(cfg)

    if spec.provider == "ollama":
        from financebench_eval_harness.embedding import OllamaEmbeddingClient
        return OllamaEmbeddingClient(cfg)

    if spec.provider == "sentence_transformers":
        return SentenceTransformersEmbeddingClient(cfg)

    if spec.provider == "openai":
        try:
            from financebench_eval_harness.embedding_providers import OpenAIEmbeddingClient
            return OpenAIEmbeddingClient(cfg)
        except ImportError as exc:
            raise EmbeddingProviderUnavailableError(
                f"openai SDK is required for provider 'openai'. "
                "Install it with: pip install openai"
            ) from exc

    if spec.provider == "voyage":
        try:
            from financebench_eval_harness.embedding_providers import VoyageEmbeddingClient
            return VoyageEmbeddingClient(cfg)
        except ImportError as exc:
            raise EmbeddingProviderUnavailableError(
                f"voyageai SDK is required for provider 'voyage'. "
                "Install it with: pip install voyageai"
            ) from exc

    raise EmbeddingProviderUnavailableError(
        f"Unknown embedding provider '{spec.provider}'. "
        "Supported: mock, ollama, sentence_transformers, openai, voyage."
    )


# ---------------------------------------------------------------------------
# Corpus loading + hashing
# ---------------------------------------------------------------------------


def _load_chunks(chunks_path: Path) -> list[Chunk]:
    """Load chunks from a JSONL file."""
    chunks: list[Chunk] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(Chunk(
                chunk_id=d["chunk_id"],
                doc_id=d.get("doc_id", ""),
                doc_name=d["doc_name"],
                page_num=int(d["page_num"]),
                text=d["text"],
                start_char=int(d.get("start_char", 0)),
                end_char=int(d.get("end_char", len(d["text"]))),
            ))
    return chunks


def _load_questions(questions_path: Path) -> list[Question]:
    """Load questions from a JSONL file."""
    questions: list[Question] = []
    with questions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions.append(Question(
                question_id=d["question_id"],
                query=d["question"],
            ))
    return questions


def _corpus_hash(chunks: list[Chunk]) -> str:
    """Stable SHA-256 of the corpus, order-independent."""
    entries = sorted(f"{c.chunk_id}\t{c.text}" for c in chunks)
    content = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------


def _build_or_load_index(
    chunks: list[Chunk],
    client: EmbeddingClient,
    config: EmbeddingComparisonConfig,
    spec: EmbeddingModelSpec,
    corpus_hash: str,
    cache: EmbeddingCache,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[Any, IndexMetadata, float]:
    """Build (or reuse) the FAISS index for one model. Returns (store, meta, latency_s)."""
    from financebench_eval_harness.vector_store import FaissVectorStore

    index_dir = config.index_base_dir / spec.slug
    index_dir.mkdir(parents=True, exist_ok=True)

    # Try to reuse existing index if corpus matches.
    meta_path = index_dir / "index_metadata.json"
    if meta_path.is_file():
        store, meta = load_index(index_dir)
        if meta.corpus_hash == corpus_hash:
            logger.info("[%s] Reusing existing index (corpus hash match)", spec.name)
            return store, meta, 0.0
        logger.warning("[%s] Corpus hash mismatch — rebuilding index", spec.name)

    # Build from scratch, using cache to avoid re-embedding known texts.
    t0 = time.perf_counter()
    cached_vecs, hits, misses = cache.get_batch([c.text for c in chunks])
    logger.info("[%s] Cache: %d hits, %d misses", spec.name, hits, misses)

    # For uncached chunks, call the embedding API.
    miss_indices = [i for i, v in enumerate(cached_vecs) if v is None]
    if miss_indices:
        miss_texts = [chunks[i].text for i in miss_indices]
        batch_size = client.config.batch_size
        if on_event:
            on_event("embed_start", {
                "model": spec.name,
                "cache_hits": hits,
                "cache_misses": misses,
                "total_to_embed": len(miss_texts),
            })
        new_vecs: list[list[float]] = []
        for start in range(0, len(miss_texts), batch_size):
            new_vecs.extend(client.embed_texts(miss_texts[start:start + batch_size]))
            if on_event:
                on_event("embed_progress", {
                    "model": spec.name,
                    "completed": min(start + batch_size, len(miss_texts)),
                    "total": len(miss_texts),
                })
        cache.put_batch(miss_texts, new_vecs)
        for idx, vec in zip(miss_indices, new_vecs):
            cached_vecs[idx] = vec

    all_vecs: list[list[float]] = [v for v in cached_vecs]  # type: ignore[misc]
    elapsed = time.perf_counter() - t0

    dim = len(all_vecs[0])
    store = FaissVectorStore(dim=dim)
    store.add(chunks, all_vecs)
    store.save(index_dir)

    from financebench_eval_harness.retrieval_config import RetrievalConfig
    ret_cfg = config.retrieval

    meta = IndexMetadata(
        embedding_provider=spec.provider,
        embedding_model=spec.name,
        chunk_size=ret_cfg.chunking.chunk_size,
        chunk_overlap=ret_cfg.chunking.chunk_overlap,
        min_chunk_chars=ret_cfg.chunking.min_chunk_chars,
        corpus_hash=corpus_hash,
        chunk_count=len(chunks),
        build_time_utc=datetime.now(timezone.utc).isoformat(),
    )
    (index_dir / "index_metadata.json").write_text(
        json.dumps(meta.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return store, meta, elapsed


def _run_one_model(
    spec: EmbeddingModelSpec,
    chunks: list[Chunk],
    questions: list[Question],
    corpus_hash: str,
    config: EmbeddingComparisonConfig,
    model_run_dir: Path,
    on_event: Callable[[str, dict], None] | None = None,
) -> ModelComparisonResult:
    """Run index build + retrieval + scoring for one model."""
    model_run_dir.mkdir(parents=True, exist_ok=True)

    client = build_client_for_model_spec(spec)
    cache = EmbeddingCache(
        config.cache_dir,
        provider=spec.provider,
        model_name=spec.name,
        dimensions=spec.dimensions,
        normalize=spec.normalize,
    )

    store, index_meta, embed_latency = _build_or_load_index(
        chunks, client, config, spec, corpus_hash, cache, on_event=on_event
    )

    # Run retrieval.
    t_ret = time.perf_counter()
    run_retrieval(
        questions,
        store,
        client,
        model_run_dir / "retrieval_results.jsonl",
        top_k=config.retrieval.top_k,
        run_id=f"{config.run_id}/{spec.slug}",
        dataset_path=config.retrieval.questions_path,
        chunks_path=config.retrieval.chunks_path,
        index_metadata=index_meta,
    )
    ret_latency = time.perf_counter() - t_ret

    # Score retrieval.
    pipeline_cfg = _make_pipeline_config(config, spec)
    summary = score_retrieval_run(pipeline_cfg, model_run_dir)

    # Index size.
    faiss_file = config.index_base_dir / spec.slug / "index.faiss"
    index_size_mb = faiss_file.stat().st_size / 1_048_576 if faiss_file.is_file() else 0.0

    return ModelComparisonResult(
        spec=spec,
        run_dir=model_run_dir,
        summary=summary,
        embedding_latency_s=embed_latency,
        retrieval_latency_s=ret_latency,
        index_size_mb=index_size_mb,
        estimated_cost_usd=_estimate_cost(spec, chunks),
    )


def _make_pipeline_config(config: EmbeddingComparisonConfig, spec: EmbeddingModelSpec) -> PipelineConfig:
    """Build a minimal PipelineConfig for score_retrieval_run."""
    ret = config.retrieval
    return PipelineConfig(
        pages_path=ret.chunks_path,   # unused by scorer but required by dataclass
        chunks_path=ret.chunks_path,
        index_dir=config.index_base_dir / spec.slug,
        questions_path=ret.questions_path,
        runs_dir=config.runs_dir / config.run_id / "model_runs" / spec.slug,
        top_k=ret.top_k,
        evidence_overlap_threshold=ret.evidence_overlap_threshold,
        chunking=ret.chunking,
        embedding=spec.to_embedding_config(),
    )


def _estimate_cost(spec: EmbeddingModelSpec, chunks: list[Chunk]) -> float | None:
    """Return a rough API cost estimate in USD, or None for local models."""
    total_chars = sum(len(c.text) for c in chunks)
    approx_tokens = total_chars / 4  # rough char→token ratio
    per_million = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "voyage-finance-2": 0.12,
    }
    rate = per_million.get(spec.name)
    if rate is None:
        return None
    return round(approx_tokens / 1_000_000 * rate, 6)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_leaderboard(results: list[ModelComparisonResult]) -> list[dict[str, Any]]:
    """Build a leaderboard sorted by evidence_hit@10 descending.

    Failed models (result.error is not None) are excluded.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        if not r.succeeded or r.summary is None:
            continue
        s = r.summary
        row: dict[str, Any] = {
            "model": r.spec.name,
            "provider": r.spec.provider,
            "category": r.spec.category,
            "embedding_dim": r.spec.dimensions,
            "doc_hit@10": s.get("doc_hit@10_rate", 0.0),
            "page_hit@10": s.get("page_hit@10_rate", 0.0),
            "evidence_hit@10": s.get("evidence_text_hit@10_rate", 0.0),
            "mrr@10": s.get("doc_mrr@10", 0.0),
            "median_first_hit_rank": s.get("median_first_hit_rank"),
            "embedding_latency_s": r.embedding_latency_s,
            "retrieval_latency_s": r.retrieval_latency_s,
            "index_size_mb": r.index_size_mb,
            "estimated_cost_usd": r.estimated_cost_usd,
        }
        rows.append(row)
    rows.sort(key=lambda x: x["evidence_hit@10"], reverse=True)
    return rows


def _write_leaderboard(leaderboard: list[dict[str, Any]], run_dir: Path) -> None:
    """Write leaderboard as JSON and CSV."""
    (run_dir / "embedding_leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not leaderboard:
        (run_dir / "embedding_leaderboard.csv").write_text("", encoding="utf-8")
        return
    columns = list(leaderboard[0].keys())
    lines = [",".join(columns)]
    for row in leaderboard:
        values = [str(row.get(c, "")) for c in columns]
        lines.append(",".join(values))
    (run_dir / "embedding_leaderboard.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def make_model_decision(results: list[ModelComparisonResult]) -> dict[str, Any]:
    """Assign model roles based on category and evidence_hit@10.

    Roles:
      default_model            — highest evidence_hit@10 overall
      cheap_baseline           — best cheap_api model
      quality_baseline         — best quality_api model
      local_baseline           — best open_source model (no API cost)
      finance_specialized_candidate — best finance_specialized model
    """
    successful = [r for r in results if r.succeeded and r.summary is not None]

    def _best_in_category(category: str) -> str | None:
        candidates = [r for r in successful if r.spec.category == category]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.summary.get("evidence_text_hit@10_rate", 0.0)).spec.name  # type: ignore[union-attr]

    def _best_overall() -> str | None:
        if not successful:
            return None
        return max(successful, key=lambda r: r.summary.get("evidence_text_hit@10_rate", 0.0)).spec.name  # type: ignore[union-attr]

    default = _best_overall()
    local = _best_in_category("open_source")
    cheap = _best_in_category("cheap_api")
    quality = _best_in_category("quality_api")
    finance = _best_in_category("finance_specialized")

    if default is not None:
        winner = next((r for r in successful if r.spec.name == default), None)
        score = winner.summary.get("evidence_text_hit@10_rate", 0.0) if winner and winner.summary else 0.0
        reason = (
            f"{default} achieved {score:.2f} evidence_hit@10"
            + (" with zero API cost and local reproducibility."
               if winner and winner.spec.category == "open_source"
               else ".")
        )
    else:
        reason = "No successful model runs to evaluate."

    return {
        "default_model": default,
        "cheap_baseline": cheap,
        "quality_baseline": quality,
        "local_baseline": local,
        "finance_specialized_candidate": finance,
        "primary_metric": "evidence_hit@10",
        "reason": reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_embedding_comparison(
    config: EmbeddingComparisonConfig,
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> EmbeddingComparisonResult:
    """Run retrieval comparison across all embedding models in the config.

    For each model: build/load index, retrieve, score.
    Failed models are logged; the loop continues unless fail_fast=True.

    on_event: optional callback fired with (event_name, info_dict) at key
    points. Recognised events: model_start, model_done, model_failed,
    embed_start, embed_progress.
    """
    # Load corpus.
    chunks = _load_chunks(config.retrieval.chunks_path)
    questions = _load_questions(config.retrieval.questions_path)
    corpus_hash = _corpus_hash(chunks)

    # Create comparison run directory.
    run_dir = config.runs_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config snapshot (M6.7 — corpus hash recorded).
    snapshot: dict[str, Any] = {
        "run_id": config.run_id,
        "corpus_hash": corpus_hash,
        "chunk_count": len(chunks),
        "question_count": len(questions),
        "top_k": config.retrieval.top_k,
        "evidence_overlap_threshold": config.retrieval.evidence_overlap_threshold,
        "chunks_path": str(config.retrieval.chunks_path),
        "questions_path": str(config.retrieval.questions_path),
        "embedding_models": [
            {"name": s.name, "provider": s.provider, "category": s.category}
            for s in config.embedding_models
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.yaml").write_text(yaml.dump(snapshot, allow_unicode=True), encoding="utf-8")

    model_results: list[ModelComparisonResult] = []
    model_runs_dir = run_dir / "model_runs"

    n_models = len(config.embedding_models)
    for idx, spec in enumerate(config.embedding_models):
        model_run_dir = model_runs_dir / spec.slug
        logger.info("Running comparison for model: %s (%s)", spec.name, spec.provider)
        if on_event:
            on_event("model_start", {
                "model": spec.name,
                "provider": spec.provider,
                "index": idx + 1,
                "total": n_models,
            })
        t_model = time.perf_counter()
        try:
            result = _run_one_model(
                spec, chunks, questions, corpus_hash, config, model_run_dir,
                on_event=on_event,
            )
            model_results.append(result)
            hit_rate = (
                result.summary.get(f"evidence_text_hit@{config.retrieval.top_k}_rate", 0.0)
                if result.summary else 0.0
            )
            logger.info("[%s] Done. evidence_hit@%d = %.3f", spec.name, config.retrieval.top_k, hit_rate)
            if on_event:
                on_event("model_done", {
                    "model": spec.name,
                    "evidence_hit_at_10": hit_rate,
                    "elapsed_s": time.perf_counter() - t_model,
                })
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("[%s] FAILED: %s", spec.name, msg)
            model_results.append(ModelComparisonResult(
                spec=spec,
                run_dir=model_run_dir,
                error=msg,
            ))
            if on_event:
                on_event("model_failed", {
                    "model": spec.name,
                    "error": msg,
                })
            if config.fail_fast:
                raise

    leaderboard = compute_leaderboard(model_results)
    _write_leaderboard(leaderboard, run_dir)

    decision = make_model_decision(model_results)
    (run_dir / "embedding_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return EmbeddingComparisonResult(run_dir=run_dir, model_results=model_results)


__all__ = [
    "EmbeddingComparisonResult",
    "EmbeddingProviderUnavailableError",
    "ModelComparisonResult",
    "build_client_for_model_spec",
    "compute_leaderboard",
    "make_model_decision",
    "run_embedding_comparison",
]
