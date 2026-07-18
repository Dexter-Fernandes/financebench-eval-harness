"""M5.1 — RAG evaluation mode definitions (TDD)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from financebench_eval_harness.evaluation import (
    EvaluationMode,
    load_evaluation_config,
    render_prompt,
)

# ---------------------------------------------------------------------------
# Cycle 1 — Enum values
# ---------------------------------------------------------------------------


def test_rag_dense_mode_round_trips() -> None:
    assert EvaluationMode("rag_dense") is EvaluationMode.RAG_DENSE
    assert EvaluationMode.RAG_DENSE.value == "rag_dense"


def test_rag_oracle_mode_round_trips() -> None:
    assert EvaluationMode("rag_oracle") is EvaluationMode.RAG_ORACLE
    assert EvaluationMode.RAG_ORACLE.value == "rag_oracle"


def test_rag_no_context_mode_round_trips() -> None:
    assert EvaluationMode("rag_no_context") is EvaluationMode.RAG_NO_CONTEXT
    assert EvaluationMode.RAG_NO_CONTEXT.value == "rag_no_context"


def test_rag_hybrid_mode_round_trips() -> None:
    assert EvaluationMode("rag_hybrid") is EvaluationMode.RAG_HYBRID
    assert EvaluationMode.RAG_HYBRID.value == "rag_hybrid"


def test_rag_reranked_mode_round_trips() -> None:
    assert EvaluationMode("rag_reranked") is EvaluationMode.RAG_RERANKED
    assert EvaluationMode.RAG_RERANKED.value == "rag_reranked"


# ---------------------------------------------------------------------------
# Cycle 2 — Config loading
# ---------------------------------------------------------------------------

RAG_CONFIG_PATH = Path("configs/evaluation/rag_modes.yaml")


def test_load_rag_modes_config_returns_five_modes() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    assert set(config.modes) == {
        EvaluationMode.RAG_DENSE,
        EvaluationMode.RAG_ORACLE,
        EvaluationMode.RAG_NO_CONTEXT,
        EvaluationMode.RAG_HYBRID,
        EvaluationMode.RAG_RERANKED,
    }


def test_load_rag_modes_config_has_correct_prompt_ids() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    assert config.prompt_for(EvaluationMode.RAG_DENSE).id == "rag_dense_v2"
    assert config.prompt_for(EvaluationMode.RAG_ORACLE).id == "rag_oracle_v1"
    assert config.prompt_for(EvaluationMode.RAG_NO_CONTEXT).id == "rag_no_context_v1"
    assert config.prompt_for(EvaluationMode.RAG_HYBRID).id == "rag_hybrid_v1"
    assert config.prompt_for(EvaluationMode.RAG_RERANKED).id == "rag_reranked_v1"


# ---------------------------------------------------------------------------
# Cycle 3 — context_source field
# ---------------------------------------------------------------------------


def test_rag_dense_context_source_is_retrieved_chunks() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_DENSE).context_source == "retrieved_chunks"


def test_rag_oracle_context_source_is_gold_evidence() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_ORACLE).context_source == "gold_evidence"


def test_rag_no_context_context_source_is_none() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_NO_CONTEXT).context_source == "none"


def test_rag_hybrid_context_source_is_retrieved_chunks() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_HYBRID).context_source == "retrieved_chunks"


def test_rag_reranked_context_source_is_retrieved_chunks() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())
    assert config.prompt_for(EvaluationMode.RAG_RERANKED).context_source == "retrieved_chunks"


# ---------------------------------------------------------------------------
# Cycle 4-6 — render_prompt dispatches on context_source
# ---------------------------------------------------------------------------


def test_render_rag_dense_injects_retrieved_chunks_into_prompt() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_DENSE,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million.", "Gross profit was $4 million."],
    )

    assert "What was FY2022 revenue?" in rendered.text
    assert "[Evidence 1]" in rendered.text
    assert "Revenue was $10 million." in rendered.text
    assert "[Evidence 2]" in rendered.text
    assert "Gross profit was $4 million." in rendered.text
    assert rendered.mode is EvaluationMode.RAG_DENSE


def test_render_rag_no_context_omits_evidence_even_when_supplied() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_NO_CONTEXT,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million."],
    )

    assert "What was FY2022 revenue?" in rendered.text
    assert "Revenue was $10 million." not in rendered.text
    assert rendered.mode is EvaluationMode.RAG_NO_CONTEXT


def test_render_rag_oracle_injects_gold_evidence_text() -> None:
    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    rendered = render_prompt(
        config,
        EvaluationMode.RAG_ORACLE,
        question="What was FY2022 revenue?",
        evidence_texts=["Revenue was $10 million."],
    )

    assert "What was FY2022 revenue?" in rendered.text
    assert "Revenue was $10 million." in rendered.text
    assert rendered.mode is EvaluationMode.RAG_ORACLE


def test_render_rag_dense_raises_when_no_evidence_supplied() -> None:
    from financebench_eval_harness.evaluation import EvaluationConfigError

    config = load_evaluation_config(RAG_CONFIG_PATH, required_modes=frozenset())

    with pytest.raises(EvaluationConfigError):
        render_prompt(
            config,
            EvaluationMode.RAG_DENSE,
            question="What was FY2022 revenue?",
            evidence_texts=None,
        )


# ---------------------------------------------------------------------------
# Cycle 7 — baselines still work after context_source is added to baselines.yaml
# ---------------------------------------------------------------------------


def test_baseline_closed_book_context_source_is_none() -> None:
    from financebench_eval_harness.evaluation import DEFAULT_EVALUATION_CONFIG_PATH

    config = load_evaluation_config(DEFAULT_EVALUATION_CONFIG_PATH)
    assert config.prompt_for(EvaluationMode.CLOSED_BOOK).context_source == "none"


def test_baseline_oracle_context_context_source_is_gold_evidence() -> None:
    from financebench_eval_harness.evaluation import DEFAULT_EVALUATION_CONFIG_PATH

    config = load_evaluation_config(DEFAULT_EVALUATION_CONFIG_PATH)
    assert config.prompt_for(EvaluationMode.ORACLE_CONTEXT).context_source == "gold_evidence"


# ---------------------------------------------------------------------------
# M9.9 — rag_hybrid / rag_reranked end-to-end:
# hybrid retrieve -> run-rag -> score-rag -> eval-retrieval -> join-metrics -> analyze-grounding
# ---------------------------------------------------------------------------


def _build_rag_modes_index(tmp_path: Path) -> Path:
    from financebench_eval_harness.chunking import chunk_pages
    from financebench_eval_harness.embedding import EmbeddingConfig, MockEmbeddingClient
    from financebench_eval_harness.index_builder import build_index
    from financebench_eval_harness.retrieval_config import load_retrieval_config
    from financebench_eval_harness.retrieval_types import DocumentPage

    pages = [
        DocumentPage(doc_id="ACME", doc_name="ACME.pdf", page_num=1,
                     text="Total revenue for fiscal year 2022 was $4.2 billion."),
        DocumentPage(doc_id="ACME", doc_name="ACME.pdf", page_num=2,
                     text="Net income was $540 million, down 3% year over year."),
    ]
    retrieval_cfg_path = tmp_path / "retrieval.yaml"
    retrieval_cfg_path.write_text(
        "\n".join([
            "retrieval:",
            "  evidence_overlap_threshold: 0.5",
            "  chunking:",
            "    strategy: recursive_text",
            "    chunk_size: 200",
            "    chunk_overlap: 50",
            "    min_chunk_chars: 10",
        ]),
        encoding="utf-8",
    )
    cfg = load_retrieval_config(retrieval_cfg_path)
    chunks = chunk_pages(pages, cfg.chunking)

    embed_cfg = EmbeddingConfig(provider="mock", model_name="mock-embed")
    client = MockEmbeddingClient(embed_cfg, embedding_dim=8)

    index_dir = tmp_path / "index"
    build_index(chunks, client, cfg, index_dir)
    return index_dir


def _run_rag_hybrid_pipeline(tmp_path: Path, *, mode: str, retrieve_mode: str) -> Path:
    """Runs hybrid retrieve -> run-rag -> score-rag for `mode`; returns the run_dir."""
    import json as json_module

    from financebench_eval_harness.cli import main

    index_dir = _build_rag_modes_index(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json_module.dumps({"question_id": "q1", "question": "What was FY2022 revenue?"}) + "\n",
        encoding="utf-8",
    )
    embedding_cfg = tmp_path / "embedding.yaml"
    embedding_cfg.write_text(
        "embedding:\n  provider: mock\n  model_name: mock-embed\n  batch_size: 32\n",
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"
    run_id = f"{mode}-test"
    run_dir = runs_dir / run_id
    retrieval_results_path = run_dir / "retrieval_results.jsonl"

    retrieve_args = [
        "retrieve",
        "--index-dir", str(index_dir),
        "--questions", str(questions_path),
        "--embedding-config", str(embedding_cfg),
        "--output", str(retrieval_results_path),
        "--mode", retrieve_mode,
    ]
    if retrieve_mode == "hybrid_reranked":
        retrieve_args += ["--reranker-config", "configs/reranker/mock.yaml"]

    retrieve_exit = main(retrieve_args)
    assert retrieve_exit == 0

    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text(
        json_module.dumps({
            "question_id": "q1",
            "question": "What was FY2022 revenue?",
            "gold_answer": "$4.2 billion",
            "evidence": [],
        }) + "\n",
        encoding="utf-8",
    )

    rag_config_path = tmp_path / "rag.yaml"
    rag_config_path.write_text(
        "\n".join([
            "rag_eval:",
            f"  examples_path: {examples_path}",
            f"  retrieval_results_path: {retrieval_results_path}",
            f"  output_dir: {runs_dir}",
            f"  mode: {mode}",
            "  top_k: 3",
            "model:",
            "  provider: mock",
            "  model_name: mock-llm",
            "  temperature: 0.0",
            "  max_tokens: 512",
            "  timeout_seconds: 30",
            "judge:",
            "  enabled: false",
            "  provider: mock",
            "  model_name: mock-judge",
            "  temperature: 0.0",
            "  max_tokens: 256",
            "  timeout_seconds: 30",
            "  prompt:",
            "    id: answer_correctness_v1",
            "    version: v1",
            "    template_path: prompts/judges/answer_correctness_v1.txt",
        ]),
        encoding="utf-8",
    )

    run_rag_exit = main(["run-rag", "--config", str(rag_config_path), "--run-id", run_id])
    assert run_rag_exit == 0
    assert (run_dir / "rag_predictions.jsonl").is_file()

    score_rag_config_path = tmp_path / "rag_score.yaml"
    score_rag_config_path.write_text(
        f"rag_score:\n  run_dir: {run_dir}\n  k: 3\n",
        encoding="utf-8",
    )
    score_rag_exit = main(["score-rag", "--config", str(score_rag_config_path)])
    assert score_rag_exit == 0
    assert (run_dir / "rag_answer_scores.jsonl").is_file()

    return run_dir


def _score_retrieval_and_join(tmp_path: Path, run_dir: Path) -> None:
    from financebench_eval_harness.chunking import ChunkingConfig
    from financebench_eval_harness.cli import main
    from financebench_eval_harness.embedding import EmbeddingConfig
    from financebench_eval_harness.eval_retrieval import score_retrieval_run
    from financebench_eval_harness.pipeline_config import PipelineConfig

    examples_path = tmp_path / "examples.jsonl"
    cfg = PipelineConfig(
        pages_path=tmp_path / "pages.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "index",
        questions_path=examples_path,
        runs_dir=run_dir.parent,
        top_k=3,
        evidence_overlap_threshold=0.5,
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=50, strategy="recursive_text", min_chunk_chars=10),
        embedding=EmbeddingConfig(provider="mock", model_name="mock-embed"),
    )
    summary = score_retrieval_run(cfg, run_dir)
    assert "evidence_text_ndcg@10" in summary
    assert (run_dir / "retrieval_scores.jsonl").is_file()

    analysis_dir = run_dir / "analysis"
    join_exit = main([
        "join-metrics",
        "--retrieval-scores", str(run_dir / "retrieval_scores.jsonl"),
        "--answer-scores", str(run_dir / "rag_answer_scores.jsonl"),
        "--output-dir", str(analysis_dir),
        "--k", "3",
    ])
    assert join_exit == 0
    assert (analysis_dir / "joined_metrics.jsonl").is_file()


def _analyze_grounding_completes(run_dir: Path) -> None:
    from financebench_eval_harness.grounding_analysis import analyze_grounding
    from financebench_eval_harness.grounding_analysis_config import (
        GroundingAnalysisConfig,
        GroundingAnalysisSettings,
    )

    config = GroundingAnalysisConfig(
        settings=GroundingAnalysisSettings(run_dir=run_dir, k=3),
        grounding_judge=None,
    )
    result = analyze_grounding(config)
    assert result.failure_summary_path.is_file()
    assert result.example_count == 1


def test_rag_hybrid_end_to_end_pipeline_completes(tmp_path: Path) -> None:
    run_dir = _run_rag_hybrid_pipeline(tmp_path, mode="rag_hybrid", retrieve_mode="hybrid")
    _score_retrieval_and_join(tmp_path, run_dir)
    _analyze_grounding_completes(run_dir)

    metadata = json.loads((run_dir / "retrieval_run_metadata.json").read_text())
    assert metadata["retrieval_mode"] == "hybrid"
    assert metadata["fusion"] == {
        "k": 60, "candidate_k": 50, "dense_weight": 1.0, "lexical_weight": 1.0,
    }


def test_rag_reranked_end_to_end_pipeline_completes(tmp_path: Path) -> None:
    run_dir = _run_rag_hybrid_pipeline(tmp_path, mode="rag_reranked", retrieve_mode="hybrid_reranked")
    _score_retrieval_and_join(tmp_path, run_dir)
    _analyze_grounding_completes(run_dir)

    metadata = json.loads((run_dir / "retrieval_run_metadata.json").read_text())
    assert metadata["retrieval_mode"] == "hybrid_reranked"
    assert metadata["reranker"] == {
        "provider": "mock", "model_name": "mock-reranker", "adapter_path": None,
    }
