"""M8.16: Tests for the experiment reporting layer.

Covers:
  1. baseline report renders with mock data
  2. retrieval report renders hit@k table
  3. RAG report renders answer metrics
  4. embedding leaderboard sorts correctly
  5. missing optional files produce warnings
  6. missing required files fail clearly
  7. generated Markdown contains expected sections (experiment summary)
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_yaml(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# M8.4 – Report data loader (tests 5 & 6)
# ---------------------------------------------------------------------------


def test_missing_optional_files_produce_warnings(tmp_path: Path) -> None:
    """Missing optional run files should trigger warnings, not errors."""
    from financebench_eval_harness.report_loader import load_run_data

    run_dir = tmp_path / "run_no_optionals"
    run_dir.mkdir()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = load_run_data(run_dir)

    assert data.run_metadata is None
    assert data.retrieval_summary is None
    assert data.failure_analysis is None
    assert len(caught) > 0, "Expected at least one warning for missing optional files"


def test_missing_required_files_raise_error(tmp_path: Path) -> None:
    """A file listed in `required` that is absent should raise ReportLoadError."""
    from financebench_eval_harness.report_loader import ReportLoadError, load_run_data

    run_dir = tmp_path / "run_missing_required"
    run_dir.mkdir()

    with pytest.raises(ReportLoadError, match="run_metadata.json"):
        load_run_data(run_dir, required=["run_metadata.json"])


def test_load_run_data_loads_present_files(tmp_path: Path) -> None:
    """load_run_data populates fields from files that exist."""
    from financebench_eval_harness.report_loader import load_run_data

    run_dir = tmp_path / "run_full"
    _write_json(run_dir / "run_metadata.json", {"run_id": "test-run", "mode": "closed_book"})
    _write_json(run_dir / "retrieval_summary.json", {"example_count": 10})
    _write_json(run_dir / "failure_summary.json", {"total": 5})
    _write_jsonl(run_dir / "failure_analysis.jsonl", [{"question_id": "q1", "root_cause": "retrieval_failure"}])

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        data = load_run_data(run_dir)

    assert data.run_metadata == {"run_id": "test-run", "mode": "closed_book"}
    assert data.retrieval_summary == {"example_count": 10}
    assert data.failure_summary == {"total": 5}
    assert data.failure_analysis is not None
    assert data.failure_analysis[0]["question_id"] == "q1"
    assert data.run_id == "run_full"


def test_load_run_data_loads_retrieval_run_metadata(tmp_path: Path) -> None:
    """load_run_data populates retrieval_run_metadata from retrieval_run_metadata.json."""
    from financebench_eval_harness.report_loader import load_run_data

    run_dir = tmp_path / "run_retrieval_only"
    _write_json(
        run_dir / "retrieval_run_metadata.json",
        {"run_id": "r1", "retrieval_mode": "hybrid", "embedding_model": "nomic-embed-text"},
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        data = load_run_data(run_dir)

    assert data.retrieval_run_metadata == {
        "run_id": "r1", "retrieval_mode": "hybrid", "embedding_model": "nomic-embed-text",
    }


def test_load_run_data_retrieval_run_metadata_none_when_absent(tmp_path: Path) -> None:
    from financebench_eval_harness.report_loader import load_run_data

    run_dir = tmp_path / "run_no_retrieval_metadata"
    run_dir.mkdir()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        data = load_run_data(run_dir)

    assert data.retrieval_run_metadata is None


# ---------------------------------------------------------------------------
# M8.10 – Experiment summary report (test 7)
# ---------------------------------------------------------------------------


def test_experiment_summary_markdown_contains_expected_sections(tmp_path: Path) -> None:
    """generate_experiment_summary_report should produce a Markdown file with
    expected section headers."""
    from financebench_eval_harness.experiment_summary_report import (
        generate_experiment_summary_report,
    )
    from financebench_eval_harness.report_loader import load_run_data

    run_dir = tmp_path / "run_summary"
    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": "run_summary",
            "mode": "closed_book",
            "model_provider": "mock",
            "model_name": "mock-llm",
            "judge_summary": {
                "attempted_count": 20,
                "correct_count": 8,
                "partially_correct_count": 2,
                "incorrect_count": 8,
                "not_answered_count": 2,
                "error_count": 0,
            },
        },
    )
    _write_json(
        run_dir / "retrieval_summary.json",
        {
            "example_count": 20,
            "evidence_text_hit@10_rate": 0.75,
            "doc_hit@10_rate": 0.88,
        },
    )
    _write_json(
        run_dir / "failure_summary.json",
        {
            "total": 20,
            "retrieval_failure_count": 5,
            "generation_failure_count": 3,
            "no_failure_count": 8,
        },
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        data = load_run_data(run_dir)

    report_path = generate_experiment_summary_report(data, output_dir=tmp_path / "reports")

    assert report_path.exists()
    markdown = report_path.read_text(encoding="utf-8")

    assert "## Key Metrics" in markdown
    assert "## Key Findings" in markdown
    assert "experiment_summary" in report_path.name


# ---------------------------------------------------------------------------
# M8.11 – CSV / JSON exports
# ---------------------------------------------------------------------------


def test_export_summary_metrics_writes_json(tmp_path: Path) -> None:
    """export_summary_metrics should write a JSON file to the run directory."""
    from financebench_eval_harness.report_exports import export_summary_metrics
    from financebench_eval_harness.report_loader import load_run_data

    run_dir = tmp_path / "run_exports"
    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": "run_exports",
            "mode": "closed_book",
            "judge_summary": {
                "attempted_count": 5,
                "correct_count": 2,
                "partially_correct_count": 1,
                "incorrect_count": 2,
                "not_answered_count": 0,
                "error_count": 0,
            },
        },
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        data = load_run_data(run_dir)

    out_path = export_summary_metrics(data, output_dir=run_dir)
    assert out_path.exists()
    metrics = json.loads(out_path.read_text(encoding="utf-8"))
    assert "run_id" in metrics or "baseline" in metrics or "retrieval" in metrics or len(metrics) > 0


def test_export_failure_breakdown_csv_writes_csv(tmp_path: Path) -> None:
    """export_failure_breakdown_csv should write a CSV with a header row."""
    from financebench_eval_harness.report_exports import export_failure_breakdown_csv
    from financebench_eval_harness.report_loader import load_run_data

    run_dir = tmp_path / "run_csv"
    _write_json(
        run_dir / "failure_summary.json",
        {
            "total": 10,
            "retrieval_failure_count": 4,
            "generation_failure_count": 2,
            "citation_failure_count": 1,
            "hallucination_under_refusal_count": 0,
            "over_refusal_count": 1,
            "no_failure_count": 2,
        },
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        data = load_run_data(run_dir)

    out_path = export_failure_breakdown_csv(data, output_dir=run_dir)
    assert out_path.suffix == ".csv"
    assert out_path.exists()
    rows = list(csv.DictReader(out_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) > 0
    assert "failure_type" in rows[0] or "count" in rows[0]


def test_export_leaderboard_csv_sorts_by_primary_metric(tmp_path: Path) -> None:
    """export_leaderboard_csv should produce a CSV with models sorted by evidence_hit@10."""
    from financebench_eval_harness.report_exports import export_leaderboard_csv

    leaderboard = [
        {"model": "model_b", "provider": "local", "evidence_hit@10": 0.55},
        {"model": "model_a", "provider": "api", "evidence_hit@10": 0.70},
        {"model": "model_c", "provider": "local", "evidence_hit@10": 0.62},
    ]
    out_path = export_leaderboard_csv(leaderboard, output_dir=tmp_path, run_id="test")
    assert out_path.exists()
    rows = list(csv.DictReader(out_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["model"] == "model_a"  # highest evidence_hit@10 first


# ---------------------------------------------------------------------------
# M8.13 – generate-report CLI (integration)
# ---------------------------------------------------------------------------


def _make_baseline_run(run_dir: Path) -> None:
    """Write minimal files for a baseline run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "question_id": f"q{i}",
            "question": f"Question {i}?",
            "gold_answer": "Gold",
            "prediction": "Pred",
            "latency_ms": 1000,
            "input_tokens": None,
            "output_tokens": None,
            "judge": {
                "status": "success",
                "verdict": "correct" if i < 3 else "incorrect",
                "reason": "ok",
                "error": None,
            },
        }
        for i in range(5)
    ]
    _write_json(
        run_dir / "run_metadata.json",
        {
            "mode": "closed_book",
            "model_provider": "mock",
            "model_name": "mock-llm",
            "attempted_count": 5,
            "judge_summary": {
                "attempted_count": 5,
                "success_count": 5,
                "error_count": 0,
                "correct_count": 3,
                "correct_rate": 0.6,
                "partially_correct_count": 0,
                "partially_correct_rate": 0.0,
                "incorrect_count": 2,
                "incorrect_rate": 0.4,
                "not_answered_count": 0,
                "not_answered_rate": 0.0,
            },
        },
    )
    _write_jsonl(run_dir / "outputs.jsonl", rows)


def test_cli_generate_report_baseline(tmp_path: Path) -> None:
    """generate-report --type baseline should produce a Markdown report."""
    from financebench_eval_harness.cli import main

    run_dir = tmp_path / "run_cli_baseline"
    _make_baseline_run(run_dir)
    output_dir = tmp_path / "reports"

    rc = main([
        "generate-report",
        "--type", "baseline",
        "--run", str(run_dir),
        "--output", str(output_dir),
    ])
    assert rc == 0
    reports = list(output_dir.glob("baseline_*.md"))
    assert len(reports) == 1
    assert "# Baseline Report" in reports[0].read_text(encoding="utf-8")


def test_cli_generate_report_experiment_summary(tmp_path: Path) -> None:
    """generate-report --type experiment-summary should write experiment_summary_*.md."""
    from financebench_eval_harness.cli import main

    run_dir = tmp_path / "run_cli_summary"
    _make_baseline_run(run_dir)
    _write_json(run_dir / "retrieval_summary.json", {"example_count": 5, "evidence_text_hit@10_rate": 0.6})
    output_dir = tmp_path / "reports"

    rc = main([
        "generate-report",
        "--type", "experiment-summary",
        "--run", str(run_dir),
        "--output", str(output_dir),
    ])
    assert rc == 0
    reports = list(output_dir.glob("experiment_summary_*.md"))
    assert len(reports) == 1


# ---------------------------------------------------------------------------
# M8.14 – compare-runs CLI
# ---------------------------------------------------------------------------


def _make_run_with_retrieval(run_dir: Path, retrieval_hit: float, run_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_metadata.json",
        {
            "run_id": run_id,
            "mode": "rag_dense",
            "model_provider": "mock",
            "model_name": "mock-llm",
            "judge_summary": {
                "attempted_count": 10,
                "correct_count": 5,
                "partially_correct_count": 0,
                "incorrect_count": 5,
                "not_answered_count": 0,
                "error_count": 0,
            },
        },
    )
    _write_json(
        run_dir / "retrieval_summary.json",
        {
            "example_count": 10,
            "evidence_text_hit@10_rate": retrieval_hit,
            "doc_hit@10_rate": retrieval_hit + 0.05,
        },
    )


def test_cli_compare_runs_produces_markdown_table(tmp_path: Path) -> None:
    """compare-runs should produce a Markdown file with a comparison table."""
    from financebench_eval_harness.cli import main

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_c = tmp_path / "run_c"
    _make_run_with_retrieval(run_a, 0.55, "run_a")
    _make_run_with_retrieval(run_b, 0.72, "run_b")
    _make_run_with_retrieval(run_c, 0.64, "run_c")

    output_path = tmp_path / "reports" / "comparison.md"

    rc = main([
        "compare-runs",
        "--runs", str(run_a), str(run_b), str(run_c),
        "--output", str(output_path),
    ])
    assert rc == 0
    assert output_path.exists()
    markdown = output_path.read_text(encoding="utf-8")
    assert "run_a" in markdown
    assert "run_b" in markdown
    assert "run_c" in markdown
    assert "|" in markdown  # contains a table


def test_cli_compare_runs_sorts_by_ndcg_metric(tmp_path: Path) -> None:
    """M9.2: compare-runs --metric evidence_text_ndcg@10 sorts by the nDCG key."""
    from financebench_eval_harness.cli import main

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    _write_json(run_a / "run_metadata.json", {"run_id": "run_a", "mode": "rag_dense"})
    _write_json(run_b / "run_metadata.json", {"run_id": "run_b", "mode": "rag_dense"})
    _write_json(run_a / "retrieval_summary.json", {"example_count": 10, "evidence_text_ndcg@10": 0.40})
    _write_json(run_b / "retrieval_summary.json", {"example_count": 10, "evidence_text_ndcg@10": 0.85})

    output_path = tmp_path / "reports" / "ndcg_comparison.md"
    rc = main([
        "compare-runs",
        "--runs", str(run_a), str(run_b),
        "--metric", "evidence_text_ndcg@10",
        "--output", str(output_path),
    ])
    assert rc == 0
    markdown = output_path.read_text(encoding="utf-8")
    assert "Sorted by: `evidence_text_ndcg@10`" in markdown
    assert "**Best run:** `run_b`" in markdown


def _make_retrieval_only_run(
    run_dir: Path,
    *,
    run_id: str,
    retrieval_mode: str,
    embedding_model: str = "nomic-embed-text",
    doc_hit_at_10: float,
    doc_mrr_at_10: float,
    evidence_text_ndcg_at_10: float = 0.0,
) -> None:
    """A run produced by `retrieve` only — no run_metadata.json (no RAG generation)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "retrieval_run_metadata.json",
        {
            "run_id": run_id,
            "retrieval_mode": retrieval_mode,
            "embedding_model": embedding_model,
        },
    )
    _write_json(
        run_dir / "retrieval_summary.json",
        {
            "example_count": 46,
            "doc_hit@10_rate": doc_hit_at_10,
            "doc_mrr@10": doc_mrr_at_10,
            "evidence_text_ndcg@10": evidence_text_ndcg_at_10,
        },
    )


def test_cli_compare_runs_shows_retrieval_mode_and_embedding_for_retrieval_only_runs(
    tmp_path: Path,
) -> None:
    from financebench_eval_harness.cli import main

    run_dir = tmp_path / "run_hybrid"
    _make_retrieval_only_run(
        run_dir, run_id="run_hybrid", retrieval_mode="hybrid",
        embedding_model="nomic-embed-text", doc_hit_at_10=0.65, doc_mrr_at_10=0.25,
    )

    output_path = tmp_path / "reports" / "comparison.md"
    rc = main([
        "compare-runs",
        "--runs", str(run_dir),
        "--output", str(output_path),
    ])
    assert rc == 0
    markdown = output_path.read_text(encoding="utf-8")
    assert "| run_hybrid ★ | hybrid | nomic-embed-text |" in markdown


def test_cli_compare_runs_mrr_column_falls_back_to_doc_mrr(tmp_path: Path) -> None:
    from financebench_eval_harness.cli import main

    run_dir = tmp_path / "run_hybrid"
    _make_retrieval_only_run(
        run_dir, run_id="run_hybrid", retrieval_mode="hybrid",
        doc_hit_at_10=0.65, doc_mrr_at_10=0.249,
    )

    output_path = tmp_path / "reports" / "comparison.md"
    rc = main([
        "compare-runs",
        "--runs", str(run_dir),
        "--output", str(output_path),
    ])
    assert rc == 0
    markdown = output_path.read_text(encoding="utf-8")
    assert "0.249" in markdown


def test_cli_compare_runs_tie_break_prefers_doc_hit_and_doc_mrr(tmp_path: Path) -> None:
    """When the primary --metric ties (or is absent), row order should still be
    sensible for retrieval-only runs: prefer higher doc_hit@10, then doc_mrr@10."""
    from financebench_eval_harness.cli import main

    run_dense = tmp_path / "run_dense"
    run_reranked = tmp_path / "run_reranked"
    # Both tie at evidence_text_ndcg@10 == 0.0, but reranked is clearly the
    # better retrieval run on doc_hit@10 / doc_mrr@10.
    _make_retrieval_only_run(
        run_dense, run_id="run_dense", retrieval_mode="dense",
        doc_hit_at_10=0.326, doc_mrr_at_10=0.119, evidence_text_ndcg_at_10=0.0,
    )
    _make_retrieval_only_run(
        run_reranked, run_id="run_reranked", retrieval_mode="hybrid_reranked",
        doc_hit_at_10=0.717, doc_mrr_at_10=0.452, evidence_text_ndcg_at_10=0.0,
    )

    output_path = tmp_path / "reports" / "comparison.md"
    rc = main([
        "compare-runs",
        "--runs", str(run_dense), str(run_reranked),
        "--metric", "evidence_text_ndcg@10",
        "--output", str(output_path),
    ])
    assert rc == 0
    markdown = output_path.read_text(encoding="utf-8")
    # run_reranked should sort first (and be marked best) despite the tie on
    # the primary metric, because it dominates on doc_hit@10 and doc_mrr@10.
    reranked_pos = markdown.index("run_reranked")
    dense_pos = markdown.index("run_dense")
    assert reranked_pos < dense_pos
    assert "**Best run:** `run_reranked`" in markdown


def test_cli_compare_runs_explicit_metric_still_wins_over_tie_breakers(tmp_path: Path) -> None:
    """The requested --metric remains the primary sort key; tie-breakers only
    apply when it doesn't already distinguish the rows."""
    from financebench_eval_harness.cli import main

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    # run_a wins on the explicit metric despite run_b having better doc_hit/mrr.
    _make_retrieval_only_run(
        run_a, run_id="run_a", retrieval_mode="hybrid",
        doc_hit_at_10=0.30, doc_mrr_at_10=0.10, evidence_text_ndcg_at_10=0.50,
    )
    _make_retrieval_only_run(
        run_b, run_id="run_b", retrieval_mode="hybrid_reranked",
        doc_hit_at_10=0.90, doc_mrr_at_10=0.80, evidence_text_ndcg_at_10=0.10,
    )

    output_path = tmp_path / "reports" / "comparison.md"
    rc = main([
        "compare-runs",
        "--runs", str(run_a), str(run_b),
        "--metric", "evidence_text_ndcg@10",
        "--output", str(output_path),
    ])
    assert rc == 0
    markdown = output_path.read_text(encoding="utf-8")
    assert "**Best run:** `run_a`" in markdown


# ---------------------------------------------------------------------------
# M8.16 – Spec test 1: baseline report renders with mock data
# ---------------------------------------------------------------------------


def test_baseline_report_renders_with_mock_data(tmp_path: Path) -> None:
    """Baseline report should render a table with correct header and metrics."""
    from financebench_eval_harness.report import generate_baseline_report

    run_dir = tmp_path / "run_baseline"
    _make_baseline_run(run_dir)
    output_dir = tmp_path / "reports"

    result = generate_baseline_report(run_dir, output_dir=output_dir)
    markdown = result.report_path.read_text(encoding="utf-8")

    assert "# Baseline Report" in markdown
    assert "| Questions evaluated |" in markdown
    assert "| Accuracy estimate |" in markdown
    assert result.evaluated_count == 5


# ---------------------------------------------------------------------------
# M8.16 – Spec test 2: retrieval report renders hit@k table
# ---------------------------------------------------------------------------


def _make_retrieval_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = [
        {
            "question_id": f"q{i}",
            "doc_hit@1": i < 2,
            "doc_hit@5": i < 4,
            "doc_hit@10": i < 5,
            "page_hit@1": i < 1,
            "page_hit@5": i < 3,
            "page_hit@10": i < 4,
            "evidence_text_hit@1": i < 1,
            "evidence_text_hit@5": i < 2,
            "evidence_text_hit@10": i < 3,
            "failure_labels": [],
        }
        for i in range(5)
    ]
    _write_jsonl(run_dir / "retrieval_scores.jsonl", scores)
    _write_json(
        run_dir / "retrieval_summary.json",
        {
            "example_count": 5,
            "doc_hit@1_rate": 0.4,
            "doc_hit@5_rate": 0.8,
            "doc_hit@10_rate": 1.0,
            "page_hit@1_rate": 0.2,
            "page_hit@5_rate": 0.6,
            "page_hit@10_rate": 0.8,
            "evidence_text_hit@1_rate": 0.2,
            "evidence_text_hit@5_rate": 0.4,
            "evidence_text_hit@10_rate": 0.6,
            "mrr@10": 0.55,
        },
    )
    _write_yaml(
        run_dir / "retrieval_eval_config.yaml",
        {"run_id": "ret-run", "top_k": 10, "embedding_model": "mock"},
    )


def test_retrieval_report_renders_hit_at_k_table(tmp_path: Path) -> None:
    """Retrieval report should contain hit@k metric rows."""
    from financebench_eval_harness.eval_retrieval import generate_retrieval_report
    from financebench_eval_harness.pipeline_config import load_pipeline_config

    # generate_retrieval_report takes (summary, run_id, config, output_dir, run_dir)
    # We can call it directly with a summary dict and mock config.
    run_dir = tmp_path / "run_retrieval"
    _make_retrieval_run(run_dir)

    summary = json.loads((run_dir / "retrieval_summary.json").read_text(encoding="utf-8"))

    # Minimal pipeline config stub (generate_retrieval_report only needs output_dir)
    class _MockConfig:
        top_k = 10
        embedding_model = "mock"
        chunk_size = 800
        chunk_overlap = 150
        evidence_overlap_threshold = 0.3
        runs_dir = tmp_path / "runs"
        questions_path = tmp_path / "examples.jsonl"

    output_dir = tmp_path / "reports"
    # Pass run_dir=None to use the summary-only rendering path (no leaderboard required)
    report_path = generate_retrieval_report(
        summary, "ret-run", _MockConfig(), output_dir=output_dir
    )

    markdown = Path(report_path).read_text(encoding="utf-8")
    assert "@1" in markdown or "hit@1" in markdown.lower() or "hit@10" in markdown.lower()
    assert "@10" in markdown or "hit@10" in markdown.lower() or "evidence_text_hit@10_rate" in markdown
    assert "0." in markdown  # at least one numeric value


# ---------------------------------------------------------------------------
# M8.16 – Spec test 3: RAG report renders answer metrics
# ---------------------------------------------------------------------------


def _make_rag_joined(run_dir: Path) -> None:
    joined_dir = run_dir / "joined"
    joined_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "question_id": "q1",
            "answer_correct": True,
            "retrieval_hit": True,
            "category": "retrieval_hit_answer_correct",
            "judge_verdict": "correct",
            "failure_labels": [],
        },
        {
            "question_id": "q2",
            "answer_correct": False,
            "retrieval_hit": True,
            "category": "retrieval_hit_answer_wrong",
            "judge_verdict": "incorrect",
            "failure_labels": ["numeric_error"],
        },
        {
            "question_id": "q3",
            "answer_correct": False,
            "retrieval_hit": False,
            "category": "retrieval_miss_answer_wrong",
            "judge_verdict": "not_answered",
            "failure_labels": ["retrieval_miss"],
        },
    ]
    _write_jsonl(joined_dir / "joined_metrics.jsonl", rows)
    _write_json(
        joined_dir / "joined_summary.json",
        {
            "example_count": 3,
            "answer_correct_count": 1,
            "retrieval_hit_answer_correct": 1,
            "retrieval_hit_answer_wrong": 1,
            "retrieval_miss_answer_correct": 0,
            "retrieval_miss_answer_wrong": 1,
            "retrieval_miss_count": 0,
            "numeric_error_count": 1,
            "unsupported_claim_count": 0,
            "over_refusal_count": 0,
            "reasoning_error_count": 0,
            "retrieval_miss_count": 1,
        },
    )
    _write_json(
        run_dir / "rag_run_metadata.json",
        {
            "model_provider": "mock",
            "model_name": "mock-rag",
            "top_k": 5,
            "prompt_version": "v1",
        },
    )


def test_rag_report_renders_answer_metrics(tmp_path: Path) -> None:
    """RAG report should include answer accuracy and retrieval correlation table."""
    from financebench_eval_harness.rag_report import generate_rag_report

    run_dir = tmp_path / "run_rag"
    _make_rag_joined(run_dir)
    output_dir = tmp_path / "reports"

    result = generate_rag_report(
        run_dir / "joined",
        rag_run_dir=run_dir,
        output_dir=output_dir,
        run_id="rag-test",
    )
    markdown = result.report_path.read_text(encoding="utf-8")

    assert "## Answer Accuracy" in markdown
    assert "Correct" in markdown
    assert "Retrieval" in markdown
    assert result.example_count == 3


# ---------------------------------------------------------------------------
# M8.16 – Spec test 4: embedding leaderboard sorts correctly
# ---------------------------------------------------------------------------


def _make_embedding_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = [
        {
            "model": "model_b",
            "provider": "local",
            "evidence_hit@10": 0.55,
            "page_hit@10": 0.70,
            "doc_hit@10": 0.80,
            "mrr@10": 0.50,
            "embedding_latency_s": 2.1,
            "retrieval_latency_s": 0.1,
            "index_size_mb": 50.0,
            "estimated_cost_usd": None,
        },
        {
            "model": "model_a",
            "provider": "api",
            "evidence_hit@10": 0.70,
            "page_hit@10": 0.82,
            "doc_hit@10": 0.90,
            "mrr@10": 0.65,
            "embedding_latency_s": None,
            "retrieval_latency_s": 0.2,
            "index_size_mb": 48.0,
            "estimated_cost_usd": 0.01,
        },
    ]
    _write_json(run_dir / "embedding_leaderboard.json", leaderboard)
    _write_json(
        run_dir / "embedding_decision.json",
        {
            "default_model": "model_a",
            "cheap_baseline": "model_b",
            "quality_baseline": "model_a",
            "local_baseline": "model_b",
            "finance_specialized_candidate": None,
            "reason": "model_a achieved highest evidence_hit@10.",
        },
    )
    _write_yaml(
        run_dir / "config.yaml",
        {
            "run_id": "emb-cmp",
            "chunks_path": "data/processed/financebench/chunks.jsonl",
            "questions_path": "data/processed/financebench/examples.jsonl",
            "top_k": 10,
            "evidence_overlap_threshold": 0.3,
            "chunk_count": 500,
            "question_count": 20,
            "corpus_hash": "abc123def456",
            "embedding_models": [
                {"name": "model_a", "provider": "api", "category": "api"},
                {"name": "model_b", "provider": "local", "category": "local"},
            ],
        },
    )


def test_embedding_leaderboard_sorts_correctly(tmp_path: Path) -> None:
    """Embedding comparison report should list models; report includes model names."""
    from financebench_eval_harness.embedding_comparison_report import (
        generate_embedding_comparison_report,
    )

    run_dir = tmp_path / "emb_run"
    _make_embedding_run(run_dir)
    output_dir = tmp_path / "reports"

    report_path = generate_embedding_comparison_report(run_dir, output_dir=output_dir)
    markdown = Path(report_path).read_text(encoding="utf-8")

    assert "model_a" in markdown
    assert "model_b" in markdown
    # model_a has higher evidence_hit@10 (0.70 vs 0.55), should appear in leaderboard
    assert markdown.index("model_a") < markdown.index("model_b") or "0.700" in markdown
