"""Retrieval evaluation — M4.7.

Reads a completed retrieval run, scores every question using the gold evidence
from examples.jsonl, and writes three output files to the run directory:

    retrieval_scores.jsonl      — per-question scoring metrics
    retrieval_summary.json      — aggregate metrics across all questions
    retrieval_eval_config.yaml  — full effective config snapshot for reproducibility
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import yaml

from financebench_eval_harness.pipeline_config import PipelineConfig
from financebench_eval_harness.retrieval_scoring import (
    score_evidence_overlap,
    score_hit_at_k,
    score_rank_metrics,
    summarize_hit_at_k,
    summarize_rank_metrics,
)

__all__ = ["score_retrieval_run", "generate_retrieval_report"]

_DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)


def score_retrieval_run(
    config: PipelineConfig,
    run_dir: Path,
    ks: tuple[int, ...] = _DEFAULT_KS,
) -> dict:
    """Score a completed retrieval run and save output files.

    Reads:
        run_dir/retrieval_results.jsonl  — output from the `retrieve` command
        config.questions_path            — gold examples JSONL (FinanceBench format)

    Writes:
        run_dir/retrieval_scores.jsonl      — one JSON object per question
        run_dir/retrieval_summary.json      — aggregate metrics dict
        run_dir/retrieval_eval_config.yaml  — full effective config snapshot

    Returns the summary dict.
    """
    run_dir = Path(run_dir)

    retrieval_results = _load_retrieval_results(run_dir / "retrieval_results.jsonl")
    gold_examples = _load_gold_examples(config.questions_path)

    per_question: list[dict] = []
    for qid, result in retrieval_results.items():
        gold = gold_examples.get(qid)
        if gold is None:
            continue
        evidence = gold.get("evidence", [])
        row: dict = {
            "question_id": qid,
            "k": config.top_k,
            "gold_doc_name": evidence[0]["doc_name"] if evidence else None,
            "gold_page_num": evidence[0]["gold_page_num"] if evidence else None,
        }
        row.update(score_hit_at_k(result, gold, ks=ks, overlap_threshold=config.evidence_overlap_threshold))
        row.update(score_rank_metrics(result, gold, overlap_threshold=config.evidence_overlap_threshold))
        row.update(score_evidence_overlap(result, gold, threshold=config.evidence_overlap_threshold, top_k=config.top_k))
        per_question.append(row)

    _write_jsonl(run_dir / "retrieval_scores.jsonl", per_question)

    hit_summary = summarize_hit_at_k(per_question, ks=ks)
    rank_summary = summarize_rank_metrics(per_question, ks=ks)
    summary = {**hit_summary, **rank_summary}

    (run_dir / "retrieval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _save_config_snapshot(config, run_dir / "retrieval_eval_config.yaml")

    leaderboard = _build_leaderboard_summary(per_question, hit_summary, rank_summary, config, run_dir.name)
    (run_dir / "retrieval_leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2), encoding="utf-8"
    )

    return summary


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def generate_retrieval_report(
    summary: dict,
    run_id: str,
    config: PipelineConfig,
    *,
    output_dir: Path,
) -> Path:
    """Write a markdown report for a retrieval evaluation run.

    Returns the path to the written report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"retrieval_eval_{run_id}.md"
    report_path.write_text(_render_retrieval_report(summary, run_id, config), encoding="utf-8")
    return report_path


def _render_retrieval_report(summary: dict, run_id: str, config: PipelineConfig) -> str:
    lines: list[str] = [
        f"# Retrieval Evaluation — {run_id}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| top_k | {config.top_k} |",
        f"| evidence_overlap_threshold | {config.evidence_overlap_threshold} |",
        f"| questions_path | {config.questions_path} |",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| example_count | {summary.get('example_count', 0)} |",
    ]
    for key, value in summary.items():
        if key == "example_count":
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        elif value is not None:
            lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def _build_leaderboard_summary(
    per_question: list[dict],
    hit_summary: dict,
    rank_summary: dict,
    config: PipelineConfig,
    run_id: str,
) -> dict:
    k = config.top_k
    n = len(per_question)
    mean_overlap = (
        statistics.mean(r["best_evidence_overlap"] for r in per_question) if per_question else 0.0
    )
    return {
        "run_id": run_id,
        "num_questions": n,
        "k": k,
        "doc_hit@k": hit_summary.get(f"doc_hit@{k}_rate", 0.0),
        "page_hit@k": hit_summary.get(f"page_hit@{k}_rate", 0.0),
        "evidence_text_hit@k": hit_summary.get(f"evidence_text_hit@{k}_rate", 0.0),
        "mrr@k": rank_summary.get(f"doc_mrr@{k}", 0.0),
        "mean_best_evidence_overlap": mean_overlap,
        "median_first_doc_hit_rank": rank_summary.get("doc_median_first_hit_rank"),
        "median_first_page_hit_rank": rank_summary.get("page_median_first_hit_rank"),
        "evidence_overlap_threshold": config.evidence_overlap_threshold,
        "embedding_provider": config.embedding.provider,
        "embedding_model_name": config.embedding.model_name,
        "chunking_strategy": config.chunking.strategy,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
    }


def _load_retrieval_results(path: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            results[obj["question_id"]] = obj
    return results


def _load_gold_examples(path: Path) -> dict[str, dict]:
    """Load examples.jsonl → {question_id: example_dict}.

    Each example_dict retains the full raw structure with the 'evidence' array,
    compatible with the scoring function signatures in retrieval_scoring.py.
    """
    examples: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            examples[obj["question_id"]] = obj
    return examples


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _save_config_snapshot(config: PipelineConfig, path: Path) -> None:
    path.write_text(
        yaml.dump({"retrieval": config.to_dict()}, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
