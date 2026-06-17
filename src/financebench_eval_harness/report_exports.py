"""M8.11: CSV and JSON exports for machine-readable report outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from financebench_eval_harness.report_loader import RunData


def export_summary_metrics(run_data: RunData, output_dir: Path) -> Path:
    """Write summary_metrics.json to *output_dir*.

    Aggregates baseline, retrieval, and RAG metrics into one flat JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"run_id": run_data.run_id}

    meta = run_data.run_metadata or {}
    judge = meta.get("judge_summary") or {}
    if judge.get("attempted_count"):
        n = judge["attempted_count"]
        metrics["baseline"] = {
            "mode": meta.get("mode"),
            "model": meta.get("model_name"),
            "questions_evaluated": n,
            "correct_count": judge.get("correct_count", 0),
            "accuracy": round(judge.get("correct_count", 0) / n, 4) if n else None,
        }

    ret = run_data.retrieval_summary or {}
    if ret:
        metrics["retrieval"] = {
            k: v for k, v in ret.items()
            if isinstance(v, (int, float, str)) or v is None
        }

    joined = run_data.joined_summary or {}
    if joined:
        n_j = joined.get("example_count", 0)
        metrics["rag"] = {
            "questions_evaluated": n_j,
            "answer_correct_count": joined.get("answer_correct_count", 0),
            "accuracy": round(joined.get("answer_correct_count", 0) / n_j, 4) if n_j else None,
            "retrieval_hit_answer_correct": joined.get("retrieval_hit_answer_correct"),
            "retrieval_miss_answer_wrong": joined.get("retrieval_miss_answer_wrong"),
        }

    fs = run_data.failure_summary or {}
    if fs:
        metrics["failure_summary"] = dict(fs)

    out_path = output_dir / "summary_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def export_failure_breakdown_csv(run_data: RunData, output_dir: Path) -> Path:
    """Write failure_breakdown.csv to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = run_data.failure_summary or {}
    total = fs.get("total", 0)

    rows = [
        ("retrieval_failure", fs.get("retrieval_failure_count", 0)),
        ("generation_failure", fs.get("generation_failure_count", 0)),
        ("citation_failure", fs.get("citation_failure_count", 0)),
        ("hallucination_under_refusal", fs.get("hallucination_under_refusal_count", 0)),
        ("over_refusal", fs.get("over_refusal_count", 0)),
        ("no_failure", fs.get("no_failure_count", 0)),
    ]

    out_path = output_dir / "failure_breakdown.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_type", "count", "rate"])
        for label, count in rows:
            rate = f"{count / total:.4f}" if total else ""
            writer.writerow([label, count, rate])

    return out_path


def export_leaderboard_csv(
    leaderboard: list[dict],
    output_dir: Path,
    run_id: str,
) -> Path:
    """Write leaderboard.csv sorted by evidence_hit@10 descending."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_lb = sorted(
        leaderboard,
        key=lambda r: r.get("evidence_hit@10") or 0.0,
        reverse=True,
    )

    if not sorted_lb:
        out_path = output_dir / f"leaderboard_{run_id}.csv"
        out_path.write_text("model,provider,evidence_hit@10\n", encoding="utf-8")
        return out_path

    fieldnames = list(sorted_lb[0].keys())
    out_path = output_dir / f"leaderboard_{run_id}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_lb)

    return out_path


__all__ = [
    "export_failure_breakdown_csv",
    "export_leaderboard_csv",
    "export_summary_metrics",
]
