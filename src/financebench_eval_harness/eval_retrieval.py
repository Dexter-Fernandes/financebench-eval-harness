"""Retrieval evaluation — M4.7.

Reads a completed retrieval run, scores every question using the gold evidence
from examples.jsonl, and writes three output files to the run directory:

    retrieval_scores.jsonl      — per-question scoring metrics
    retrieval_summary.json      — aggregate metrics across all questions
    retrieval_eval_config.yaml  — full effective config snapshot for reproducibility
"""

from __future__ import annotations

import json
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

__all__ = ["score_retrieval_run"]

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
        row: dict = {"question_id": qid}
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

    return summary


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
