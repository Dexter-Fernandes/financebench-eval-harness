"""Retrieval evaluation — M4.7–M4.12.

Reads a completed retrieval run, scores every question using the gold evidence
from examples.jsonl, and writes four output files to the run directory:

    retrieval_scores.jsonl      — per-question scoring metrics (including failure_labels)
    retrieval_summary.json      — aggregate metrics and failure label counts
    retrieval_eval_config.yaml  — full effective config snapshot for reproducibility
    retrieval_leaderboard.json  — compact cross-run comparison summary
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

__all__ = ["score_retrieval_run", "generate_retrieval_report", "format_retrieval_failure_report"]

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
        row["failure_labels"] = _compute_failure_labels(row, config.top_k, config.good_rank_threshold)
        per_question.append(row)

    _write_jsonl(run_dir / "retrieval_scores.jsonl", per_question)

    hit_summary = summarize_hit_at_k(per_question, ks=ks)
    rank_summary = summarize_rank_metrics(per_question, ks=ks)
    label_counts = _count_failure_labels(per_question)
    summary = {**hit_summary, **rank_summary, **label_counts}

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
    run_dir: Path | None = None,
) -> Path:
    """Write a markdown report for a retrieval evaluation run.

    When run_dir is provided, generates a rich report with metadata, focused metrics,
    best/worst examples, and failure type analysis. Otherwise renders a simple summary table.

    Returns the path to the written report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"retrieval_eval_{run_id}.md"
    if run_dir is not None:
        leaderboard = json.loads((Path(run_dir) / "retrieval_leaderboard.json").read_text(encoding="utf-8"))
        per_question = _load_scores_jsonl(Path(run_dir) / "retrieval_scores.jsonl")
        gold_examples = _load_gold_examples(config.questions_path)
        content = _render_rich_retrieval_report(leaderboard, per_question, gold_examples, run_id)
    else:
        content = _render_retrieval_report(summary, run_id, config)
    report_path.write_text(content, encoding="utf-8")
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


def _count_failure_labels(per_question: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_question:
        for label in row.get("failure_labels", []):
            counts[label] = counts.get(label, 0) + 1
    return counts


def _compute_failure_labels(row: dict, k: int, good_rank_threshold: int) -> list[str]:
    labels: list[str] = []
    if not row.get(f"doc_hit@{k}"):
        labels.append("wrong_document")
        return labels
    if not row.get(f"page_hit@{k}"):
        labels.append("right_document_wrong_page")
    page_rank = row.get("page_first_hit_rank")
    if page_rank is not None and page_rank > good_rank_threshold:
        labels.append("right_page_low_rank")
    if row.get(f"page_hit@{k}") and not row.get(f"evidence_text_hit@{k}"):
        labels.append("evidence_not_in_chunk")
        if row.get("best_evidence_overlap", 1.0) < 0.1:
            labels.append("table_extraction_issue")
    return labels


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
        "failure_label_counts": _count_failure_labels(per_question),
    }


def _load_scores_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _classify_failures(per_question: list[dict], k: int) -> dict[str, int]:
    doc_miss = page_miss = text_miss = 0
    for row in per_question:
        if not row.get(f"doc_hit@{k}"):
            doc_miss += 1
        elif not row.get(f"page_hit@{k}"):
            page_miss += 1
        elif not row.get(f"evidence_text_hit@{k}"):
            text_miss += 1
    return {"doc_miss": doc_miss, "page_miss": page_miss, "text_miss": text_miss}


def _render_rich_retrieval_report(
    leaderboard: dict,
    per_question: list[dict],
    gold_examples: dict[str, dict],
    run_id: str,
) -> str:
    k = leaderboard.get("k", 5)
    n = leaderboard.get("num_questions", 0)
    lines: list[str] = [f"# Retrieval Evaluation — {run_id}", ""]

    # --- Run Metadata ---
    lines += [
        "## Run Metadata", "",
        "| Setting | Value |", "| --- | --- |",
        f"| run_id | {leaderboard.get('run_id', run_id)} |",
        f"| num_questions | {n} |",
        f"| k | {k} |",
        f"| evidence_overlap_threshold | {leaderboard.get('evidence_overlap_threshold')} |",
        f"| embedding_provider | {leaderboard.get('embedding_provider')} |",
        f"| embedding_model_name | {leaderboard.get('embedding_model_name')} |",
        f"| chunking_strategy | {leaderboard.get('chunking_strategy')} |",
        f"| chunk_size | {leaderboard.get('chunk_size')} |",
        f"| chunk_overlap | {leaderboard.get('chunk_overlap')} |",
        "",
    ]

    # --- Retrieval Metrics ---
    def _fmt(v: object) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines += [
        "## Retrieval Metrics", "",
        "| Metric | Value |", "| --- | ---: |",
        f"| doc_hit@k | {_fmt(leaderboard.get('doc_hit@k'))} |",
        f"| page_hit@k | {_fmt(leaderboard.get('page_hit@k'))} |",
        f"| evidence_text_hit@k | {_fmt(leaderboard.get('evidence_text_hit@k'))} |",
        f"| mrr@k | {_fmt(leaderboard.get('mrr@k'))} |",
        f"| mean_best_evidence_overlap | {_fmt(leaderboard.get('mean_best_evidence_overlap'))} |",
        f"| median_first_doc_hit_rank | {_fmt(leaderboard.get('median_first_doc_hit_rank'))} |",
        f"| median_first_page_hit_rank | {_fmt(leaderboard.get('median_first_page_hit_rank'))} |",
        "",
    ]

    # --- Best Examples ---
    best = [
        r for r in per_question
        if r.get(f"doc_hit@{k}") and r.get(f"page_hit@{k}")
    ]
    best.sort(key=lambda r: r.get("best_evidence_overlap", 0.0), reverse=True)
    lines += ["## Best Examples", ""]
    if best:
        for row in best[:3]:
            qid = row["question_id"]
            gold = gold_examples.get(qid, {})
            lines += [
                f"### {qid}",
                f"- **Company**: {gold.get('company', '—')}",
                f"- **Document**: {row.get('gold_doc_name', '—')} (page {row.get('gold_page_num', '—')})",
                f"- **Question**: {gold.get('question', '—')}",
                f"- **Gold answer**: {gold.get('gold_answer', '—')}",
                f"- **Evidence overlap**: {row.get('best_evidence_overlap', 0.0):.4f}",
                "",
            ]
    else:
        lines += ["_(no questions with both doc and page hit)_", ""]

    # --- Worst Examples ---
    worst = [r for r in per_question if not r.get(f"doc_hit@{k}")]
    worst.sort(key=lambda r: r.get("best_evidence_overlap", 0.0))
    lines += ["## Worst Examples", ""]
    if worst:
        for row in worst[:3]:
            qid = row["question_id"]
            gold = gold_examples.get(qid, {})
            lines += [
                f"### {qid}",
                f"- **Company**: {gold.get('company', '—')}",
                f"- **Document**: {row.get('gold_doc_name', '—')} (page {row.get('gold_page_num', '—')})",
                f"- **Question**: {gold.get('question', '—')}",
                f"- **Gold answer**: {gold.get('gold_answer', '—')}",
                f"- **Best evidence overlap**: {row.get('best_evidence_overlap', 0.0):.4f}",
                "",
            ]
    else:
        lines += ["_(all questions had document hits)_", ""]

    # --- Common Failure Types ---
    label_counts = leaderboard.get("failure_label_counts", {})
    _all_labels = (
        "wrong_document",
        "right_document_wrong_page",
        "right_page_low_rank",
        "evidence_not_in_chunk",
        "table_extraction_issue",
    )
    lines += ["## Common Failure Types", "", "| Label | Count | Rate |", "| --- | ---: | ---: |"]
    for label in _all_labels:
        count = label_counts.get(label, 0)
        rate = f"{count/n:.1%}" if n else "—"
        lines.append(f"| {label} | {count} | {rate} |")
    lines.append("")

    return "\n".join(lines)


def format_retrieval_failure_report(
    config: PipelineConfig,
    run_dir: Path,
    question_id: str,
) -> str:
    """Return a plain-text inspection report for one question from a scored retrieval run.

    Raises KeyError if question_id is not found in retrieval_scores.jsonl.
    """
    run_dir = Path(run_dir)
    results = _load_retrieval_results(run_dir / "retrieval_results.jsonl")
    scores = {r["question_id"]: r for r in _load_scores_jsonl(run_dir / "retrieval_scores.jsonl")}
    gold_examples = _load_gold_examples(config.questions_path)

    if question_id not in scores:
        raise KeyError(f"Question ID not found in retrieval scores: {question_id!r}")

    score_row = scores[question_id]
    gold = gold_examples.get(question_id, {})
    result = results.get(question_id, {})
    evidence_items = gold.get("evidence", [])
    k = score_row.get("k", config.top_k)

    lines: list[str] = []
    lines.append(f"Question ID: {question_id}")
    lines.append("")
    lines.append("Question:")
    lines.append(f"  {gold.get('question', '—')}")
    lines.append("")

    lines.append("Gold:")
    for ev in evidence_items:
        lines.append(f"  Document: {ev.get('doc_name', '—')}")
        lines.append(f"  Page: {ev.get('gold_page_num', '—')}")
        lines.append("  Evidence:")
        lines.append(f"    {ev.get('evidence_text', '—')}")
    lines.append("")

    lines.append("Retrieval Scores:")
    lines.append(f"  doc_hit@k:             {score_row.get(f'doc_hit@{k}', '—')}")
    lines.append(f"  page_hit@k:            {score_row.get(f'page_hit@{k}', '—')}")
    lines.append(f"  evidence_text_hit@k:   {score_row.get(f'evidence_text_hit@{k}', '—')}")
    overlap = score_row.get("best_evidence_overlap", 0.0)
    lines.append(f"  best_evidence_overlap: {overlap:.4f}")
    labels = score_row.get("failure_labels", [])
    lines.append(f"  failure_labels:        {json.dumps(labels)}")
    lines.append("")

    lines.append("Top Retrieved Chunks:")
    for chunk in result.get("retrieved", []):
        chunk_overlap = _chunk_evidence_overlap(chunk.get("text", ""), evidence_items)
        lines.append("")
        lines.append(f"  Rank {chunk.get('rank', '?')}")
        lines.append(f"  Document: {chunk.get('doc_name', '—')}")
        lines.append(f"  Page: {chunk.get('page_num', '—')}")
        lines.append(f"  Score: {chunk.get('score', 0.0):.4f}")
        lines.append(f"  Evidence overlap: {chunk_overlap:.4f}")
        lines.append("  Text:")
        lines.append(f"    {chunk.get('text', '')[:300]}")

    return "\n".join(lines)


def _chunk_evidence_overlap(chunk_text: str, evidence_items: list[dict]) -> float:
    """Compute max overlap between a single chunk and all gold evidence texts."""
    norm_chunk = chunk_text.lower()
    chunk_tokens = set(norm_chunk.split())
    best = 0.0
    for ev in evidence_items:
        ev_text = ev.get("evidence_text", "")
        norm_ev = ev_text.lower()
        if norm_ev and norm_ev in norm_chunk:
            return 1.0
        ev_tokens = set(norm_ev.split())
        if ev_tokens:
            coverage = len(ev_tokens & chunk_tokens) / len(ev_tokens)
            best = max(best, coverage)
    return best


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
