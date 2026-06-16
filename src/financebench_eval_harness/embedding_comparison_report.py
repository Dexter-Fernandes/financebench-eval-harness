"""Markdown report generator for M6 embedding model comparison.

Reads the leaderboard JSON and decision JSON produced by run_embedding_comparison()
and renders a portfolio-ready Markdown report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def generate_embedding_comparison_report(
    run_dir: Path,
    *,
    output_dir: Path,
    dry_run: bool = False,
) -> Path:
    """Generate a Markdown comparison report from a completed comparison run.

    Reads:
        {run_dir}/embedding_leaderboard.json
        {run_dir}/embedding_decision.json
        {run_dir}/config.yaml

    Returns the path to the written .md file.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: list[dict] = json.loads(
        (run_dir / "embedding_leaderboard.json").read_text(encoding="utf-8")
    )
    decision: dict = json.loads(
        (run_dir / "embedding_decision.json").read_text(encoding="utf-8")
    )
    config_snapshot: dict = yaml.safe_load(
        (run_dir / "config.yaml").read_text(encoding="utf-8")
    )

    run_id: str = config_snapshot.get("run_id", run_dir.name)
    report_path = output_dir / f"embedding_comparison_{run_id}.md"
    content = _render_report(run_id, leaderboard, decision, config_snapshot, dry_run=dry_run)
    report_path.write_text(content, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_report(
    run_id: str,
    leaderboard: list[dict],
    decision: dict,
    config: dict,
    *,
    dry_run: bool = False,
) -> str:
    lines: list[str] = []

    lines += [
        f"# Embedding Model Comparison — {run_id}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    if dry_run:
        lines += [
            "> **DRY RUN** — no embeddings were computed. "
            "Leaderboard and decision sections are empty. "
            "Re-run without `--dry-run` to populate results.",
            "",
        ]

    # --- Experiment goal ---
    lines += [
        "## Experiment Goal",
        "",
        "Identify which embedding model retrieves FinanceBench evidence best "
        "under identical chunking and retrieval settings.",
        "",
        "**Primary metric:** `evidence_hit@10` — whether the gold evidence text "
        "appears in the top-10 retrieved chunks.",
        "",
        "**Secondary metrics:** `page_hit@10`, `doc_hit@10`, `MRR@10`, "
        "embedding latency, index size, estimated API cost.",
        "",
    ]

    # --- Dataset and fixed settings ---
    lines += [
        "## Dataset and Fixed Settings",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| chunks_path | `{config.get('chunks_path', '')}` |",
        f"| questions_path | `{config.get('questions_path', '')}` |",
        f"| top_k | {config.get('top_k', '')} |",
        f"| evidence_overlap_threshold | {config.get('evidence_overlap_threshold', '')} |",
        f"| chunk_count | {config.get('chunk_count', '')} |",
        f"| question_count | {config.get('question_count', '')} |",
        f"| corpus_hash | `{config.get('corpus_hash', '')[:16]}…` |",
        "",
    ]

    # --- Candidate models ---
    model_list: list[dict] = config.get("embedding_models", [])
    if model_list:
        lines += [
            "## Candidate Models",
            "",
            "| Model | Provider | Category |",
            "| --- | --- | --- |",
        ]
        for m in model_list:
            lines.append(f"| {m['name']} | {m['provider']} | {m['category']} |")
        lines.append("")

    # --- Retrieval leaderboard ---
    lines += ["## Retrieval Leaderboard", ""]
    if leaderboard:
        cols = ["model", "provider", "evidence_hit@10", "page_hit@10", "doc_hit@10", "mrr@10"]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        lines += [header, sep]
        for row in leaderboard:
            def _fmt(v):
                if isinstance(v, float):
                    return f"{v:.3f}"
                if v is None:
                    return "—"
                return str(v)
            lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
        lines.append("")
    else:
        lines += ["_No successful model runs._", ""]

    # --- Cost and latency ---
    lines += ["## Cost and Latency Comparison", ""]
    if leaderboard:
        cost_cols = ["model", "embedding_latency_s", "retrieval_latency_s", "index_size_mb", "estimated_cost_usd"]
        header = "| " + " | ".join(cost_cols) + " |"
        sep = "| " + " | ".join("---" for _ in cost_cols) + " |"
        lines += [header, sep]
        for row in leaderboard:
            def _fmt2(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                if v is None:
                    return "local/free"
                return str(v)
            lines.append("| " + " | ".join(_fmt2(row.get(c)) for c in cost_cols) + " |")
        lines.append("")

    # --- Best models ---
    lines += [
        "## Best Models by Role",
        "",
        "| Role | Model |",
        "| --- | --- |",
        f"| Best retrieval quality | {decision.get('default_model') or '—'} |",
        f"| Cheap baseline | {decision.get('cheap_baseline') or '—'} |",
        f"| Quality baseline | {decision.get('quality_baseline') or '—'} |",
        f"| Local baseline | {decision.get('local_baseline') or '—'} |",
        f"| Finance-specialized | {decision.get('finance_specialized_candidate') or '—'} |",
        "",
    ]

    # --- Recommended default ---
    lines += [
        "## Recommended Default",
        "",
        f"**default_model:** `{decision.get('default_model') or 'none'}`",
        "",
        f"{decision.get('reason', '')}",
        "",
    ]

    # --- Next steps ---
    lines += [
        "## Next Steps",
        "",
        "- Run M6.11 end-to-end RAG comparison for the top-2 embedding models.",
        "- Investigate retrieval failures for the winning model.",
        "- Consider chunking experiments if `evidence_hit@10` plateaus.",
        "",
    ]

    return "\n".join(lines)


__all__ = ["generate_embedding_comparison_report"]
