"""M8.10: Experiment summary report — portfolio-friendly one-pager."""

from __future__ import annotations

from pathlib import Path

from financebench_eval_harness.report_loader import RunData


def generate_experiment_summary_report(
    run_data: RunData,
    *,
    output_dir: Path = Path("reports"),
) -> Path:
    """Generate reports/experiment_summary_<run_id>.md from *run_data*.

    Sections are skipped gracefully when the underlying data is absent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"experiment_summary_{run_data.run_id}.md"
    report_path.write_text(_render(run_data), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(d: RunData) -> str:
    lines: list[str] = [f"# Experiment Summary — {d.run_id}", ""]

    lines += [
        "## Project Objective",
        "",
        "Evaluate how well a RAG system answers financial questions from FinanceBench, "
        "measuring answer correctness, evidence grounding, and retrieval quality.",
        "",
    ]

    lines += _dataset_section(d)
    lines += _setup_section(d)
    lines += _key_metrics_section(d)
    lines += _key_findings_section(d)
    lines += _limitations_section(d)
    lines += _next_steps_section(d)

    return "\n".join(lines).rstrip() + "\n"


def _dataset_section(d: RunData) -> list[str]:
    meta = d.run_metadata or {}
    n = _count(meta) or _count(d.retrieval_summary or {}) or "unknown"
    return [
        "## Dataset",
        "",
        f"| Field | Value |",
        f"| --- | --- |",
        f"| Dataset | FinanceBench |",
        f"| Questions evaluated | {n} |",
        "",
    ]


def _setup_section(d: RunData) -> list[str]:
    meta = d.run_metadata or {}
    rag_meta = d.rag_run_metadata or {}
    lines = [
        "## Experimental Setup",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Evaluation mode | {meta.get('mode', '—')} |",
        f"| Generator model | {rag_meta.get('model_provider', meta.get('model_provider', '—'))} / "
        f"{rag_meta.get('model_name', meta.get('model_name', '—'))} |",
    ]
    if rag_meta.get("top_k") is not None:
        lines.append(f"| Retrieval top-k | {rag_meta['top_k']} |")
    if d.config:
        emb = d.config.get("embedding_model") or d.config.get("embedding", {})
        if isinstance(emb, str):
            lines.append(f"| Embedding model | {emb} |")
    lines.append("")
    return lines


def _key_metrics_section(d: RunData) -> list[str]:
    lines = ["## Key Metrics", "", "| Metric | Value |", "| --- | --- |"]

    # Baseline accuracy
    meta = d.run_metadata or {}
    judge = meta.get("judge_summary") or {}
    attempted = judge.get("attempted_count") or 0
    if attempted:
        correct = judge.get("correct_count", 0)
        lines.append(f"| Baseline accuracy | {_pct(correct, attempted)} |")

    # Retrieval quality
    ret = d.retrieval_summary or {}
    if ret.get("evidence_text_hit@10_rate") is not None:
        lines.append(f"| Evidence hit@10 | {_fmt(ret['evidence_text_hit@10_rate'])} |")
    if ret.get("doc_hit@10_rate") is not None:
        lines.append(f"| Doc hit@10 | {_fmt(ret['doc_hit@10_rate'])} |")

    # RAG accuracy (from joined summary)
    joined = d.joined_summary or {}
    n_joined = joined.get("example_count", 0)
    if n_joined:
        correct_rag = joined.get("answer_correct_count", 0)
        lines.append(f"| RAG answer accuracy | {_pct(correct_rag, n_joined)} |")
        rhac = joined.get("retrieval_hit_answer_correct", 0)
        rh_total = rhac + joined.get("retrieval_hit_answer_wrong", 0)
        if rh_total:
            lines.append(f"| Correct when evidence retrieved | {_pct(rhac, rh_total)} |")

    # Failure breakdown (from failure_summary)
    fs = d.failure_summary or {}
    total_fs = fs.get("total", 0)
    if total_fs:
        ret_fail = fs.get("retrieval_failure_count", 0)
        gen_fail = fs.get("generation_failure_count", 0)
        lines.append(f"| Retrieval failures | {ret_fail} / {total_fs} ({_pct(ret_fail, total_fs)}) |")
        lines.append(f"| Generation failures | {gen_fail} / {total_fs} ({_pct(gen_fail, total_fs)}) |")

    lines.append("")
    return lines


def _key_findings_section(d: RunData) -> list[str]:
    findings: list[str] = []

    # Compare baseline vs RAG
    meta = d.run_metadata or {}
    judge = meta.get("judge_summary") or {}
    attempted = judge.get("attempted_count") or 0
    baseline_acc = judge.get("correct_count", 0) / attempted if attempted else None

    joined = d.joined_summary or {}
    n_joined = joined.get("example_count", 0)
    rag_acc = joined.get("answer_correct_count", 0) / n_joined if n_joined else None

    if baseline_acc is not None and rag_acc is not None:
        if rag_acc > baseline_acc:
            findings.append(
                f"RAG improved answer accuracy over closed-book baseline "
                f"({_fmt(rag_acc)} vs {_fmt(baseline_acc)})."
            )
        else:
            findings.append(
                f"RAG did not improve over closed-book baseline "
                f"({_fmt(rag_acc)} vs {_fmt(baseline_acc)})."
            )

    # Retrieval quality
    ret = d.retrieval_summary or {}
    if ret.get("evidence_text_hit@10_rate") is not None:
        ev = ret["evidence_text_hit@10_rate"]
        findings.append(
            f"Evidence retrieval hit@10 was {_fmt(ev)}, "
            f"{'indicating strong' if ev >= 0.7 else 'indicating moderate'} retrieval quality."
        )

    # Dominant failure
    fs = d.failure_summary or {}
    total_fs = fs.get("total", 0)
    if total_fs:
        ret_fail = fs.get("retrieval_failure_count", 0)
        gen_fail = fs.get("generation_failure_count", 0)
        if ret_fail > gen_fail:
            findings.append("Most incorrect answers stem from retrieval misses.")
        elif gen_fail > ret_fail:
            findings.append("Most incorrect answers stem from generation errors despite retrieved evidence.")

    if not findings:
        findings.append("Insufficient data to identify key findings.")

    lines = ["## Key Findings", ""]
    for f in findings:
        lines.append(f"- {f}")
    lines.append("")
    return lines


def _limitations_section(d: RunData) -> list[str]:
    return [
        "## Limitations",
        "",
        "- Results are based on a subset of FinanceBench; full-dataset evaluation pending.",
        "- Mock or local models were used; production frontier models may score differently.",
        "- Grounding and hallucination detection uses rule-based heuristics.",
        "",
    ]


def _next_steps_section(d: RunData) -> list[str]:
    return [
        "## Next Steps",
        "",
        "- Run evaluation with a stronger generator model (e.g., GPT-4o or Claude).",
        "- Experiment with hybrid retrieval (dense + BM25) to improve evidence_hit@k.",
        "- Expand to the full FinanceBench question set for statistical significance.",
        "",
    ]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _count(meta: dict) -> int | None:
    for key in ("attempted_count", "example_count", "num_examples"):
        v = meta.get(key)
        if isinstance(v, int):
            return v
    return None


def _pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "—"
    return f"{numerator / denominator:.0%}"


def _fmt(value: float) -> str:
    return f"{value:.2f}"


__all__ = ["generate_experiment_summary_report"]
