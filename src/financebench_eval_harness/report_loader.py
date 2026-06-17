"""M8.4: Report data loader — loads all run artifacts for report generation."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ReportLoadError(ValueError):
    """Raised when a required run file is missing or unreadable."""


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    run_id: str
    run_metadata: dict | None
    rag_run_metadata: dict | None
    config: dict | None
    retrieval_summary: dict | None
    joined_summary: dict | None
    failure_summary: dict | None
    retrieval_scores: list[dict] | None
    rag_predictions: list[dict] | None
    failure_analysis: list[dict] | None


_OPTIONAL_FILES: tuple[str, ...] = (
    "run_metadata.json",
    "rag_run_metadata.json",
    "config.yaml",
    "retrieval_summary.json",
    "joined/joined_summary.json",
    "failure_summary.json",
    "retrieval_scores.jsonl",
    "rag_predictions.jsonl",
    "failure_analysis.jsonl",
)


def load_run_data(
    run_dir: str | Path,
    *,
    required: Sequence[str] = (),
) -> RunData:
    """Load run artifacts from *run_dir*.

    Missing optional files produce ``warnings.warn`` calls.
    Any filename listed in *required* that is absent raises ``ReportLoadError``.
    """
    run_dir = Path(run_dir)
    required_set = set(required)

    def _check(rel: str) -> bool:
        path = run_dir / rel
        if path.is_file():
            return True
        if rel in required_set:
            raise ReportLoadError(
                f"Required run file not found: {path} ({rel!r} listed in required)"
            )
        warnings.warn(
            f"Optional run file not found, skipping: {path}",
            stacklevel=3,
        )
        return False

    def _load_json(rel: str) -> dict | None:
        if not _check(rel):
            return None
        try:
            import yaml  # optional yaml for config.yaml
            if rel.endswith(".yaml"):
                return yaml.safe_load((run_dir / rel).read_text(encoding="utf-8")) or {}
        except ImportError:
            pass
        try:
            return json.loads((run_dir / rel).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"Could not parse {run_dir / rel}: {exc}", stacklevel=3)
            return None

    def _load_jsonl(rel: str) -> list[dict] | None:
        if not _check(rel):
            return None
        rows: list[dict] = []
        for line in (run_dir / rel).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows

    run_metadata = _load_json("run_metadata.json")
    rag_run_metadata = _load_json("rag_run_metadata.json")

    config: dict | None = None
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except ImportError:
            pass
    elif "config.yaml" in required_set:
        raise ReportLoadError(f"Required run file not found: {config_path}")
    else:
        warnings.warn(f"Optional run file not found, skipping: {config_path}", stacklevel=2)

    retrieval_summary = _load_json("retrieval_summary.json")

    joined_summary: dict | None = None
    joined_path = run_dir / "joined" / "joined_summary.json"
    if joined_path.is_file():
        try:
            joined_summary = json.loads(joined_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"Could not parse {joined_path}: {exc}", stacklevel=2)
    elif "joined/joined_summary.json" in required_set:
        raise ReportLoadError(f"Required run file not found: {joined_path}")
    else:
        warnings.warn(f"Optional run file not found, skipping: {joined_path}", stacklevel=2)

    failure_summary = _load_json("failure_summary.json")
    retrieval_scores = _load_jsonl("retrieval_scores.jsonl")
    rag_predictions = _load_jsonl("rag_predictions.jsonl")
    failure_analysis = _load_jsonl("failure_analysis.jsonl")

    return RunData(
        run_dir=run_dir,
        run_id=run_dir.name,
        run_metadata=run_metadata,
        rag_run_metadata=rag_run_metadata,
        config=config,
        retrieval_summary=retrieval_summary,
        joined_summary=joined_summary,
        failure_summary=failure_summary,
        retrieval_scores=retrieval_scores,
        rag_predictions=rag_predictions,
        failure_analysis=failure_analysis,
    )


__all__ = ["ReportLoadError", "RunData", "load_run_data"]
