"""Tests for eval_retrieval.py — M4.7 retrieval evaluation config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from financebench_eval_harness.eval_retrieval import generate_retrieval_report, score_retrieval_run
from financebench_eval_harness.pipeline_config import PipelineConfig
from financebench_eval_harness.chunking import ChunkingConfig
from financebench_eval_harness.embedding import EmbeddingConfig


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_CHUNKING = ChunkingConfig(chunk_size=800, chunk_overlap=150, strategy="recursive_text", min_chunk_chars=0)
_EMBEDDING = EmbeddingConfig(provider="mock", model_name="mock-embed")


def _make_config(
    tmp_path: Path,
    *,
    top_k: int = 5,
    evidence_overlap_threshold: float = 0.5,
) -> PipelineConfig:
    return PipelineConfig(
        pages_path=tmp_path / "pages.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        index_dir=tmp_path / "idx",
        questions_path=tmp_path / "examples.jsonl",
        runs_dir=tmp_path / "runs",
        top_k=top_k,
        evidence_overlap_threshold=evidence_overlap_threshold,
        chunking=_CHUNKING,
        embedding=_EMBEDDING,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _make_retrieval_result(question_id: str, doc_name: str, page_num: int, text: str) -> dict:
    return {
        "question_id": question_id,
        "query": "What were capex?",
        "retrieved": [
            {"rank": 1, "chunk_id": "c1", "doc_name": doc_name, "page_num": page_num,
             "score": 0.9, "text": text},
        ],
    }


def _make_gold_example(
    question_id: str,
    doc_name: str,
    gold_page_num: int,
    matched_page_num: int,
    evidence_text: str,
) -> dict:
    return {
        "question_id": question_id,
        "company": "3M",
        "doc_name": doc_name,
        "question": "What were capex?",
        "gold_answer": "$1577M",
        "evidence": [
            {
                "doc_name": doc_name,
                "gold_page_num": gold_page_num,
                "matched_page_num": matched_page_num,
                "evidence_text": evidence_text,
                "page_text": "full page text",
            }
        ],
    }


# ---------------------------------------------------------------------------
# M4.7 — Slice 1: score_retrieval_run returns summary with example_count
# ---------------------------------------------------------------------------


def test_score_retrieval_run_returns_summary_with_example_count(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    _write_jsonl(
        run_dir / "retrieval_results.jsonl",
        [_make_retrieval_result("q1", "3M_2018_10K.pdf", 60,
                                "purchases of property plant and equipment 1577")],
    )
    _write_jsonl(
        cfg.questions_path,
        [_make_gold_example("q1", "3M_2018_10K", 59, 60,
                            "purchases of property plant and equipment 1577")],
    )

    summary = score_retrieval_run(cfg, run_dir)

    assert summary["example_count"] == 1


# ---------------------------------------------------------------------------
# M4.7 — Slice 2: writes retrieval_scores.jsonl with per-question metrics
# ---------------------------------------------------------------------------


def test_score_retrieval_run_writes_retrieval_scores_jsonl(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    _write_jsonl(
        run_dir / "retrieval_results.jsonl",
        [_make_retrieval_result("q1", "3M_2018_10K.pdf", 60,
                                "purchases of property plant and equipment 1577")],
    )
    _write_jsonl(
        cfg.questions_path,
        [_make_gold_example("q1", "3M_2018_10K", 59, 60,
                            "purchases of property plant and equipment 1577")],
    )

    score_retrieval_run(cfg, run_dir)

    scores_path = run_dir / "retrieval_scores.jsonl"
    assert scores_path.exists()
    rows = [json.loads(line) for line in scores_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["question_id"] == "q1"
    assert "doc_hit@5" in rows[0]
    assert "page_hit@5" in rows[0]
    assert "evidence_text_hit@5" in rows[0]
    assert "best_evidence_overlap" in rows[0]


def test_score_retrieval_run_doc_hit_true_for_matching_chunk(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    _write_jsonl(
        run_dir / "retrieval_results.jsonl",
        [_make_retrieval_result("q1", "3M_2018_10K.pdf", 60, "some text")],
    )
    _write_jsonl(
        cfg.questions_path,
        [_make_gold_example("q1", "3M_2018_10K", 59, 60, "some text")],
    )

    score_retrieval_run(cfg, run_dir)

    rows = [json.loads(l) for l in (run_dir / "retrieval_scores.jsonl").read_text().splitlines() if l.strip()]
    assert rows[0]["doc_hit@5"] is True


# ---------------------------------------------------------------------------
# M4.7 — Slice 3: config.evidence_overlap_threshold and config.top_k are used
# ---------------------------------------------------------------------------


def test_score_retrieval_run_threshold_affects_evidence_text_hit(tmp_path: Path) -> None:
    # Chunk has partial token overlap — hit status depends on threshold
    # Evidence: "alpha beta gamma delta" (4 tokens); chunk: "alpha beta gamma other" (3/4 = 0.75 coverage)
    evidence_text = "alpha beta gamma delta"
    chunk_text = "alpha beta gamma other words here"

    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    _write_jsonl(
        run_dir / "retrieval_results.jsonl",
        [_make_retrieval_result("q1", "DOC.pdf", 1, chunk_text)],
    )

    examples_path = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples_path,
        [_make_gold_example("q1", "DOC", 1, 1, evidence_text)],
    )

    # High threshold — 0.75 coverage is a hit at 0.7, not at 0.8
    cfg_low = PipelineConfig(
        pages_path=tmp_path / "p.jsonl", chunks_path=tmp_path / "c.jsonl",
        index_dir=tmp_path / "idx", questions_path=examples_path,
        runs_dir=tmp_path / "runs", top_k=5,
        evidence_overlap_threshold=0.7,
        chunking=_CHUNKING, embedding=_EMBEDDING,
    )
    cfg_high = PipelineConfig(
        pages_path=tmp_path / "p.jsonl", chunks_path=tmp_path / "c.jsonl",
        index_dir=tmp_path / "idx", questions_path=examples_path,
        runs_dir=tmp_path / "runs", top_k=5,
        evidence_overlap_threshold=0.8,
        chunking=_CHUNKING, embedding=_EMBEDDING,
    )

    run_dir_low = tmp_path / "run_low"
    run_dir_low.mkdir()
    (run_dir_low / "retrieval_results.jsonl").write_bytes((run_dir / "retrieval_results.jsonl").read_bytes())
    score_retrieval_run(cfg_low, run_dir_low)

    run_dir_high = tmp_path / "run_high"
    run_dir_high.mkdir()
    (run_dir_high / "retrieval_results.jsonl").write_bytes((run_dir / "retrieval_results.jsonl").read_bytes())
    score_retrieval_run(cfg_high, run_dir_high)

    rows_low = [json.loads(l) for l in (run_dir_low / "retrieval_scores.jsonl").read_text().splitlines() if l.strip()]
    rows_high = [json.loads(l) for l in (run_dir_high / "retrieval_scores.jsonl").read_text().splitlines() if l.strip()]

    assert rows_low[0]["evidence_text_hit"] is True
    assert rows_high[0]["evidence_text_hit"] is False


def test_score_retrieval_run_top_k_limits_overlap_scoring(tmp_path: Path) -> None:
    # Matching chunk is at rank 2; top_k=1 should exclude it from overlap scoring
    evidence_text = "alpha beta gamma delta"
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    examples_path = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples_path,
        [_make_gold_example("q1", "DOC", 1, 1, evidence_text)],
    )

    retrieval_with_match_at_rank2 = {
        "question_id": "q1",
        "query": "capex",
        "retrieved": [
            {"rank": 1, "chunk_id": "c1", "doc_name": "DOC.pdf", "page_num": 1, "score": 0.9, "text": "unrelated"},
            {"rank": 2, "chunk_id": "c2", "doc_name": "DOC.pdf", "page_num": 1, "score": 0.7, "text": evidence_text},
        ],
    }
    _write_jsonl(run_dir / "retrieval_results.jsonl", [retrieval_with_match_at_rank2])

    cfg = PipelineConfig(
        pages_path=tmp_path / "p.jsonl", chunks_path=tmp_path / "c.jsonl",
        index_dir=tmp_path / "idx", questions_path=examples_path,
        runs_dir=tmp_path / "runs", top_k=1,
        evidence_overlap_threshold=0.5,
        chunking=_CHUNKING, embedding=_EMBEDDING,
    )

    score_retrieval_run(cfg, run_dir)

    rows = [json.loads(l) for l in (run_dir / "retrieval_scores.jsonl").read_text().splitlines() if l.strip()]
    assert rows[0]["best_evidence_overlap"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# M4.7 — Slice 4: writes retrieval_eval_config.yaml with all config fields
# ---------------------------------------------------------------------------


def _run_with_one_example(tmp_path: Path, top_k: int = 5, threshold: float = 0.5) -> Path:
    """Helper: set up one retrieval result + gold example, run score_retrieval_run, return run_dir."""
    cfg = _make_config(tmp_path, top_k=top_k, evidence_overlap_threshold=threshold)
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(exist_ok=True)
    _write_jsonl(
        run_dir / "retrieval_results.jsonl",
        [_make_retrieval_result("q1", "3M_2018_10K.pdf", 60, "some text")],
    )
    _write_jsonl(
        cfg.questions_path,
        [_make_gold_example("q1", "3M_2018_10K", 59, 60, "some text")],
    )
    score_retrieval_run(cfg, run_dir)
    return run_dir


def test_score_retrieval_run_writes_retrieval_eval_config_yaml(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    assert (run_dir / "retrieval_eval_config.yaml").exists()


def test_retrieval_eval_config_yaml_contains_top_k(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path, top_k=10)
    d = yaml.safe_load((run_dir / "retrieval_eval_config.yaml").read_text())
    assert d["retrieval"]["top_k"] == 10


def test_retrieval_eval_config_yaml_contains_evidence_overlap_threshold(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path, threshold=0.7)
    d = yaml.safe_load((run_dir / "retrieval_eval_config.yaml").read_text())
    assert d["retrieval"]["evidence_overlap_threshold"] == pytest.approx(0.7)


def test_retrieval_eval_config_yaml_contains_chunking_fields(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    d = yaml.safe_load((run_dir / "retrieval_eval_config.yaml").read_text())
    assert d["retrieval"]["chunking"]["chunk_size"] == 800
    assert d["retrieval"]["chunking"]["strategy"] == "recursive_text"


def test_retrieval_eval_config_yaml_contains_embedding_provider(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    d = yaml.safe_load((run_dir / "retrieval_eval_config.yaml").read_text())
    assert d["retrieval"]["embedding"]["provider"] == "mock"
    assert d["retrieval"]["embedding"]["model_name"] == "mock-embed"


# ---------------------------------------------------------------------------
# M4.7 — Slice 5: writes retrieval_summary.json with aggregate metrics
# ---------------------------------------------------------------------------


def test_score_retrieval_run_writes_retrieval_summary_json(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    assert (run_dir / "retrieval_summary.json").exists()


def test_retrieval_summary_json_contains_example_count(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    d = json.loads((run_dir / "retrieval_summary.json").read_text())
    assert d["example_count"] == 1


def test_retrieval_summary_json_contains_hit_at_k_rates(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    d = json.loads((run_dir / "retrieval_summary.json").read_text())
    assert "doc_hit@5_rate" in d
    assert "page_hit@5_rate" in d
    assert "evidence_text_hit@5_rate" in d


def test_retrieval_summary_json_contains_rank_metrics(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    d = json.loads((run_dir / "retrieval_summary.json").read_text())
    assert "doc_mrr@5" in d
    assert "doc_median_first_hit_rank" in d


def test_retrieval_summary_doc_hit_rate_correct_for_matching_run(tmp_path: Path) -> None:
    # Exact match → doc_hit@5_rate should be 1.0
    run_dir = _run_with_one_example(tmp_path)
    d = json.loads((run_dir / "retrieval_summary.json").read_text())
    assert d["doc_hit@5_rate"] == pytest.approx(1.0)


def test_score_retrieval_run_skips_questions_not_in_gold(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    _write_jsonl(
        run_dir / "retrieval_results.jsonl",
        [
            _make_retrieval_result("q1", "3M_2018_10K.pdf", 60, "text A"),
            _make_retrieval_result("q_unknown", "OTHER.pdf", 1, "text B"),
        ],
    )
    _write_jsonl(
        cfg.questions_path,
        [_make_gold_example("q1", "3M_2018_10K", 59, 60, "text A")],
    )

    summary = score_retrieval_run(cfg, run_dir)

    assert summary["example_count"] == 1


# ---------------------------------------------------------------------------
# M4.8 — generate_retrieval_report unit tests
# ---------------------------------------------------------------------------


def _make_summary(example_count: int = 1) -> dict:
    return {
        "example_count": example_count,
        "doc_hit@5_rate": 1.0,
        "page_hit@5_rate": 1.0,
        "evidence_text_hit@5_rate": 1.0,
        "doc_mrr@5": 1.0,
        "doc_median_first_hit_rank": 1,
    }


def test_generate_retrieval_report_creates_file(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    report_dir = tmp_path / "reports"
    report_path = generate_retrieval_report(_make_summary(), "run_001", cfg, output_dir=report_dir)
    assert report_path.is_file()
    assert report_path.name == "retrieval_eval_run_001.md"


def test_generate_retrieval_report_contains_run_id(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    report_path = generate_retrieval_report(_make_summary(), "run_042", cfg, output_dir=tmp_path)
    assert "run_042" in report_path.read_text()


def test_generate_retrieval_report_contains_threshold(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, evidence_overlap_threshold=0.7)
    report_path = generate_retrieval_report(_make_summary(), "run_001", cfg, output_dir=tmp_path)
    assert "0.7" in report_path.read_text()


def test_generate_retrieval_report_contains_example_count(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    report_path = generate_retrieval_report(_make_summary(example_count=42), "run_001", cfg, output_dir=tmp_path)
    assert "42" in report_path.read_text()


def test_generate_retrieval_report_creates_output_dir(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    nested_dir = tmp_path / "a" / "b" / "c"
    generate_retrieval_report(_make_summary(), "run_001", cfg, output_dir=nested_dir)
    assert nested_dir.is_dir()


# ---------------------------------------------------------------------------
# M4.9 — gold context and k in per-question rows
# ---------------------------------------------------------------------------


def _read_scores(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "retrieval_scores.jsonl").read_text().splitlines() if l.strip()]


def test_score_retrieval_run_row_contains_k(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path, top_k=7)
    rows = _read_scores(run_dir)
    assert rows[0]["k"] == 7


def test_score_retrieval_run_row_contains_gold_doc_name(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    rows = _read_scores(run_dir)
    assert rows[0]["gold_doc_name"] == "3M_2018_10K"


def test_score_retrieval_run_row_contains_gold_page_num(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    rows = _read_scores(run_dir)
    assert rows[0]["gold_page_num"] == 59


# ---------------------------------------------------------------------------
# M4.10 — compact leaderboard summary
# ---------------------------------------------------------------------------


def _read_leaderboard(run_dir: Path) -> dict:
    return json.loads((run_dir / "retrieval_leaderboard.json").read_text())


def test_score_retrieval_run_writes_leaderboard_json(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    assert (run_dir / "retrieval_leaderboard.json").exists()


def test_leaderboard_contains_run_id(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    assert _read_leaderboard(run_dir)["run_id"] == "run_001"


def test_leaderboard_k_equals_config_top_k(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path, top_k=7)
    assert _read_leaderboard(run_dir)["k"] == 7


def test_leaderboard_doc_hit_at_k_is_rate_at_configured_k(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path, top_k=5)
    d = _read_leaderboard(run_dir)
    assert "doc_hit@k" in d
    assert isinstance(d["doc_hit@k"], float)


def test_leaderboard_mean_best_evidence_overlap_is_present(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    d = _read_leaderboard(run_dir)
    assert "mean_best_evidence_overlap" in d
    assert isinstance(d["mean_best_evidence_overlap"], float)


def test_leaderboard_contains_config_metadata(tmp_path: Path) -> None:
    run_dir = _run_with_one_example(tmp_path)
    d = _read_leaderboard(run_dir)
    assert d["embedding_provider"] == "mock"
    assert d["embedding_model_name"] == "mock-embed"
    assert d["chunking_strategy"] == "recursive_text"
    assert d["chunk_size"] == 800
    assert d["chunk_overlap"] == 150
    assert d["evidence_overlap_threshold"] == pytest.approx(0.5)
