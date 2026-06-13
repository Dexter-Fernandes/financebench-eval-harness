"""M5.5 — RAG generation config (TDD)."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from financebench_eval_harness.evaluation import EvaluationMode
from financebench_eval_harness.rag_run_config import (
    RAGEvalSettings,
    RAGRunConfig,
    RAGRunConfigError,
    load_rag_run_config,
)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, content: dict[str, Any]) -> None:
    path.write_text(yaml.dump(content), encoding="utf-8")


def _minimal_yaml_dict(
    *,
    mode: str = "rag_dense",
    top_k: int = 5,
    max_context_chars: int | None = 4000,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "rag_eval": {
            "examples_path": "data/processed/financebench/examples.jsonl",
            "retrieval_results_path": "runs/retrieval/retrieval_results.jsonl",
            "output_dir": "runs/",
            "mode": mode,
            "top_k": top_k,
        },
        "model": {
            "provider": "mock",
            "model_name": "mock-llm",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": 30.0,
        },
    }
    if max_context_chars is not None:
        d["rag_eval"]["max_context_chars"] = max_context_chars
    return d


def _full_yaml_with_judge() -> dict[str, Any]:
    d = _minimal_yaml_dict()
    d["judge"] = {
        "enabled": True,
        "provider": "mock",
        "model_name": "mock-judge",
        "temperature": 0.0,
        "max_tokens": 256,
        "timeout_seconds": 30.0,
        "prompt": {
            "id": "answer_correctness_v1",
            "version": "v1",
            "template_path": "prompts/judges/answer_correctness_v1.txt",
        },
    }
    return d


# ---------------------------------------------------------------------------
# Cycle 1 — load_rag_run_config() returns RAGRunConfig
# ---------------------------------------------------------------------------


def test_load_rag_run_config_returns_rag_run_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())

    result = load_rag_run_config(cfg_path)

    assert isinstance(result, RAGRunConfig)
    assert isinstance(result.settings, RAGEvalSettings)


# ---------------------------------------------------------------------------
# Cycle 2 — path fields are Path objects
# ---------------------------------------------------------------------------


def test_examples_path_is_path_object(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    assert isinstance(result.settings.examples_path, Path)
    assert result.settings.examples_path == Path("data/processed/financebench/examples.jsonl")


def test_retrieval_results_path_is_path_object(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    assert isinstance(result.settings.retrieval_results_path, Path)
    assert result.settings.retrieval_results_path == Path("runs/retrieval/retrieval_results.jsonl")


def test_output_dir_is_path_object(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    assert isinstance(result.settings.output_dir, Path)
    assert result.settings.output_dir == Path("runs/")


# ---------------------------------------------------------------------------
# Cycle 3 — mode is EvaluationMode
# ---------------------------------------------------------------------------


def test_mode_rag_dense_is_parsed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(mode="rag_dense"))
    result = load_rag_run_config(cfg_path)
    assert result.settings.mode is EvaluationMode.RAG_DENSE


def test_mode_rag_no_context_is_parsed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(mode="rag_no_context"))
    result = load_rag_run_config(cfg_path)
    assert result.settings.mode is EvaluationMode.RAG_NO_CONTEXT


# ---------------------------------------------------------------------------
# Cycle 4 — top_k is positive int
# ---------------------------------------------------------------------------


def test_top_k_is_parsed_as_int(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(top_k=10))
    result = load_rag_run_config(cfg_path)
    assert result.settings.top_k == 10
    assert type(result.settings.top_k) is int


# ---------------------------------------------------------------------------
# Cycles 5-6 — max_context_chars optional
# ---------------------------------------------------------------------------


def test_max_context_chars_parsed_when_present(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(max_context_chars=12000))
    result = load_rag_run_config(cfg_path)
    assert result.settings.max_context_chars == 12000


def test_max_context_chars_is_none_when_omitted(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(max_context_chars=None))
    result = load_rag_run_config(cfg_path)
    assert result.settings.max_context_chars is None


# ---------------------------------------------------------------------------
# Cycle 7 — model section populates LLMGenerationConfig
# ---------------------------------------------------------------------------


def test_model_provider_and_name_parsed(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    assert result.model.provider == "mock"
    assert result.model.model_name == "mock-llm"
    assert result.model.temperature == 0.0
    assert result.model.max_tokens == 512
    assert result.model.timeout_seconds == 30.0
    assert result.model.base_url is None


# ---------------------------------------------------------------------------
# Cycle 8 — missing required rag_eval key → RAGRunConfigError
# ---------------------------------------------------------------------------


def test_missing_examples_path_raises_error(tmp_path: Path) -> None:
    d = _minimal_yaml_dict()
    del d["rag_eval"]["examples_path"]
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, d)
    with pytest.raises(RAGRunConfigError, match="examples_path"):
        load_rag_run_config(cfg_path)


def test_missing_retrieval_results_path_raises_error(tmp_path: Path) -> None:
    d = _minimal_yaml_dict()
    del d["rag_eval"]["retrieval_results_path"]
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, d)
    with pytest.raises(RAGRunConfigError, match="retrieval_results_path"):
        load_rag_run_config(cfg_path)


def test_missing_top_k_raises_error(tmp_path: Path) -> None:
    d = _minimal_yaml_dict()
    del d["rag_eval"]["top_k"]
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, d)
    with pytest.raises(RAGRunConfigError, match="top_k"):
        load_rag_run_config(cfg_path)


def test_missing_rag_eval_section_raises_error(tmp_path: Path) -> None:
    d = _minimal_yaml_dict()
    del d["rag_eval"]
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, d)
    with pytest.raises(RAGRunConfigError):
        load_rag_run_config(cfg_path)


# ---------------------------------------------------------------------------
# Cycle 9 — invalid top_k → RAGRunConfigError
# ---------------------------------------------------------------------------


def test_top_k_zero_raises_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(top_k=0))
    with pytest.raises(RAGRunConfigError, match="top_k"):
        load_rag_run_config(cfg_path)


def test_top_k_negative_raises_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(top_k=-1))
    with pytest.raises(RAGRunConfigError, match="top_k"):
        load_rag_run_config(cfg_path)


# ---------------------------------------------------------------------------
# Cycle 10 — invalid mode → RAGRunConfigError
# ---------------------------------------------------------------------------


def test_invalid_mode_raises_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict(mode="not_a_real_mode"))
    with pytest.raises(RAGRunConfigError, match="mode"):
        load_rag_run_config(cfg_path)


# ---------------------------------------------------------------------------
# Cycle 11 — to_dict() round-trip
# ---------------------------------------------------------------------------


def test_to_dict_contains_rag_eval_section(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    d = result.to_dict()
    assert "rag_eval" in d
    assert "model" in d


def test_to_dict_rag_eval_has_expected_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    rag_eval = result.to_dict()["rag_eval"]
    assert rag_eval["mode"] == "rag_dense"
    assert rag_eval["top_k"] == 5
    assert rag_eval["examples_path"] == "data/processed/financebench/examples.jsonl"
    assert rag_eval["retrieval_results_path"] == "runs/retrieval/retrieval_results.jsonl"
    assert rag_eval["output_dir"] == "runs"  # Path normalises away trailing slash


def test_to_dict_round_trips_via_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    serialized = yaml.dump(result.to_dict())
    round_tripped_path = tmp_path / "round_tripped.yaml"
    round_tripped_path.write_text(serialized, encoding="utf-8")
    reloaded = load_rag_run_config(round_tripped_path)
    assert reloaded.settings.mode == result.settings.mode
    assert reloaded.settings.top_k == result.settings.top_k
    assert reloaded.model.model_name == result.model.model_name


# ---------------------------------------------------------------------------
# Cycles 12-13 — judge section
# ---------------------------------------------------------------------------


def test_judge_section_absent_gives_none(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    assert result.judge is None


def test_judge_section_enabled_gives_judge_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _full_yaml_with_judge())
    result = load_rag_run_config(cfg_path)
    assert result.judge is not None
    assert result.judge.enabled is True
    assert result.judge.model.model_name == "mock-judge"
    assert result.judge.prompt.id == "answer_correctness_v1"


def test_judge_section_disabled_gives_none(tmp_path: Path) -> None:
    d = _full_yaml_with_judge()
    d["judge"]["enabled"] = False
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, d)
    result = load_rag_run_config(cfg_path)
    assert result.judge is None


# ---------------------------------------------------------------------------
# Cycles 14-15 — retrieval_run_id optional field (M5.7)
# ---------------------------------------------------------------------------


def test_retrieval_run_id_parsed_when_present(tmp_path: Path) -> None:
    d = _minimal_yaml_dict()
    d["rag_eval"]["retrieval_run_id"] = "run_2026_dense"
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, d)
    result = load_rag_run_config(cfg_path)
    assert result.settings.retrieval_run_id == "run_2026_dense"


def test_retrieval_run_id_defaults_to_none_when_absent(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    assert result.settings.retrieval_run_id is None


def test_to_dict_includes_retrieval_run_id_when_set(tmp_path: Path) -> None:
    d = _minimal_yaml_dict()
    d["rag_eval"]["retrieval_run_id"] = "run_test"
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, d)
    result = load_rag_run_config(cfg_path)
    assert result.to_dict()["rag_eval"]["retrieval_run_id"] == "run_test"


def test_to_dict_omits_retrieval_run_id_when_none(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _minimal_yaml_dict())
    result = load_rag_run_config(cfg_path)
    assert "retrieval_run_id" not in result.to_dict()["rag_eval"]
