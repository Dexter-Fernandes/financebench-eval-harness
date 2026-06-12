import os
from pathlib import Path

import pytest

from financebench_eval_harness.llm import OllamaLLMClient
from financebench_eval_harness.run_config import load_evaluation_run_config


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_SMOKE") != "1",
    reason="Set RUN_OLLAMA_SMOKE=1 to run local Ollama smoke tests.",
)


@pytest.mark.parametrize(
    ("config_path", "expected_mode"),
    [
        ("configs/evaluation/ollama_closed_book.yaml", "closed_book"),
        ("configs/evaluation/ollama_oracle_context.yaml", "oracle_context"),
    ],
)
def test_ollama_smoke_configs_load_for_local_runs(
    config_path: str,
    expected_mode: str,
) -> None:
    config = load_evaluation_run_config(Path(config_path))

    assert config.settings.mode.value == expected_mode
    assert config.settings.limit == 5
    assert config.model.provider == "ollama"
    assert config.model.model_name == "llama3.2:3b"
    assert config.model.base_url == "http://localhost:11434"
    assert config.judge is not None
    assert config.judge.model.provider == "ollama"
    assert config.judge.model.base_url == "http://localhost:11434"
    assert isinstance(OllamaLLMClient(config.model), OllamaLLMClient)
