from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATASET_CONFIG_PATH = Path("configs/datasets/financebench.yaml")
REQUIRED_DATASET_KEYS = {
    "name",
    "questions_path",
    "documents_dir",
    "processed_dir",
}


@dataclass(frozen=True)
class DatasetConfig:
    """Configured local dataset paths."""

    name: str
    questions_path: Path
    documents_dir: Path
    processed_dir: Path

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path,
        *,
        name: str = "financebench",
    ) -> "DatasetConfig":
        root = Path(data_root)
        return cls(
            name=name,
            questions_path=root / "questions.jsonl",
            documents_dir=root / "documents",
            processed_dir=Path("data/processed") / name,
        )

    def expected_layout_message(self) -> str:
        return (
            "Expected:\n"
            f"  {self.questions_path}\n"
            f"  {self.documents_dir}/"
        )


class DatasetConfigError(ValueError):
    """Raised when a dataset config cannot be loaded or validated."""


def load_dataset_config(
    config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
) -> DatasetConfig:
    path = Path(config_path)

    if not path.is_file():
        raise DatasetConfigError(f"Dataset config file not found: {path}")

    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetConfigError(f"Dataset config is not valid YAML: {path}") from exc

    if not isinstance(raw_config, dict):
        raise DatasetConfigError(f"Dataset config must be a mapping: {path}")

    dataset = raw_config.get("dataset")
    if not isinstance(dataset, dict):
        raise DatasetConfigError(f"Dataset config must contain a 'dataset' mapping: {path}")

    missing_keys = sorted(REQUIRED_DATASET_KEYS - dataset.keys())
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise DatasetConfigError(f"Dataset config missing required key(s): {missing}")

    return _dataset_config_from_mapping(dataset)


def _dataset_config_from_mapping(dataset: dict[str, Any]) -> DatasetConfig:
    name = dataset["name"]
    if not isinstance(name, str) or not name:
        raise DatasetConfigError("Dataset config key 'name' must be a non-empty string")

    return DatasetConfig(
        name=name,
        questions_path=_path_from_config_value(dataset, "questions_path"),
        documents_dir=_path_from_config_value(dataset, "documents_dir"),
        processed_dir=_path_from_config_value(dataset, "processed_dir"),
    )


def _path_from_config_value(dataset: dict[str, Any], key: str) -> Path:
    value = dataset[key]
    if not isinstance(value, str) or not value:
        raise DatasetConfigError(f"Dataset config key '{key}' must be a non-empty string")
    return Path(value)
