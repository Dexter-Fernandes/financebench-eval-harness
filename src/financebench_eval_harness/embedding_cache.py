"""Disk-backed embedding cache for M6 embedding comparison.

Cache key includes provider, model_name, text hash, dimensions, and normalize flag
so that changing any of these automatically invalidates cached entries.

Storage layout::

    {cache_dir}/{provider}__{model_slug}/
        {key[:2]}/{key}.json          # {"vector": [...], "dim": N, "cached_at": "..."}
        cache_metadata.json           # aggregate stats
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EmbeddingCache:
    """Text-hash-keyed disk cache for embedding vectors."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        provider: str,
        model_name: str,
        dimensions: int | None,
        normalize: bool,
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._dimensions = dimensions
        self._normalize = normalize

        slug = f"{provider}__{model_name}".replace("/", "__")
        self._model_dir = Path(cache_dir) / slug
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self._meta_path = self._model_dir / "cache_metadata.json"
        self._meta = self._load_or_init_meta()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> list[float] | None:
        key = self._key(text)
        entry_path = self._entry_path(key)
        if not entry_path.is_file():
            self._meta["miss_count"] += 1
            self._save_meta()
            return None
        data = json.loads(entry_path.read_text(encoding="utf-8"))
        self._meta["hit_count"] += 1
        self._save_meta()
        return data["vector"]

    def put(self, text: str, vector: list[float]) -> None:
        key = self._key(text)
        entry_path = self._entry_path(key)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        entry_path.write_text(
            json.dumps({
                "vector": vector,
                "dim": len(vector),
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )
        self._meta["total_entries"] += 1
        self._meta["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        self._save_meta()

    def get_batch(
        self, texts: list[str]
    ) -> tuple[list[list[float] | None], int, int]:
        """Return (results, hit_count, miss_count) for a batch of texts."""
        results: list[list[float] | None] = []
        hits = misses = 0
        for text in texts:
            key = self._key(text)
            entry_path = self._entry_path(key)
            if entry_path.is_file():
                data = json.loads(entry_path.read_text(encoding="utf-8"))
                results.append(data["vector"])
                hits += 1
            else:
                results.append(None)
                misses += 1
        self._meta["hit_count"] += hits
        self._meta["miss_count"] += misses
        self._save_meta()
        return results, hits, misses

    def put_batch(self, texts: list[str], vectors: list[list[float]]) -> None:
        for text, vector in zip(texts, vectors):
            self.put(text, vector)

    def load_metadata(self) -> dict[str, Any]:
        return dict(self._meta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        parts = f"{self._provider}|{self._model_name}|{text_hash}|{self._dimensions}|{self._normalize}"
        return hashlib.sha256(parts.encode("utf-8")).hexdigest()

    def _entry_path(self, key: str) -> Path:
        return self._model_dir / key[:2] / f"{key}.json"

    def _load_or_init_meta(self) -> dict[str, Any]:
        if self._meta_path.is_file():
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        return {
            "provider": self._provider,
            "model_name": self._model_name,
            "dimensions": self._dimensions,
            "normalize": self._normalize,
            "hit_count": 0,
            "miss_count": 0,
            "total_entries": 0,
            "last_updated_utc": None,
        }

    def _save_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps(self._meta, indent=2), encoding="utf-8"
        )


__all__ = ["EmbeddingCache"]
