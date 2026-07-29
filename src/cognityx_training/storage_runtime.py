"""Shared Storage runtime resolution for dataset and artifact operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_storage_runtime(
    *,
    storage_runtime: Any | None = None,
    storage_config: str | Path | None = None,
    storage_root: str | Path | None = None,
) -> Any:
    """Resolve one runtime using the documented Cognityx precedence."""
    if storage_runtime is not None:
        return storage_runtime

    from cognityx_storage import StorageConfig, StorageRuntime

    if storage_config is not None:
        return StorageRuntime.load(config_file=storage_config)
    if storage_root is not None:
        return StorageRuntime.from_config(
            StorageConfig.built_in(root=storage_root)
        )
    return StorageRuntime.load()
