"""Configuration loading and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def ensure_output_tree(root: str | Path) -> dict[str, Path]:
    output_root = Path(root).resolve()
    paths = {
        "root": output_root,
        "processed": output_root / "processed",
        "figures": output_root / "figures",
        "metrics": output_root / "metrics",
        "states": output_root / "states",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

