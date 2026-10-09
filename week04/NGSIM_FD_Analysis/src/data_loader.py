"""Load NGSIM CSV files into a canonical SI schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import pandas as pd


CANONICAL_COLUMNS = [
    "vehicle_id", "frame_id", "time_s", "global_time_ms", "x_m", "s_m",
    "length_m", "vehicle_class", "speed_m_s", "accel_m_s2", "lane_id",
]


def _validate_local_csv(path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV input, received: {path.suffix}")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Input must be a regular local file: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectory_csv(path: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load selected columns and convert them to a common metre-second system."""
    csv_path = Path(path).resolve(strict=True)
    _validate_local_csv(csv_path)
    input_cfg = config["input"]
    mapping = input_cfg["field_map"]
    missing_map = [name for name in CANONICAL_COLUMNS if name not in mapping]
    if missing_map:
        raise ValueError(f"Missing canonical field mappings: {missing_map}")

    source_columns = [mapping[name] for name in CANONICAL_COLUMNS]
    if input_cfg.get("has_header", True):
        frame = pd.read_csv(csv_path, usecols=source_columns, low_memory=False)
    else:
        standard = input_cfg.get("standard_columns")
        if not standard:
            raise ValueError("input.standard_columns is required when has_header=false")
        frame = pd.read_csv(csv_path, header=None, names=standard, usecols=source_columns, low_memory=False)
    frame = frame.rename(columns={source: canonical for canonical, source in mapping.items()})
    frame = frame[CANONICAL_COLUMNS].copy()

    for column in CANONICAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    units = str(input_cfg.get("source_units", "si")).lower()
    if units == "ngsim_imperial":
        frame["x_m"] *= 0.3048
        frame["s_m"] *= 0.3048
        frame["length_m"] *= 0.3048
        frame["speed_m_s"] *= 0.3048
        frame["accel_m_s2"] *= 0.3048
        frame["time_s"] = (frame["global_time_ms"] - frame["global_time_ms"].min()) / 1000.0
    elif units != "si":
        raise ValueError("input.source_units must be 'si' or 'ngsim_imperial'")

    summary = {
        "input_basename": csv_path.name,
        "input_bytes": int(csv_path.stat().st_size),
        "input_sha256": _sha256(csv_path),
        "rows_loaded": int(len(frame)),
        "columns_loaded": CANONICAL_COLUMNS,
        "source_units": units,
        "common_longitudinal_origin": True,
        "per_vehicle_origin_reset": False,
    }
    return frame, summary
