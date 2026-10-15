"""Shared, deterministic helpers for the NGSIM data pipeline.

The raw NGSIM files are only ever read.  All paths in the configuration are
resolved relative to the configuration file, which makes every entry point
independent of the caller's working directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)

# Windows rename retries.  See ``replace_file_atomically`` for the rationale.
REPLACE_ATTEMPTS = 12
REPLACE_INITIAL_DELAY_S = 0.05
REPLACE_MAXIMUM_DELAY_S = 1.0
_RETRYABLE_WINERRORS = frozenset({5, 32})  # ACCESS_DENIED, SHARING_VIOLATION


REQUIRED_COLUMNS = [
    "vehicle_id",
    "frame_id",
    "time_s",
    "global_time_ms",
    "local_x_m",
    "local_y_m",
    "vehicle_length_m",
    "vehicle_class",
    "speed_m_s",
    "acceleration_m_s2",
    "lane_id",
]

PAIR_COLUMNS = [
    "frame_id",
    "time_s",
    "global_time_ms",
    "lane_id",
    "leader_id",
    "follower_id",
    "leader_y",
    "follower_y",
    "leader_local_x_m",
    "follower_local_x_m",
    "leader_length",
    "follower_length",
    "gap_m",
    "leader_speed_m_s",
    "follower_speed_m_s",
    "relative_speed_m_s",
    "leader_acceleration_m_s2",
    "follower_acceleration_m_s2",
    "relative_acceleration_m_s2",
    "leader_class",
    "follower_class",
]

INTEGER_DTYPES = {
    "vehicle_id": "int32",
    "frame_id": "int32",
    "global_time_ms": "int64",
    "vehicle_class": "int8",
    "lane_id": "int8",
}


def configure_logging(name: str) -> logging.Logger:
    """Return a consistently configured script logger."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def default_config_path(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[1] / "config.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML and attach absolute, resolved project paths.

    The returned ``_paths`` mapping contains absolute :class:`Path` objects;
    the source configuration mapping remains unchanged.
    """

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping")
    configured_paths = config.get("paths")
    if not isinstance(configured_paths, dict):
        raise ValueError("config.yaml must contain a 'paths' mapping")

    project_root = path.parent
    resolved: dict[str, Path] = {}
    for key, value in configured_paths.items():
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        resolved[key] = candidate.resolve()

    config["_config_path"] = path
    config["_project_root"] = project_root
    config["_paths"] = resolved
    return config


def require_config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"config.yaml must contain a '{name}' mapping")
    return section


def ensure_output_dir(config: Mapping[str, Any], key: str) -> Path:
    paths = config.get("_paths")
    if not isinstance(paths, Mapping) or key not in paths:
        raise ValueError(f"config.yaml paths.{key} is required")
    output_dir = Path(paths[key])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_columns(
    frame: pd.DataFrame,
    expected: Iterable[str] = REQUIRED_COLUMNS,
    *,
    exact_order: bool = True,
) -> None:
    expected_list = list(expected)
    actual = list(frame.columns)
    missing = [column for column in expected_list if column not in actual]
    unexpected = [column for column in actual if column not in expected_list]
    if missing or unexpected:
        raise ValueError(
            f"CSV schema mismatch; missing={missing or 'none'}, "
            f"unexpected={unexpected or 'none'}"
        )
    if exact_order and actual != expected_list:
        raise ValueError(
            "CSV columns are not in the documented order; "
            f"expected={expected_list}, actual={actual}"
        )


def read_ngsim_csv(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Read the SI CSV with compact integer dtypes and no unit conversion."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"SI input CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path, dtype=INTEGER_DTYPES)
    if validate:
        validate_columns(frame)
    return frame


def file_sha256(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sanitize_json(value: Any) -> Any:
    """Convert non-finite numeric diagnostics to explicit JSON null values."""

    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _is_retryable_replace_error(error: OSError) -> bool:
    """Return True for the transient Windows lock errors seen while renaming."""

    if isinstance(error, FileNotFoundError):
        return False
    if getattr(error, "winerror", None) in _RETRYABLE_WINERRORS:
        return True
    # Non-Windows platforms and unusual sandboxes: a PermissionError on rename
    # is still worth retrying briefly before giving up.
    return isinstance(error, PermissionError)


def replace_file_atomically(temporary_name: str | Path, destination: str | Path) -> str:
    """Move a fully written temporary file onto its destination.

    ``os.replace`` is atomic within a volume, which is why every writer in this
    project stages output in a temporary file first.  On Windows, however, an
    external process (real-time antivirus, search indexer, backup or
    synchronisation agent) can briefly hold the freshly written temporary file
    open.  The rename then fails with ``ERROR_ACCESS_DENIED`` (WinError 5) or
    ``ERROR_SHARING_VIOLATION`` (WinError 32) even though nothing in this
    pipeline holds a handle -- observed as an intermittent failure after a
    multi-minute step had already computed its full result.

    This helper therefore retries with exponential backoff and, if the lock
    persists, falls back to overwriting the destination in place.  The fallback
    is not atomic, so it is logged as a warning, but it keeps a long pipeline
    from aborting at the very last operation.  The temporary file is always
    removed.

    Returns the strategy used: ``"replace"`` or ``"in_place_copy"``.
    """

    source = Path(temporary_name)
    target = Path(destination)
    delay = REPLACE_INITIAL_DELAY_S
    last_error: OSError | None = None
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            if attempt > 1:
                LOGGER.info(
                    "Renamed %s after %d attempts (transient file lock)", target.name, attempt
                )
            return "replace"
        except OSError as error:  # noqa: PERF203 - explicit retry loop
            if not _is_retryable_replace_error(error):
                raise
            last_error = error
            if attempt < REPLACE_ATTEMPTS:
                time.sleep(delay)
                delay = min(delay * 2.0, REPLACE_MAXIMUM_DELAY_S)
    LOGGER.warning(
        "Atomic rename of %s failed after %d attempts (%s); overwriting in place instead",
        target.name,
        REPLACE_ATTEMPTS,
        last_error,
    )
    try:
        with source.open("rb") as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
    finally:
        try:
            source.unlink()
        except FileNotFoundError:
            pass
    return "in_place_copy"


def write_json_atomic(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _sanitize_json(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
        replace_file_atomically(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_csv_atomic(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    float_format: str = "%.6f",
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        frame.to_csv(temporary_name, index=False, float_format=float_format)
        replace_file_atomically(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def reconstruct_adjacent_pairs(
    observations: pd.DataFrame,
    *,
    minimum_gap_m: float = 0.0,
    maximum_gap_m: float = float("inf"),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reconstruct direct same-frame, same-lane leader/follower pairs.

    Crucially, adjacency is established using *all* vehicle classes.  Vehicle
    class filtering belongs downstream, after this function returns.

    Frame/lane groups containing duplicate vehicle records or tied
    longitudinal positions are excluded wholesale because their direct
    adjacency is ambiguous.
    """

    validate_columns(observations)
    if not np.isfinite(minimum_gap_m):
        raise ValueError("minimum_gap_m must be finite")
    if maximum_gap_m <= minimum_gap_m:
        raise ValueError("maximum_gap_m must be greater than minimum_gap_m")

    source = observations.copy()
    duplicate_vehicle = source.duplicated(["vehicle_id", "frame_id"], keep=False)
    tied_position = source.duplicated(
        ["frame_id", "lane_id", "local_y_m"], keep=False
    )
    ambiguous_rows = duplicate_vehicle | tied_position
    ambiguous_groups = source.loc[ambiguous_rows, ["frame_id", "lane_id"]].drop_duplicates()
    if not ambiguous_groups.empty:
        ambiguous_index = pd.MultiIndex.from_frame(ambiguous_groups)
        row_index = pd.MultiIndex.from_frame(source[["frame_id", "lane_id"]])
        excluded_group_mask = row_index.isin(ambiguous_index)
        source = source.loc[~excluded_group_mask].copy()
        excluded_group_count = len(ambiguous_groups)
        excluded_row_count = int(np.count_nonzero(excluded_group_mask))
    else:
        excluded_group_count = 0
        excluded_row_count = 0

    source.sort_values(
        ["frame_id", "lane_id", "local_y_m", "vehicle_id"],
        kind="mergesort",
        inplace=True,
        ignore_index=True,
    )
    grouped = source.groupby(["frame_id", "lane_id"], sort=False, observed=True)

    leader_id = grouped["vehicle_id"].shift(-1)
    has_leader = leader_id.notna()
    follower = source.loc[has_leader]

    def leader_values(column: str) -> pd.Series:
        return grouped[column].shift(-1).loc[has_leader]

    pairs = pd.DataFrame(
        {
            "frame_id": follower["frame_id"].to_numpy(),
            "time_s": follower["time_s"].to_numpy(),
            "global_time_ms": follower["global_time_ms"].to_numpy(),
            "lane_id": follower["lane_id"].to_numpy(),
            "leader_id": leader_id.loc[has_leader].astype("int32").to_numpy(),
            "follower_id": follower["vehicle_id"].to_numpy(),
            "leader_y": leader_values("local_y_m").to_numpy(),
            "follower_y": follower["local_y_m"].to_numpy(),
            "leader_local_x_m": leader_values("local_x_m").to_numpy(),
            "follower_local_x_m": follower["local_x_m"].to_numpy(),
            "leader_length": leader_values("vehicle_length_m").to_numpy(),
            "follower_length": follower["vehicle_length_m"].to_numpy(),
            "leader_speed_m_s": leader_values("speed_m_s").to_numpy(),
            "follower_speed_m_s": follower["speed_m_s"].to_numpy(),
            "leader_acceleration_m_s2": leader_values(
                "acceleration_m_s2"
            ).to_numpy(),
            "follower_acceleration_m_s2": follower[
                "acceleration_m_s2"
            ].to_numpy(),
            "leader_class": leader_values("vehicle_class").astype("int8").to_numpy(),
            "follower_class": follower["vehicle_class"].to_numpy(),
        }
    )
    pairs["gap_m"] = pairs["leader_y"] - pairs["follower_y"] - pairs["leader_length"]
    pairs["relative_speed_m_s"] = (
        pairs["leader_speed_m_s"] - pairs["follower_speed_m_s"]
    )
    pairs["relative_acceleration_m_s2"] = (
        pairs["leader_acceleration_m_s2"] - pairs["follower_acceleration_m_s2"]
    )

    finite_columns = [
        "leader_y",
        "follower_y",
        "leader_length",
        "gap_m",
        "leader_speed_m_s",
        "follower_speed_m_s",
        "leader_acceleration_m_s2",
        "follower_acceleration_m_s2",
    ]
    finite_mask = np.isfinite(pairs[finite_columns].to_numpy()).all(axis=1)
    gap_mask = pairs["gap_m"].gt(minimum_gap_m) & pairs["gap_m"].le(maximum_gap_m)
    keep = finite_mask & gap_mask

    statistics = {
        "input_rows": int(len(observations)),
        "ambiguous_frame_lane_groups_excluded": int(excluded_group_count),
        "rows_in_ambiguous_groups_excluded": int(excluded_row_count),
        "adjacent_pairs_before_gap_checks": int(len(pairs)),
        "nonfinite_pairs_excluded": int((~finite_mask).sum()),
        "nonpositive_or_below_minimum_gap_pairs_excluded": int(
            (finite_mask & pairs["gap_m"].le(minimum_gap_m)).sum()
        ),
        "above_maximum_gap_pairs_excluded": int(
            (finite_mask & pairs["gap_m"].gt(maximum_gap_m)).sum()
        ),
        "valid_pairs": int(keep.sum()),
    }
    return pairs.loc[keep, PAIR_COLUMNS].reset_index(drop=True), statistics


def add_continuous_segment_ids(
    pairs: pd.DataFrame,
    sampling_interval_s: float,
    tolerance_s: float = 1e-6,
) -> pd.DataFrame:
    """Label maximal continuous runs for each leader/follower/lane triple."""

    required = {"leader_id", "follower_id", "lane_id", "frame_id", "time_s"}
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing columns required for segmentation: {missing}")
    ordered = pairs.sort_values(
        ["leader_id", "follower_id", "lane_id", "frame_id"],
        kind="mergesort",
    ).copy()
    keys = ["leader_id", "follower_id", "lane_id"]
    grouped = ordered.groupby(keys, sort=False, observed=True)
    frame_delta = grouped["frame_id"].diff()
    time_delta = grouped["time_s"].diff()
    new_segment = (
        frame_delta.isna()
        | frame_delta.ne(1)
        | time_delta.sub(float(sampling_interval_s)).abs().gt(tolerance_s)
    )
    ordered["segment_number"] = (
        new_segment.astype("int64").groupby([ordered[key] for key in keys]).cumsum() - 1
    )
    ordered["segment_id"] = (
        ordered["leader_id"].astype(str)
        + "_"
        + ordered["follower_id"].astype(str)
        + "_L"
        + ordered["lane_id"].astype(str)
        + "_S"
        + ordered["segment_number"].astype(str)
    )
    return ordered
