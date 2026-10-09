"""Trajectory quality control and lane-specific segment reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


REQUIRED = [
    "vehicle_id", "frame_id", "time_s", "global_time_ms", "s_m", "speed_m_s",
    "accel_m_s2", "lane_id",
]


def _drop_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    keys = ["vehicle_id", "frame_id"]
    duplicated = frame.duplicated(keys, keep=False)
    conflicting = 0
    if duplicated.any():
        check_columns = [column for column in frame.columns if column not in keys]
        variants = frame.loc[duplicated].groupby(keys, sort=False)[check_columns].nunique(dropna=False)
        conflicting = int((variants.max(axis=1) > 1).sum())
        if conflicting:
            raise ValueError(f"Found {conflicting} conflicting duplicate primary keys")
    exact_extra = int(frame.duplicated(keys, keep="first").sum())
    return frame.drop_duplicates(keys, keep="first").copy(), exact_extra, conflicting


def _smooth_group(group: pd.DataFrame, window: int, polyorder: int) -> pd.DataFrame:
    result = group.copy()
    length = len(result)
    usable = min(window, length if length % 2 else length - 1)
    if usable < max(polyorder + 2, 5):
        return result
    result["s_clean_m"] = savgol_filter(result["s_raw_m"], usable, polyorder, mode="interp")
    result["speed_clean_m_s"] = savgol_filter(
        result["speed_raw_m_s"], usable, polyorder, mode="interp"
    )
    result["accel_clean_m_s2"] = savgol_filter(
        result["accel_raw_m_s2"], usable, polyorder, mode="interp"
    )
    return result


def preprocess_lane(
    raw: pd.DataFrame, lane_id: int, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean rows, retain target-lane runs, and assign continuous segment IDs."""
    cfg = config["preprocessing"]
    summary: dict[str, Any] = {"input_rows": int(len(raw)), "target_lane": int(lane_id)}
    work, exact_duplicates, conflicting = _drop_exact_duplicates(raw)
    summary["exact_duplicate_rows_removed"] = exact_duplicates
    summary["conflicting_duplicate_keys"] = conflicting

    missing_by_column = {column: int(work[column].isna().sum()) for column in REQUIRED}
    missing_mask = work[REQUIRED].isna().any(axis=1)
    summary["missing_by_column"] = missing_by_column
    summary["rows_removed_for_missing_required_values"] = int(missing_mask.sum())
    work = work.loc[~missing_mask].copy()

    integer_columns = ["vehicle_id", "frame_id", "global_time_ms", "vehicle_class", "lane_id"]
    for column in integer_columns:
        work[column] = work[column].astype("int64")
    work = work.sort_values(["vehicle_id", "time_s", "frame_id"], kind="mergesort").reset_index(drop=True)

    grouped = work.groupby("vehicle_id", sort=False)
    dt = grouped["time_s"].diff()
    ds = grouped["s_m"].diff()
    lane_change = grouped["lane_id"].diff().fillna(0).ne(0)
    implied_speed = ds / dt
    base_break = (
        grouped.cumcount().eq(0)
        | lane_change
        | dt.le(0)
        | dt.gt(float(cfg["max_time_gap_s"]))
    )
    work["source_lane_run_id"] = base_break.cumsum().astype("int64")

    anomaly_masks = {
        "negative_or_high_speed": work["speed_m_s"].lt(float(cfg["min_speed_m_s"]))
        | work["speed_m_s"].gt(float(cfg["max_speed_m_s"])),
        "extreme_acceleration": work["accel_m_s2"].abs().gt(float(cfg["max_abs_accel_m_s2"])),
        "backward_position_jump": ds.lt(-float(cfg["max_backward_jump_m"])),
        "excessive_position_jump": implied_speed.abs().gt(float(cfg["max_implied_speed_m_s"])),
        "nonpositive_time_step": dt.le(0),
    }
    for name, mask in anomaly_masks.items():
        summary[f"rows_flagged_{name}"] = int(mask.fillna(False).sum())
    anomaly = np.logical_or.reduce([mask.fillna(False).to_numpy() for mask in anomaly_masks.values()])
    summary["unique_anomalous_rows_removed"] = int(anomaly.sum())
    work = work.loc[~anomaly].copy()

    lane = work.loc[work["lane_id"].eq(int(lane_id))].copy()
    summary["target_lane_rows_before_short_segment_filter"] = int(len(lane))
    lane = lane.sort_values(["vehicle_id", "time_s", "frame_id"], kind="mergesort").reset_index(drop=True)
    lane_group = lane.groupby("vehicle_id", sort=False)
    lane_dt = lane_group["time_s"].diff()
    run_changed = lane_group["source_lane_run_id"].diff().fillna(0).ne(0)
    segment_break = (
        lane_group.cumcount().eq(0)
        | run_changed
        | lane_dt.le(0)
        | lane_dt.gt(float(cfg["max_time_gap_s"]))
    )
    lane["trajectory_segment_id"] = segment_break.cumsum().astype("int64")
    segment_sizes = lane.groupby("trajectory_segment_id")["trajectory_segment_id"].transform("size")
    short_rows = segment_sizes.lt(2)
    summary["single_point_segment_rows_removed"] = int(short_rows.sum())
    lane = lane.loc[~short_rows].copy()

    lane["s_raw_m"] = lane["s_m"]
    lane["speed_raw_m_s"] = lane["speed_m_s"]
    lane["accel_raw_m_s2"] = lane["accel_m_s2"]
    lane["s_clean_m"] = lane["s_raw_m"]
    lane["speed_clean_m_s"] = lane["speed_raw_m_s"]
    lane["accel_clean_m_s2"] = lane["accel_raw_m_s2"]

    smooth_cfg = cfg["smoothing"]
    if bool(smooth_cfg.get("enabled", False)):
        lane = (
            lane.groupby("trajectory_segment_id", group_keys=False, sort=False)
            .apply(
                _smooth_group,
                window=int(smooth_cfg["window_length"]),
                polyorder=int(smooth_cfg["polyorder"]),
            )
            .reset_index(drop=True)
        )
    summary["smoothing_enabled"] = bool(smooth_cfg.get("enabled", False))
    summary["rows_corrected_by_smoothing"] = int(
        np.count_nonzero(~np.isclose(lane["s_clean_m"], lane["s_raw_m"], rtol=0, atol=1e-12))
    )
    summary["output_rows"] = int(len(lane))
    summary["continuous_segments"] = int(lane["trajectory_segment_id"].nunique())
    summary["vehicles_retained"] = int(lane["vehicle_id"].nunique())
    summary["time_range_s"] = [float(lane["time_s"].min()), float(lane["time_s"].max())]
    summary["space_range_m"] = [float(lane["s_clean_m"].min()), float(lane["s_clean_m"].max())]
    return lane, summary
