"""Convert cleaned point trajectories to piecewise-linear segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectorySegments:
    t0: np.ndarray
    t1: np.ndarray
    s0: np.ndarray
    s1: np.ndarray
    speed0: np.ndarray
    speed1: np.ndarray
    accel: np.ndarray
    vehicle_id: np.ndarray
    segment_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.t0.size)


def _downsample_group(group: pd.DataFrame, target_interval_s: float) -> pd.DataFrame:
    if len(group) <= 2 or target_interval_s <= 0.100001:
        return group
    times = group["time_s"].to_numpy(float)
    keep = [0]
    next_time = times[0] + target_interval_s - 1e-9
    for index in range(1, len(times) - 1):
        if times[index] >= next_time:
            keep.append(index)
            next_time = times[index] + target_interval_s - 1e-9
    if keep[-1] != len(group) - 1:
        keep.append(len(group) - 1)
    return group.iloc[keep]


def downsample_points(frame: pd.DataFrame, target_interval_s: float) -> pd.DataFrame:
    """Downsample each continuous segment while preserving both endpoints."""
    return (
        frame.groupby("trajectory_segment_id", group_keys=False, sort=False)
        .apply(_downsample_group, target_interval_s=float(target_interval_s))
        .reset_index(drop=True)
    )


def build_segments(frame: pd.DataFrame) -> TrajectorySegments:
    """Build valid line segments between consecutive points in each trajectory run."""
    ordered = frame.sort_values(
        ["trajectory_segment_id", "time_s", "frame_id"], kind="mergesort"
    ).copy()
    same = ordered["trajectory_segment_id"].eq(ordered["trajectory_segment_id"].shift(-1))
    dt = ordered["time_s"].shift(-1) - ordered["time_s"]
    valid = same & dt.gt(0)
    start = ordered.loc[valid]
    end = ordered.shift(-1).loc[valid]
    order = np.argsort(start["time_s"].to_numpy(float), kind="mergesort")

    def values(series: pd.Series, dtype: str) -> np.ndarray:
        return series.to_numpy(dtype=dtype)[order]

    return TrajectorySegments(
        t0=values(start["time_s"], "float64"),
        t1=values(end["time_s"], "float64"),
        s0=values(start["s_clean_m"], "float64"),
        s1=values(end["s_clean_m"], "float64"),
        speed0=values(start["speed_clean_m_s"], "float64"),
        speed1=values(end["speed_clean_m_s"], "float64"),
        accel=values((start["accel_clean_m_s2"] + end["accel_clean_m_s2"]) / 2.0, "float64"),
        vehicle_id=values(start["vehicle_id"], "int64"),
        segment_id=values(start["trajectory_segment_id"], "int64"),
    )
