"""Rectangular aggregation and shared window/anchor utilities."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .edie_calculation import edie_state, parallelogram_polygon, rectangle_polygon
from .trajectory_processing import TrajectorySegments


def generate_common_anchors(
    time_range_s: tuple[float, float],
    space_range_m: tuple[float, float],
    wave_speed_m_s: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Generate anchors where every configured candidate remains inside observations."""
    cfg = config["aggregation"]
    max_t = max(float(value) for value in cfg["temporal_lengths_s"])
    max_l = max(float(value) for value in cfg["spatial_lengths_m"])
    dt = float(cfg["anchor_time_step_s"])
    ds = float(cfg["anchor_space_step_m"])
    t_min, t_max = time_range_s
    s_min, s_max = space_range_m
    min_wave = min(0.0, float(wave_speed_m_s))
    max_wave = max(0.0, float(wave_speed_m_s))
    safe_s_min = s_min + max(0.0, -min_wave * max_t)
    safe_s_max = s_max - max_l - max(0.0, max_wave * max_t)
    safe_t_max = t_max - max_t
    first_t = np.ceil(t_min / dt) * dt
    first_s = np.ceil(safe_s_min / ds) * ds
    times = np.arange(first_t, safe_t_max + 1e-9, dt)
    spaces = np.arange(first_s, safe_s_max + 1e-9, ds)
    if not times.size or not spaces.size:
        raise ValueError("Observation domain is too small for configured candidate windows")
    records = []
    for time_value in times:
        for space_value in spaces:
            records.append(
                {"anchor_id": len(records), "t0_s": float(time_value), "s0_m": float(space_value)}
            )
    return pd.DataFrame.from_records(records)


def aggregate_fixed_windows(
    segments: TrajectorySegments,
    anchors: pd.DataFrame,
    *,
    method: str,
    length_m: float,
    duration_s: float,
    wave_speed_m_s: float,
    config: dict[str, Any],
    progress_label: str | None = None,
) -> pd.DataFrame:
    cfg = config["aggregation"]
    records: list[dict[str, Any]] = []
    total = len(anchors)
    for count, anchor in enumerate(anchors.itertuples(index=False), start=1):
        if wave_speed_m_s == 0.0:
            polygon = rectangle_polygon(anchor.t0_s, anchor.s0_m, duration_s, length_m)
        else:
            polygon = parallelogram_polygon(
                anchor.t0_s, anchor.s0_m, duration_s, length_m, wave_speed_m_s
            )
        state = edie_state(
            segments,
            polygon,
            t_start=float(anchor.t0_s),
            s_start=float(anchor.s0_m),
            duration_s=float(duration_s),
            length_m=float(length_m),
            wave_speed_m_s=float(wave_speed_m_s),
            min_vehicles=int(cfg["min_vehicles"]),
            minimum_occupied_time_fraction=float(cfg["minimum_occupied_time_fraction"]),
        )
        state.update(
            {
                "method": method,
                "anchor_id": int(anchor.anchor_id),
                "t0_s": float(anchor.t0_s),
                "s0_m": float(anchor.s0_m),
                "length_m": float(length_m),
                "duration_s": float(duration_s),
                "wave_speed_m_s": float(wave_speed_m_s),
            }
        )
        records.append(state)
        if progress_label and (count % 100 == 0 or count == total):
            print(f"[{progress_label}] {count}/{total} windows", flush=True)
    return pd.DataFrame.from_records(records)


def rectangular_states(
    segments: TrajectorySegments,
    anchors: pd.DataFrame,
    config: dict[str, Any],
    *,
    length_m: float | None = None,
    duration_s: float | None = None,
) -> pd.DataFrame:
    cfg = config["aggregation"]
    return aggregate_fixed_windows(
        segments,
        anchors,
        method="Rectangular",
        length_m=float(length_m if length_m is not None else cfg["reference_length_m"]),
        duration_s=float(duration_s if duration_s is not None else cfg["reference_duration_s"]),
        wave_speed_m_s=0.0,
        config=config,
        progress_label="rectangular",
    )


def rectangular_parameter_sweep(
    segments: TrajectorySegments, anchors: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    frames = []
    cfg = config["aggregation"]
    for length_m in cfg["spatial_lengths_m"]:
        for duration_s in cfg["temporal_lengths_s"]:
            frames.append(
                aggregate_fixed_windows(
                    segments,
                    anchors,
                    method="Rectangular",
                    length_m=float(length_m),
                    duration_s=float(duration_s),
                    wave_speed_m_s=0.0,
                    config=config,
                )
            )
    return pd.concat(frames, ignore_index=True)

