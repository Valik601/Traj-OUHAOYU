"""Wave-speed estimation and simplified parallelogram aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .rectangular_aggregation import aggregate_fixed_windows
from .trajectory_processing import TrajectorySegments


def estimate_wave_speed(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[float, pd.DataFrame]:
    """Estimate upstream propagation speed by lagged speed-field correlation."""
    cfg = config["aggregation"]["wave_speed"]
    manual = cfg.get("manual_m_s")
    if manual is not None:
        return float(manual), pd.DataFrame([{"wave_speed_m_s": float(manual), "score": np.nan, "pairs": 0}])

    dt = float(cfg["grid_time_s"])
    dx = float(cfg["grid_space_m"])
    t0 = float(frame["time_s"].min())
    s0 = float(frame["s_clean_m"].min())
    work = frame[["time_s", "s_clean_m", "speed_clean_m_s"]].copy()
    work["tb"] = np.floor((work["time_s"] - t0) / dt).astype(int)
    work["sb"] = np.floor((work["s_clean_m"] - s0) / dx).astype(int)
    grouped = work.groupby(["tb", "sb"])["speed_clean_m_s"].mean()
    nt = int(work["tb"].max()) + 1
    ns = int(work["sb"].max()) + 1
    field = np.full((nt, ns), np.nan, dtype=float)
    for (tb, sb), value in grouped.items():
        field[int(tb), int(sb)] = float(value)

    candidates = np.arange(
        float(cfg["candidate_min_m_s"]),
        float(cfg["candidate_max_m_s"]) + float(cfg["candidate_step_m_s"]) / 2,
        float(cfg["candidate_step_m_s"]),
    )
    threshold = float(cfg["congested_speed_threshold_m_s"])
    records = []
    for wave in candidates:
        weighted_score = 0.0
        total_pairs = 0
        for lag in range(1, int(cfg["max_lag_steps"]) + 1):
            shift = int(round(wave * lag * dt / dx))
            if shift < 0:
                first = field[:-lag, -shift:]
                second = field[lag:, : ns + shift]
            elif shift > 0:
                first = field[:-lag, : ns - shift]
                second = field[lag:, shift:]
            else:
                first = field[:-lag]
                second = field[lag:]
            mask = np.isfinite(first) & np.isfinite(second) & ((first <= threshold) | (second <= threshold))
            if int(mask.sum()) < 50:
                continue
            a = first[mask]
            b = second[mask]
            a = a - a.mean()
            b = b - b.mean()
            denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
            if denominator <= 0:
                continue
            correlation = float(np.dot(a, b) / denominator)
            pairs = int(mask.sum())
            weighted_score += correlation * pairs
            total_pairs += pairs
        score = weighted_score / total_pairs if total_pairs else np.nan
        records.append({"wave_speed_m_s": float(wave), "score": score, "pairs": total_pairs})
    diagnostics = pd.DataFrame.from_records(records)
    valid = diagnostics.dropna(subset=["score"])
    estimated = float(valid.loc[valid["score"].idxmax(), "wave_speed_m_s"]) if len(valid) else -4.44
    return estimated, diagnostics


def parallelogram_states(
    segments: TrajectorySegments,
    anchors: pd.DataFrame,
    wave_speed_m_s: float,
    config: dict[str, Any],
    *,
    length_m: float | None = None,
    duration_s: float | None = None,
) -> pd.DataFrame:
    cfg = config["aggregation"]
    return aggregate_fixed_windows(
        segments,
        anchors,
        method="Parallelogram",
        length_m=float(length_m if length_m is not None else cfg["reference_length_m"]),
        duration_s=float(duration_s if duration_s is not None else cfg["reference_duration_s"]),
        wave_speed_m_s=float(wave_speed_m_s),
        config=config,
        progress_label="parallelogram",
    )


def parallelogram_parameter_sweep(
    segments: TrajectorySegments,
    anchors: pd.DataFrame,
    wave_speed_m_s: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    frames = []
    cfg = config["aggregation"]
    for length_m in cfg["spatial_lengths_m"]:
        for duration_s in cfg["temporal_lengths_s"]:
            frames.append(
                aggregate_fixed_windows(
                    segments,
                    anchors,
                    method="Parallelogram",
                    length_m=float(length_m),
                    duration_s=float(duration_s),
                    wave_speed_m_s=float(wave_speed_m_s),
                    config=config,
                )
            )
    return pd.concat(frames, ignore_index=True)
