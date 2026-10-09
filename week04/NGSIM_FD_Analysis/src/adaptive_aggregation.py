"""Exploratory adaptive space-time window selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .edie_calculation import edie_state, parallelogram_polygon, rectangle_polygon
from .trajectory_processing import TrajectorySegments


def _stability_score(state: dict[str, Any], length_m: float, duration_s: float, config: dict[str, Any]) -> float:
    adaptive = config["aggregation"]["adaptive"]
    weights = adaptive["weights"]
    terms = [
        float(state["speed_cv"]),
        float(state["temporal_speed_change_m_s"]) / float(adaptive["speed_reference_m_s"]),
        abs(float(state["mean_accel_m_s2"])) / float(adaptive["acceleration_reference_m_s2"]),
        float(state["spatial_heterogeneity"]),
    ]
    if not np.isfinite(terms).all():
        return float("inf")
    base = (
        float(weights["speed_cv"]) * terms[0]
        + float(weights["temporal_change"]) * terms[1]
        + float(weights["mean_acceleration"]) * terms[2]
        + float(weights["spatial_heterogeneity"]) * terms[3]
    )
    max_l = max(float(value) for value in config["aggregation"]["spatial_lengths_m"])
    max_t = max(float(value) for value in config["aggregation"]["temporal_lengths_s"])
    scale_penalty = float(adaptive["scale_penalty"]) * (
        max_l / float(length_m) - 1.0 + max_t / float(duration_s) - 1.0
    )
    return float(base + scale_penalty)


def _state_class(state: dict[str, Any], score: float, config: dict[str, Any]) -> str:
    adaptive = config["aggregation"]["adaptive"]
    if score <= float(adaptive["quasi_stationary_threshold"]):
        return "quasi_stationary"
    acceleration = float(state["mean_accel_m_s2"])
    threshold = float(adaptive["acceleration_threshold_m_s2"])
    if acceleration > threshold:
        return "acceleration"
    if acceleration < -threshold:
        return "deceleration"
    return "transient"


def adaptive_states(
    segments: TrajectorySegments,
    anchors: pd.DataFrame,
    wave_speed_m_s: float,
    config: dict[str, Any],
    *,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Search size and orientation per anchor; retain all selected state classes."""
    aggregation = config["aggregation"]
    orientations = aggregation["adaptive"]["orientations"]
    selected_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    total = len(anchors)

    for count, anchor in enumerate(anchors.itertuples(index=False), start=1):
        local_candidates: list[dict[str, Any]] = []
        for length_m in aggregation["spatial_lengths_m"]:
            for duration_s in aggregation["temporal_lengths_s"]:
                for orientation in orientations:
                    wave = 0.0 if orientation == "rectangular" else float(wave_speed_m_s)
                    polygon = (
                        rectangle_polygon(anchor.t0_s, anchor.s0_m, duration_s, length_m)
                        if wave == 0.0
                        else parallelogram_polygon(anchor.t0_s, anchor.s0_m, duration_s, length_m, wave)
                    )
                    state = edie_state(
                        segments,
                        polygon,
                        t_start=float(anchor.t0_s),
                        s_start=float(anchor.s0_m),
                        duration_s=float(duration_s),
                        length_m=float(length_m),
                        wave_speed_m_s=wave,
                        min_vehicles=int(aggregation["min_vehicles"]),
                        minimum_occupied_time_fraction=float(aggregation["minimum_occupied_time_fraction"]),
                    )
                    score = _stability_score(state, float(length_m), float(duration_s), config)
                    state.update(
                        {
                            "method": "Adaptive",
                            "anchor_id": int(anchor.anchor_id),
                            "t0_s": float(anchor.t0_s),
                            "s0_m": float(anchor.s0_m),
                            "length_m": float(length_m),
                            "duration_s": float(duration_s),
                            "wave_speed_m_s": wave,
                            "orientation": orientation,
                            "stability_score": score,
                        }
                    )
                    local_candidates.append(state)
                    candidate_records.append(state.copy())

        valid = [candidate for candidate in local_candidates if candidate["valid"] and np.isfinite(candidate["stability_score"])]
        if valid:
            choice = min(valid, key=lambda candidate: candidate["stability_score"])
            choice = choice.copy()
            choice["state_class"] = _state_class(choice, float(choice["stability_score"]), config)
            choice["valid_candidate_count"] = len(valid)
            selected_records.append(choice)
        elif local_candidates:
            choice = max(local_candidates, key=lambda candidate: candidate["vehicle_count"]).copy()
            choice["state_class"] = "insufficient_coverage"
            choice["valid_candidate_count"] = 0
            selected_records.append(choice)
        if progress and (count % 25 == 0 or count == total):
            print(f"[adaptive] {count}/{total} anchors", flush=True)

    return pd.DataFrame.from_records(selected_records), pd.DataFrame.from_records(candidate_records)
