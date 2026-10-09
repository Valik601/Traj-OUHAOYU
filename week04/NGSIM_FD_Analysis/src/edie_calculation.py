"""Exact piecewise-linear trajectory clipping and Edie traffic states."""

from __future__ import annotations

from typing import Any

import numpy as np

from .trajectory_processing import TrajectorySegments


def rectangle_polygon(t0: float, s0: float, duration_s: float, length_m: float) -> np.ndarray:
    return np.array(
        [[t0, s0], [t0 + duration_s, s0], [t0 + duration_s, s0 + length_m], [t0, s0 + length_m]],
        dtype=float,
    )


def parallelogram_polygon(
    t0: float, s0: float, duration_s: float, length_m: float, wave_speed_m_s: float
) -> np.ndarray:
    shift = wave_speed_m_s * duration_s
    return np.array(
        [
            [t0, s0],
            [t0 + duration_s, s0 + shift],
            [t0 + duration_s, s0 + length_m + shift],
            [t0, s0 + length_m],
        ],
        dtype=float,
    )


def polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _counter_clockwise(polygon: np.ndarray) -> np.ndarray:
    x = polygon[:, 0]
    y = polygon[:, 1]
    signed = float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
    return polygon if signed > 0 else polygon[::-1].copy()


def clip_segments_to_convex_polygon(
    segments: TrajectorySegments, polygon: np.ndarray
) -> dict[str, np.ndarray]:
    """Clip every relevant trajectory segment to a convex time-space polygon."""
    poly = _counter_clockwise(np.asarray(polygon, dtype=float))
    min_t = float(poly[:, 0].min())
    max_t = float(poly[:, 0].max())
    max_dt = float(np.max(segments.t1 - segments.t0)) if segments.size else 0.0
    lo = int(np.searchsorted(segments.t0, min_t - max_dt - 1e-9, side="left"))
    hi = int(np.searchsorted(segments.t0, max_t + 1e-9, side="right"))
    base = np.arange(lo, hi, dtype=int)
    if base.size == 0:
        return {"indices": base, "lambda0": np.empty(0), "lambda1": np.empty(0)}

    overlap = (segments.t1[base] > min_t + 1e-12) & (segments.t0[base] < max_t - 1e-12)
    indices = base[overlap]
    if indices.size == 0:
        return {"indices": indices, "lambda0": np.empty(0), "lambda1": np.empty(0)}

    p_t = segments.t0[indices]
    p_s = segments.s0[indices]
    d_t = segments.t1[indices] - p_t
    d_s = segments.s1[indices] - p_s
    lower = np.zeros(indices.size, dtype=float)
    upper = np.ones(indices.size, dtype=float)
    alive = np.ones(indices.size, dtype=bool)
    eps = 1e-12

    for vertex_a, vertex_b in zip(poly, np.roll(poly, -1, axis=0)):
        edge_t = vertex_b[0] - vertex_a[0]
        edge_s = vertex_b[1] - vertex_a[1]
        c0 = edge_t * (p_s - vertex_a[1]) - edge_s * (p_t - vertex_a[0])
        c1 = edge_t * d_s - edge_s * d_t
        parallel = np.abs(c1) <= eps
        alive &= ~(parallel & (c0 < -eps))
        positive = c1 > eps
        negative = c1 < -eps
        lower[positive] = np.maximum(lower[positive], -c0[positive] / c1[positive])
        upper[negative] = np.minimum(upper[negative], -c0[negative] / c1[negative])

    keep = alive & (upper - lower > 1e-10) & (upper > 0) & (lower < 1)
    lower = np.clip(lower[keep], 0.0, 1.0)
    upper = np.clip(upper[keep], 0.0, 1.0)
    final = upper - lower > 1e-10
    return {
        "indices": indices[keep][final],
        "lambda0": lower[final],
        "lambda1": upper[final],
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0 else float("nan")


def edie_state(
    segments: TrajectorySegments,
    polygon: np.ndarray,
    *,
    t_start: float,
    s_start: float,
    duration_s: float,
    length_m: float,
    wave_speed_m_s: float,
    min_vehicles: int,
    minimum_occupied_time_fraction: float,
) -> dict[str, Any]:
    """Compute exact distance/time intersections and Edie's q, k, and v."""
    clipped = clip_segments_to_convex_polygon(segments, polygon)
    idx = clipped["indices"]
    area = polygon_area(polygon)
    empty = {
        "distance_sum_m": 0.0, "time_sum_s": 0.0, "area_m_s": area,
        "flow_veh_h": np.nan, "density_veh_km": np.nan, "speed_km_h": np.nan,
        "closure_relative_error": np.nan, "vehicle_count": 0,
        "occupied_time_fraction": 0.0, "speed_cv": np.nan,
        "temporal_speed_change_m_s": np.nan, "mean_accel_m_s2": np.nan,
        "mean_abs_accel_m_s2": np.nan, "spatial_heterogeneity": np.nan,
        "valid": False,
    }
    if idx.size == 0 or area <= 0:
        return empty

    lam0 = clipped["lambda0"]
    lam1 = clipped["lambda1"]
    fraction = lam1 - lam0
    full_dt = segments.t1[idx] - segments.t0[idx]
    full_ds = segments.s1[idx] - segments.s0[idx]
    clipped_dt = fraction * full_dt
    clipped_ds = fraction * full_ds
    total_time = float(clipped_dt.sum())
    total_distance = float(clipped_ds.sum())
    if total_time <= 0:
        return empty

    q = total_distance / area * 3600.0
    k = total_time / area * 1000.0
    v = total_distance / total_time * 3.6
    closure = abs(q - k * v) / max(abs(q), 1e-12)

    clip_t0 = segments.t0[idx] + lam0 * full_dt
    clip_t1 = segments.t0[idx] + lam1 * full_dt
    clip_s0 = segments.s0[idx] + lam0 * full_ds
    clip_s1 = segments.s0[idx] + lam1 * full_ds
    mid_t = (clip_t0 + clip_t1) / 2.0
    mid_s = (clip_s0 + clip_s1) / 2.0
    observed_speed = (segments.speed0[idx] + segments.speed1[idx]) / 2.0
    speed_mean = _weighted_mean(observed_speed, clipped_dt)
    speed_var = _weighted_mean((observed_speed - speed_mean) ** 2, clipped_dt)
    speed_cv = float(np.sqrt(max(speed_var, 0.0)) / max(abs(speed_mean), 1e-6))

    midpoint = t_start + duration_s / 2.0
    first_weights = np.maximum(0.0, np.minimum(clip_t1, midpoint) - clip_t0)
    second_weights = np.maximum(0.0, clip_t1 - np.maximum(clip_t0, midpoint))
    first_mean = _weighted_mean(observed_speed, first_weights)
    second_mean = _weighted_mean(observed_speed, second_weights)
    temporal_change = (
        abs(first_mean - second_mean)
        if np.isfinite(first_mean) and np.isfinite(second_mean)
        else float("nan")
    )

    relative_s = mid_s - (s_start + wave_speed_m_s * (mid_t - t_start))
    spatial_bin = np.clip((relative_s / max(length_m, 1e-9) * 3).astype(int), 0, 2)
    spatial_means = [
        _weighted_mean(observed_speed[spatial_bin == band], clipped_dt[spatial_bin == band])
        for band in range(3)
        if np.any(spatial_bin == band)
    ]
    spatial_heterogeneity = (
        float(np.std(spatial_means, ddof=0) / max(abs(speed_mean), 1e-6))
        if len(spatial_means) >= 2 else 0.0
    )
    mean_accel = _weighted_mean(segments.accel[idx], clipped_dt)
    mean_abs_accel = _weighted_mean(np.abs(segments.accel[idx]), clipped_dt)
    occupied_bins = np.unique(np.clip(np.floor(mid_t - t_start).astype(int), 0, max(int(np.ceil(duration_s)) - 1, 0)))
    occupied_fraction = min(1.0, float(occupied_bins.size / max(int(np.ceil(duration_s)), 1)))
    vehicle_count = int(np.unique(segments.vehicle_id[idx]).size)

    return {
        "distance_sum_m": total_distance,
        "time_sum_s": total_time,
        "area_m_s": area,
        "flow_veh_h": q,
        "density_veh_km": k,
        "speed_km_h": v,
        "closure_relative_error": closure,
        "vehicle_count": vehicle_count,
        "occupied_time_fraction": occupied_fraction,
        "speed_cv": speed_cv,
        "temporal_speed_change_m_s": float(temporal_change),
        "mean_accel_m_s2": float(mean_accel),
        "mean_abs_accel_m_s2": float(mean_abs_accel),
        "spatial_heterogeneity": spatial_heterogeneity,
        "valid": bool(
            vehicle_count >= min_vehicles
            and occupied_fraction >= minimum_occupied_time_fraction
            and np.isfinite([q, k, v]).all()
        ),
    }
