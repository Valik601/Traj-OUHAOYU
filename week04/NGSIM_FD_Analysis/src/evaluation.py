"""Time-block generalization, physical checks, coverage, and stability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .fundamental_diagram import fit_greenshields, greenshields_flow


def _time_split(states: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    ordered_times = np.sort(states["t0_s"].dropna().unique())
    if ordered_times.size < 2:
        return states.iloc[0:0], states.iloc[0:0], np.nan
    split_index = min(max(int(np.floor(ordered_times.size * train_fraction)), 1), ordered_times.size - 1)
    cutoff = float(ordered_times[split_index])
    return states.loc[states["t0_s"] < cutoff], states.loc[states["t0_s"] >= cutoff], cutoff


def _capacity_stability(states: pd.DataFrame, blocks: int) -> tuple[float, list[float]]:
    times = states["t0_s"].to_numpy(float)
    if len(states) < blocks * 10 or np.nanmax(times) <= np.nanmin(times):
        return np.nan, []
    edges = np.linspace(float(np.nanmin(times)), float(np.nanmax(times)) + 1e-9, blocks + 1)
    capacities = []
    for index in range(blocks):
        block = states.loc[(states["t0_s"] >= edges[index]) & (states["t0_s"] < edges[index + 1])]
        fit = fit_greenshields(block)
        if fit.get("fit_success"):
            capacities.append(float(fit["capacity_veh_h"]))
    if len(capacities) < 2 or np.mean(capacities) == 0:
        return np.nan, capacities
    return float(np.std(capacities, ddof=1) / np.mean(capacities) * 100.0), capacities


def evaluate_method(
    method: str, states: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = states.loc[states["valid"].astype(bool)].copy()
    train, test, cutoff = _time_split(valid, float(config["evaluation"]["train_fraction"]))
    fit = fit_greenshields(train)
    mae = rmse = np.nan
    if fit.get("fit_success") and len(test):
        predicted = greenshields_flow(
            test["density_veh_km"].to_numpy(float),
            float(fit["free_flow_speed_km_h"]),
            float(fit["jam_density_veh_km"]),
        )
        errors = predicted - test["flow_veh_h"].to_numpy(float)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
    stability, block_capacities = _capacity_stability(
        valid, int(config["evaluation"]["stability_time_blocks"])
    )
    full_fit = fit_greenshields(valid)
    critical = float(full_fit.get("critical_density_veh_km", np.nan))
    free_samples = int((valid["density_veh_km"] <= critical).sum()) if np.isfinite(critical) else 0
    congested_samples = int((valid["density_veh_km"] > critical).sum()) if np.isfinite(critical) else 0
    metric = {
        "method": method,
        "candidate_windows": int(len(states)),
        "valid_samples": int(len(valid)),
        "valid_window_fraction": float(len(valid) / len(states)) if len(states) else 0.0,
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "time_split_s": cutoff,
        "test_mae_veh_h": mae,
        "test_rmse_veh_h": rmse,
        "free_flow_speed_km_h": full_fit.get("free_flow_speed_km_h", np.nan),
        "critical_density_veh_km": full_fit.get("critical_density_veh_km", np.nan),
        "capacity_veh_h": full_fit.get("capacity_veh_h", np.nan),
        "capacity_stability_cv_percent": stability,
        "free_branch_samples": free_samples,
        "congested_branch_samples": congested_samples,
        "max_q_kv_relative_error": float(valid["closure_relative_error"].max()) if len(valid) else np.nan,
    }
    detail = {
        "train_fit": fit,
        "full_fit": full_fit,
        "block_capacities_veh_h": block_capacities,
    }
    return metric, detail


def evaluate_methods(
    states_by_method: dict[str, pd.DataFrame], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = []
    details: dict[str, Any] = {}
    for method, states in states_by_method.items():
        metric, detail = evaluate_method(method, states, config)
        metrics.append(metric)
        details[method] = detail
    return pd.DataFrame.from_records(metrics), details


def parameter_sensitivity(sweep: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, length_m, duration_s), group in sweep.groupby(
        ["method", "length_m", "duration_s"], sort=True
    ):
        fit = fit_greenshields(group)
        records.append(
            {
                "method": method,
                "length_m": float(length_m),
                "duration_s": float(duration_s),
                "valid_samples": int(group["valid"].sum()),
                "capacity_veh_h": fit.get("capacity_veh_h", np.nan),
                "critical_density_veh_km": fit.get("critical_density_veh_km", np.nan),
                "free_flow_speed_km_h": fit.get("free_flow_speed_km_h", np.nan),
            }
        )
    return pd.DataFrame.from_records(records)
