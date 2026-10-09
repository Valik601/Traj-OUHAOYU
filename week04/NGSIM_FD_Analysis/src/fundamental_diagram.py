"""Classical reference-model fitting for empirical traffic states."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def greenshields_flow(density: np.ndarray, free_flow_speed: float, jam_density: float) -> np.ndarray:
    return free_flow_speed * density * (1.0 - density / jam_density)


def greenshields_speed(density: np.ndarray, free_flow_speed: float, jam_density: float) -> np.ndarray:
    return free_flow_speed * (1.0 - density / jam_density)


def fit_greenshields(states: pd.DataFrame) -> dict[str, Any]:
    """Fit q(k) robustly; fitting is separate from state extraction quality."""
    clean = states.loc[
        states["valid"].astype(bool)
        & np.isfinite(states["density_veh_km"])
        & np.isfinite(states["flow_veh_h"])
        & states["density_veh_km"].gt(0)
        & states["flow_veh_h"].ge(0)
    ]
    if len(clean) < 10:
        return {
            "fit_success": False, "free_flow_speed_km_h": np.nan,
            "jam_density_veh_km": np.nan, "critical_density_veh_km": np.nan,
            "capacity_veh_h": np.nan, "sample_count": int(len(clean)),
        }
    density = clean["density_veh_km"].to_numpy(float)
    flow = clean["flow_veh_h"].to_numpy(float)
    max_density = float(np.max(density))
    speed_guess = float(np.clip(clean["speed_km_h"].quantile(0.9), 20.0, 140.0))
    jam_lower = max(max_density * 1.01, 30.0)
    jam_upper = max(500.0, max_density * 3.0)
    initial = np.array([speed_guess, min(max(max_density * 1.5, jam_lower * 1.05), jam_upper * 0.95)])

    def residual(parameters: np.ndarray) -> np.ndarray:
        return greenshields_flow(density, parameters[0], parameters[1]) - flow

    result = least_squares(
        residual,
        initial,
        bounds=(np.array([1.0, jam_lower]), np.array([200.0, jam_upper])),
        loss="soft_l1",
        f_scale=max(float(np.std(flow)), 100.0),
        max_nfev=5000,
    )
    free_flow_speed, jam_density = (float(result.x[0]), float(result.x[1]))
    critical_density = jam_density / 2.0
    capacity = free_flow_speed * jam_density / 4.0
    return {
        "fit_success": bool(result.success),
        "free_flow_speed_km_h": free_flow_speed,
        "jam_density_veh_km": jam_density,
        "critical_density_veh_km": critical_density,
        "capacity_veh_h": capacity,
        "sample_count": int(len(clean)),
        "cost": float(result.cost),
    }


def model_curve(parameters: dict[str, Any], points: int = 200) -> pd.DataFrame:
    if not parameters.get("fit_success", False):
        return pd.DataFrame(columns=["density_veh_km", "flow_veh_h", "speed_km_h"])
    jam_density = float(parameters["jam_density_veh_km"])
    free_flow = float(parameters["free_flow_speed_km_h"])
    density = np.linspace(0.0, jam_density, points)
    return pd.DataFrame(
        {
            "density_veh_km": density,
            "flow_veh_h": greenshields_flow(density, free_flow, jam_density),
            "speed_km_h": greenshields_speed(density, free_flow, jam_density),
        }
    )

