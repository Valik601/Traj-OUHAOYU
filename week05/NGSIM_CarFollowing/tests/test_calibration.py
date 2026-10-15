"""Small synthetic tests for the optional calibration stage."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_calibration():
    name = "test_target_calibration"
    if name in sys.modules:
        return sys.modules[name]
    path = PROJECT_ROOT / "src" / "07_calibrate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CAL = _load_calibration()
TRUE_OVM = {"kappa": 0.6, "v_max": 30.0, "s_c": 22.0, "ell": 7.0}


def _synthetic_trajectory(sample_count: int = 18, dt: float = 0.1) -> pd.DataFrame:
    t = np.arange(sample_count, dtype=float) * dt
    leader_x = 100.0 + 15.0 * t
    seed = pd.DataFrame(
        {
            "t": t,
            "leader_x": leader_x,
            "leader_v": np.full(sample_count, 15.0),
            "leader_a": np.zeros(sample_count),
            "follower_x_true": 70.0 + 10.0 * t,
            "follower_v_true": np.full(sample_count, 10.0),
            "follower_a_true": np.zeros(sample_count),
            "leader_length": np.full(sample_count, 5.0),
            "gap_true": 25.0 + 5.0 * t,
        }
    )
    simulated, status = CAL.SIMULATION.simulate_model(
        seed,
        "OVM",
        TRUE_OVM,
        maximum_abs_acceleration_m_s2=100.0,
    )
    assert status["completed"]
    return pd.DataFrame(
        {
            "t": t,
            "leader_x": leader_x,
            "leader_v": np.full(sample_count, 15.0),
            "leader_a": np.zeros(sample_count),
            "follower_x_true": simulated["x_sim"],
            "follower_v_true": simulated["v_sim"],
            "follower_a_true": simulated["a_sim"],
            "leader_length": np.full(sample_count, 5.0),
            "gap_true": simulated["gap_sim"],
        }
    )


def test_missing_calibration_section_uses_small_de_defaults() -> None:
    settings = CAL.build_calibration_settings({})
    assert settings["optimizer"] == "differential_evolution"
    assert settings["optimizer_options"]["maxiter"] <= 10
    assert settings["optimizer_options"]["polish"] is False


def test_known_parameters_have_zero_fit_objective() -> None:
    trajectory = _synthetic_trajectory()
    settings = CAL.build_calibration_settings({})
    scales = CAL._normalization_scales(trajectory, settings)
    evaluated = CAL.evaluate_parameters(
        trajectory,
        "OVM",
        TRUE_OVM,
        settings=settings,
        scales=scales,
        simulation_settings={
            "time_step_s": 0.1,
            "collision_gap_m": 0.0,
            "maximum_abs_acceleration_m_s2": 100.0,
            "stop_on_unreasonable_acceleration": True,
            "time_tolerance_s": 1.0e-6,
        },
    )
    assert evaluated["metrics"]["completed"] is True
    assert evaluated["metrics"]["objective_value"] == pytest.approx(0.0, abs=1.0e-12)


def test_least_squares_optimizer_branch_runs() -> None:
    trajectory = _synthetic_trajectory()
    config = {
        "models": {
            "OVM": {"kappa": 0.5, "v_max": 28.0, "s_c": 25.0, "ell": 8.0}
        },
        "calibration": {
            "optimizer": "least_squares",
            "least_squares": {"max_nfev": 3},
        },
    }
    settings = CAL.build_calibration_settings(config)
    scales = CAL._normalization_scales(trajectory, settings)
    record, row = CAL.calibrate_model(
        trajectory,
        "OVM",
        config,
        settings=settings,
        scales=scales,
        simulation_settings={
            "time_step_s": 0.1,
            "collision_gap_m": 0.0,
            "maximum_abs_acceleration_m_s2": 100.0,
            "stop_on_unreasonable_acceleration": True,
            "time_tolerance_s": 1.0e-6,
        },
    )
    assert record["optimizer"]["method"] == "least_squares"
    assert row["function_evaluations"] <= 3
    assert np.isfinite(row["objective_value"])


def test_run_calibration_writes_in_sample_metadata_and_metrics(tmp_path: Path) -> None:
    prepared = tmp_path / "outputs" / "04_prepared"
    prepared.mkdir(parents=True)
    _synthetic_trajectory().to_csv(prepared / "trajectory_clean.csv", index=False)
    config = {
        "project": {"random_seed": 7},
        "paths": {
            "prepared_dir": "outputs/04_prepared",
            "calibration_dir": "outputs/07_calibration",
        },
        "models": {
            "OVM": {"kappa": 0.5, "v_max": 28.0, "s_c": 25.0, "ell": 8.0},
        },
        "simulation": {
            "time_step_s": 0.1,
            "collision_gap_m": 0.0,
            "maximum_abs_acceleration_m_s2": 100.0,
            "stop_on_unreasonable_acceleration": True,
        },
        "calibration": {
            "optimizer": "differential_evolution",
            "differential_evolution": {"maxiter": 1, "popsize": 2, "polish": False},
            "parameter_bounds": {
                "OVM": {
                    "kappa": [0.4, 0.8],
                    "v_max": [25.0, 35.0],
                    "s_c": [18.0, 28.0],
                    "ell": [4.0, 12.0],
                }
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    payload = CAL.run_calibration(config_path, "OVM")
    output = tmp_path / "outputs" / "07_calibration"
    saved = json.loads((output / "calibrated_parameters.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(output / "calibration_metrics.csv")

    assert payload["metadata"]["in_sample"] is True
    assert saved["metadata"]["generalization_result"] is False
    assert "IN-SAMPLE" in saved["metadata"]["scope_warning"]
    assert set(saved["models"]["OVM"]["calibrated_parameters"]) == set(TRUE_OVM)
    assert metrics.loc[0, "calibration_scope"] == "in_sample_current_selected_case"
    assert bool(metrics.loc[0, "in_sample"])
