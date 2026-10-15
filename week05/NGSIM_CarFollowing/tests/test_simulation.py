"""Tests for external-leader simulation, integration, and failure reporting."""

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


def _load_simulation():
    name = "test_target_simulation"
    if name in sys.modules:
        return sys.modules[name]
    path = PROJECT_ROOT / "src" / "05_simulate.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SIM = _load_simulation()
OVM_PARAMETERS = {"kappa": 0.5, "v_max": 33.33, "s_c": 25.0, "ell": 8.0}
FVDM_PARAMETERS = {**OVM_PARAMETERS, "lambda": 0.5}
IDM_PARAMETERS = {"v0": 33.33, "T": 1.5, "s0": 2.0, "a": 1.2, "b": 2.0, "delta": 4.0}


def make_trajectory(sample_count: int = 8, dt: float = 0.1) -> pd.DataFrame:
    t = np.arange(sample_count, dtype=float) * dt
    leader_x = 100.0 + 15.0 * t
    follower_x = 70.0 + 10.0 * t
    leader_length = np.full(sample_count, 5.0)
    return pd.DataFrame(
        {
            "t": t,
            "leader_x": leader_x,
            "leader_v": np.full(sample_count, 15.0),
            "leader_a": np.zeros(sample_count),
            "follower_x_true": follower_x,
            "follower_v_true": np.full(sample_count, 10.0),
            "follower_a_true": np.zeros(sample_count),
            "leader_length": leader_length,
            "gap_true": leader_x - follower_x - leader_length,
        }
    )


def test_first_step_uses_euler_speed_and_trapezoidal_position() -> None:
    trajectory = make_trajectory()
    result, status = SIM.simulate_model(trajectory, "OVM", OVM_PARAMETERS)
    acceleration = SIM.MODELS.ovm_acceleration(25.0, 10.0, 15.0, OVM_PARAMETERS)
    expected_speed = max(0.0, 10.0 + acceleration * 0.1)
    expected_position = 70.0 + 0.5 * (10.0 + expected_speed) * 0.1

    assert result.loc[0, "a_sim"] == pytest.approx(acceleration)
    assert result.loc[1, "v_sim"] == pytest.approx(expected_speed)
    assert result.loc[1, "x_sim"] == pytest.approx(expected_position)
    assert result.loc[1, "gap_sim"] == pytest.approx(
        trajectory.loc[1, "leader_x"] - expected_position - trajectory.loc[1, "leader_length"]
    )
    assert status["status"] == "completed"


def test_truth_after_initial_sample_never_corrects_simulated_follower() -> None:
    original = make_trajectory()
    altered_truth = original.copy()
    altered_truth.loc[1:, "follower_x_true"] += np.linspace(1.0, 4.0, len(altered_truth) - 1)
    altered_truth.loc[1:, "follower_v_true"] = 1.0
    altered_truth.loc[1:, "follower_a_true"] = -7.0
    altered_truth["gap_true"] = (
        altered_truth["leader_x"]
        - altered_truth["follower_x_true"]
        - altered_truth["leader_length"]
    )

    first, _ = SIM.simulate_model(original, "FVDM", FVDM_PARAMETERS)
    second, _ = SIM.simulate_model(altered_truth, "FVDM", FVDM_PARAMETERS)
    pd.testing.assert_frame_equal(first, second)


def test_all_models_start_from_exactly_the_same_measured_state() -> None:
    trajectory = make_trajectory()
    configurations = {
        "OVM": OVM_PARAMETERS,
        "FVDM": FVDM_PARAMETERS,
        "IDM": IDM_PARAMETERS,
    }
    starts = []
    for model_name, parameters in configurations.items():
        result, _ = SIM.simulate_model(trajectory, model_name, parameters)
        starts.append(tuple(result.loc[0, ["x_sim", "v_sim", "gap_sim"]]))
    assert starts == [(70.0, 10.0, 25.0)] * 3


def test_collision_row_and_pre_collision_state_are_recorded_then_simulation_stops() -> None:
    t = np.arange(5, dtype=float) * 0.1
    trajectory = pd.DataFrame(
        {
            "t": t,
            "leader_x": np.full(5, 5.1),
            "leader_v": np.zeros(5),
            "leader_a": np.zeros(5),
            "follower_x_true": np.zeros(5),
            "follower_v_true": np.full(5, 10.0),
            "follower_a_true": np.zeros(5),
            "leader_length": np.full(5, 5.0),
            "gap_true": np.full(5, 0.1),
        }
    )
    parameters = {"kappa": 0.1, "v_max": 30.0, "s_c": 25.0, "ell": 8.0}
    result, status = SIM.simulate_model(trajectory, "OVM", parameters)

    assert status["status"] == "collision"
    assert status["completed"] is False
    assert status["failure_time_s"] == pytest.approx(0.1)
    assert status["failure_state"]["gap_m"] <= 0.0
    assert status["pre_failure_state"]["gap_m"] == pytest.approx(0.1)
    assert len(result) == 2
    assert result["valid"].tolist() == [True, False]
    assert result.loc[1, "status"] == "collision"


def test_initial_collision_is_reported_without_calling_model() -> None:
    trajectory = make_trajectory()
    # A positive measured gap can still be inside a configurable safety/collision boundary.
    result, status = SIM.simulate_model(
        trajectory,
        "IDM",
        IDM_PARAMETERS,
        collision_gap_m=25.0,
    )
    assert len(result) == 1
    assert not bool(result.loc[0, "valid"])
    assert np.isnan(result.loc[0, "a_sim"])
    assert status["failure_reason"] == "initial_net_gap_at_or_below_collision_threshold"
    assert status["valid_samples"] == 0


def test_unreasonable_acceleration_is_a_numerical_failure_when_configured_to_stop() -> None:
    trajectory = make_trajectory()
    parameters = {**FVDM_PARAMETERS, "lambda": 10.0}
    trajectory["leader_v"] = 0.0
    trajectory["follower_v_true"] = 10.0
    result, status = SIM.simulate_model(
        trajectory,
        "FVDM",
        parameters,
        maximum_abs_acceleration_m_s2=20.0,
        stop_on_unreasonable_acceleration=True,
    )
    assert len(result) == 1
    assert abs(result.loc[0, "a_sim"]) > 20.0
    assert not bool(result.loc[0, "valid"])
    assert status["status"] == "numerical_failure"
    assert status["failure_reason"] == "absolute_acceleration_exceeds_limit"
    assert len(status["warning_events"]) == 1


def test_unreasonable_acceleration_can_be_logged_without_masking_later_collision() -> None:
    t = np.arange(4, dtype=float) * 0.1
    trajectory = pd.DataFrame(
        {
            "t": t,
            "leader_x": np.full(4, 6.0),
            "leader_v": np.zeros(4),
            "leader_a": np.zeros(4),
            "follower_x_true": np.zeros(4),
            "follower_v_true": np.full(4, 20.0),
            "follower_a_true": np.zeros(4),
            "leader_length": np.full(4, 5.0),
            "gap_true": np.ones(4),
        }
    )
    parameters = {"kappa": 0.1, "lambda": 2.0, "v_max": 30.0, "s_c": 25.0, "ell": 8.0}
    result, status = SIM.simulate_model(
        trajectory,
        "FVDM",
        parameters,
        maximum_abs_acceleration_m_s2=5.0,
        stop_on_unreasonable_acceleration=False,
    )
    assert status["warning_events"]
    assert status["status"] == "collision"
    assert result.iloc[-1]["gap_sim"] <= 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns=["gap_true"]), "missing required columns"),
        (lambda frame: frame.assign(t=[0.0] * len(frame)), "strictly increasing"),
        (lambda frame: frame.assign(leader_v=np.nan), "non-finite"),
        (lambda frame: frame.assign(leader_length=-1.0), "leader_length must be positive"),
        (lambda frame: frame.assign(gap_true=999.0), "gap_true is inconsistent"),
    ],
)
def test_invalid_prepared_trajectories_are_rejected(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SIM.validate_trajectory(mutation(make_trajectory()), time_step_s=0.1)


def test_configured_time_step_must_match_input_grid() -> None:
    with pytest.raises(ValueError, match="does not match configured time_step_s"):
        SIM.validate_trajectory(make_trajectory(dt=0.2), time_step_s=0.1)


def test_initial_condition_cross_check_prefers_prepared_values() -> None:
    trajectory = make_trajectory()
    conditions = {
        "leader": {
            "raw_longitudinal_position_m": -999.0,
            "prepared_longitudinal_position_m": 100.0,
            "prepared_speed_m_s": 15.0,
            "vehicle_length_m": 5.0,
        },
        "follower": {
            "raw_longitudinal_position_m": -999.0,
            "prepared_longitudinal_position_m": 70.0,
            "prepared_speed_m_s": 10.0,
        },
        "relative": {"initial_gap_m": 25.0},
    }
    SIM.validate_initial_conditions(conditions, trajectory)

    conditions["follower"]["prepared_speed_m_s"] = 9.0
    with pytest.raises(ValueError, match="follower initial speed"):
        SIM.validate_initial_conditions(conditions, trajectory)


def test_run_simulations_writes_all_artifacts_and_auditable_metadata(tmp_path: Path) -> None:
    prepared = tmp_path / "outputs" / "04_prepared"
    prepared.mkdir(parents=True)
    trajectory = make_trajectory()
    trajectory.to_csv(prepared / "trajectory_clean.csv", index=False)
    conditions = {
        "leader": {
            "prepared_longitudinal_position_m": 100.0,
            "prepared_speed_m_s": 15.0,
            "vehicle_length_m": 5.0,
        },
        "follower": {
            "prepared_longitudinal_position_m": 70.0,
            "prepared_speed_m_s": 10.0,
        },
        "relative": {"initial_gap_m": 25.0},
    }
    (prepared / "initial_conditions.json").write_text(
        json.dumps(conditions), encoding="utf-8"
    )
    config = {
        "paths": {
            "prepared_dir": "outputs/04_prepared",
            "simulation_dir": "outputs/05_simulation",
        },
        "data": {"time_grid_tolerance_s": 1.0e-6},
        "models": {
            "OVM": OVM_PARAMETERS,
            "FVDM": FVDM_PARAMETERS,
            "IDM": IDM_PARAMETERS,
        },
        "simulation": {
            "time_step_s": 0.1,
            "collision_gap_m": 0.0,
            "maximum_abs_acceleration_m_s2": 100.0,
            "stop_on_unreasonable_acceleration": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    payload = SIM.run_simulations(config_path)
    output = tmp_path / "outputs" / "05_simulation"
    for model_name in ("OVM", "FVDM", "IDM"):
        csv_path = output / f"simulation_{model_name}.csv"
        assert csv_path.is_file()
        table = pd.read_csv(csv_path)
        assert ["t", "x_sim", "v_sim", "a_sim", "gap_sim"] == table.columns[:5].tolist()
        assert tuple(table.loc[0, ["x_sim", "v_sim", "gap_sim"]]) == (70.0, 10.0, 25.0)
        assert payload["models"][model_name]["parameters"]

    status_path = output / "simulation_status.json"
    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert saved["metadata"]["time_step_s"] == 0.1
    assert saved["metadata"]["follower_truth_usage"].startswith("initial")
    assert saved["metadata"]["source_trajectory_sha256"]
    assert set(saved["models"]) == {"OVM", "FVDM", "IDM"}

