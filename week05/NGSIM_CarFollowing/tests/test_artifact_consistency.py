"""Independent verification of the persisted pipeline artifacts.

These tests deliberately do not import the project's model or simulation code.
They re-implement the documented OVM/FVDM/IDM equations and the documented
integrator from scratch, then check that the CSV/JSON artifacts in ``outputs/``
actually satisfy those equations, the net-gap identity, and the reported error
metrics.  When the upstream stages have not been run yet the tests skip.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PREPARED_DIR = PROJECT_ROOT / "outputs" / "04_prepared"
SIMULATION_DIR = PROJECT_ROOT / "outputs" / "05_simulation"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "06_evaluation"

MODEL_NAMES = ("OVM", "FVDM", "IDM")


# --------------------------------------------------------------------------
# Independent reference implementations of the documented equations.
# --------------------------------------------------------------------------


def reference_optimal_velocity(gap: float, params: dict[str, float]) -> float:
    """V(s) = v_max * (tanh((s - s_c)/ell) + tanh(s_c/ell)) / (1 + tanh(s_c/ell))."""

    offset = math.tanh(params["s_c"] / params["ell"])
    value = params["v_max"] * (math.tanh((gap - params["s_c"]) / params["ell"]) + offset) / (
        1.0 + offset
    )
    return min(max(value, 0.0), params["v_max"])


def reference_acceleration(
    model: str, gap: float, follower_speed: float, leader_speed: float, params: dict[str, float]
) -> float:
    if model in {"OVM", "FVDM"}:
        acceleration = params["kappa"] * (
            reference_optimal_velocity(gap, params) - follower_speed
        )
        if model == "FVDM":
            acceleration += params["lambda"] * (leader_speed - follower_speed)
        return acceleration

    epsilon = params.get("minimum_gap_epsilon_m", 0.001)
    closing_speed = follower_speed - leader_speed
    dynamic_term = follower_speed * params["T"] + follower_speed * closing_speed / (
        2.0 * math.sqrt(params["a"] * params["b"])
    )
    desired_gap = params["s0"] + max(0.0, dynamic_term)
    effective_gap = max(gap, epsilon)
    acceleration = params["a"] * (
        1.0
        - (follower_speed / params["v0"]) ** params["delta"]
        - (desired_gap / effective_gap) ** 2
    )
    acceleration = max(acceleration, -params.get("maximum_braking_m_s2", 50.0))
    if gap <= epsilon:
        acceleration = -params.get("maximum_braking_m_s2", 50.0)
    return acceleration


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_trajectory():
    path = PREPARED_DIR / "trajectory_clean.csv"
    if not path.is_file():
        pytest.skip("run stages 03-06 before the artifact consistency checks")
    return pd.read_csv(path)


def _load_simulation(model: str):
    path = SIMULATION_DIR / f"simulation_{model}.csv"
    if not path.is_file():
        pytest.skip(f"{path.name} is missing; run stage 05 first")
    return pd.read_csv(path)


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_simulated_state_follows_the_documented_model_and_integrator(model: str) -> None:
    """Replay the simulation independently and compare against the saved CSV."""

    trajectory = _load_trajectory()
    simulation = _load_simulation(model)
    config = _load_config()
    params = dict(config["models"][model])
    time_step = float(config["simulation"]["time_step_s"])

    leader_x = trajectory["leader_x"].to_numpy(dtype=float)
    leader_v = trajectory["leader_v"].to_numpy(dtype=float)
    leader_length = trajectory["leader_length"].to_numpy(dtype=float)
    truth_x = trajectory["follower_x_true"].to_numpy(dtype=float)

    assert len(simulation) == len(trajectory), "simulation and truth must share the time base"
    assert np.allclose(simulation["t"].to_numpy(dtype=float), trajectory["t"].to_numpy(dtype=float))

    # The recorded initial state must be the measured follower state.
    assert simulation["x_sim"].iloc[0] == pytest.approx(truth_x[0], abs=1e-6)
    assert simulation["v_sim"].iloc[0] == pytest.approx(
        float(trajectory["follower_v_true"].iloc[0]), abs=1e-6
    )

    # Net-gap identity must hold on every recorded row.
    recorded_gap = simulation["gap_sim"].to_numpy(dtype=float)
    expected_gap = leader_x - simulation["x_sim"].to_numpy(dtype=float) - leader_length
    assert np.allclose(recorded_gap, expected_gap, rtol=0.0, atol=1e-5)

    # Every recorded acceleration must equal the model equation evaluated at the
    # recorded state (never the observed true follower acceleration).
    speeds = simulation["v_sim"].to_numpy(dtype=float)
    recorded_acceleration = simulation["a_sim"].to_numpy(dtype=float)
    reference = np.array(
        [
            reference_acceleration(model, recorded_gap[k], speeds[k], leader_v[k], params)
            for k in range(len(simulation))
        ]
    )
    finite = np.isfinite(recorded_acceleration)
    assert finite.any(), "no recorded acceleration values"
    assert np.allclose(
        recorded_acceleration[finite], reference[finite], rtol=1e-7, atol=1e-7
    ), f"{model} recorded accelerations disagree with the reference equation"

    # Replay the trapezoidal integrator from the shared initial state.
    positions = [float(simulation["x_sim"].iloc[0])]
    velocities = [float(simulation["v_sim"].iloc[0])]
    for index in range(len(simulation) - 1):
        gap_k = leader_x[index] - positions[-1] - leader_length[index]
        acceleration = reference_acceleration(model, gap_k, velocities[-1], leader_v[index], params)
        next_speed = max(0.0, velocities[-1] + acceleration * time_step)
        next_position = positions[-1] + 0.5 * (velocities[-1] + next_speed) * time_step
        positions.append(next_position)
        velocities.append(next_speed)

    assert np.allclose(
        np.asarray(positions), simulation["x_sim"].to_numpy(dtype=float), rtol=0.0, atol=2e-5
    ), f"{model} position trajectory is not reproducible from the documented integrator"
    assert np.allclose(
        np.asarray(velocities), speeds, rtol=0.0, atol=2e-6
    ), f"{model} speed trajectory is not reproducible from the documented integrator"


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_initial_acceleration_is_computed_not_copied_from_truth(model: str) -> None:
    """The models must evaluate their own equation at t=0."""

    trajectory = _load_trajectory()
    simulation = _load_simulation(model)
    config = _load_config()
    params = dict(config["models"][model])

    expected = reference_acceleration(
        model,
        float(simulation["gap_sim"].iloc[0]),
        float(simulation["v_sim"].iloc[0]),
        float(trajectory["leader_v"].iloc[0]),
        params,
    )
    assert float(simulation["a_sim"].iloc[0]) == pytest.approx(expected, rel=1e-7, abs=1e-7)

    initial_conditions = json.loads(
        (PREPARED_DIR / "initial_conditions.json").read_text(encoding="utf-8")
    )
    note = initial_conditions["model_initialization_note"]
    assert "must compute their initial accelerations from their equations" in note
    # The observed value is retained for comparison even when it differs from
    # whatever a model would predict.
    observed = float(initial_conditions["follower"]["initial_acceleration_m_s2"])
    assert observed == pytest.approx(
        float(trajectory["follower_a_true"].iloc[0]), rel=0.0, abs=1e-6
    )
    assert simulation["a_sim"].iloc[0] == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_documented_initial_state_matches_the_prepared_trajectory() -> None:
    """The initialisation table must be the state the simulation actually uses."""

    trajectory = _load_trajectory()
    initial = json.loads(
        (PREPARED_DIR / "initial_conditions.json").read_text(encoding="utf-8")
    )
    first = trajectory.iloc[0]
    leader = initial["leader"]
    follower = initial["follower"]

    # trajectory_clean.csv stores six decimals, hence the 1e-6 comparison floor.
    assert leader["initial_longitudinal_position_m"] == pytest.approx(
        float(first["leader_x"]), rel=0.0, abs=1e-6
    )
    assert follower["initial_longitudinal_position_m"] == pytest.approx(
        float(first["follower_x_true"]), rel=0.0, abs=1e-6
    )
    assert leader["initial_speed_m_s"] == pytest.approx(float(first["leader_v"]), abs=1e-6)
    assert follower["initial_speed_m_s"] == pytest.approx(
        float(first["follower_v_true"]), abs=1e-6
    )
    assert leader["initial_acceleration_m_s2"] == pytest.approx(
        float(first["leader_a"]), abs=1e-6
    )
    assert follower["initial_acceleration_m_s2"] == pytest.approx(
        float(first["follower_a_true"]), abs=1e-6
    )

    # The documented initial gap must follow from the documented positions.
    recomputed_gap = (
        leader["initial_longitudinal_position_m"]
        - follower["initial_longitudinal_position_m"]
        - float(first["leader_length"])
    )
    assert initial["relative"]["initial_gap_m"] == pytest.approx(recomputed_gap, abs=1e-9)
    assert initial["relative"]["initial_gap_m"] == pytest.approx(
        float(first["gap_true"]), abs=1e-6
    )
    assert initial["relative"]["initial_relative_speed_m_s"] == pytest.approx(
        leader["initial_speed_m_s"] - follower["initial_speed_m_s"], abs=1e-9
    )

    # The untouched source observation is still recorded alongside.
    assert "source_frame_speed_m_s" in leader
    assert "raw_speed_m_s" in follower


def test_simulation_reported_initial_state_matches_the_initialisation_table() -> None:
    """The simulated start state must equal the follower's documented initial state."""

    trajectory = _load_trajectory()
    initial = json.loads(
        (PREPARED_DIR / "initial_conditions.json").read_text(encoding="utf-8")
    )
    status_path = SIMULATION_DIR / "simulation_status.json"
    if not status_path.is_file():
        pytest.skip("run stage 05 before the simulation status checks")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    follower = initial["follower"]
    for record in status["models"].values():
        assert record["initial_state"]["x_m"] == pytest.approx(
            float(trajectory["follower_x_true"].iloc[0]), abs=1e-6
        )
        assert record["initial_state"]["v_m_s"] == pytest.approx(
            follower["initial_speed_m_s"], abs=1e-6
        )


def test_reported_metrics_match_independent_recomputation() -> None:
    """Recompute every headline metric from the raw artifacts."""

    metrics_path = EVALUATION_DIR / "evaluation_metrics.csv"
    if not metrics_path.is_file():
        pytest.skip("run stage 06 before the metric consistency checks")
    metrics = pd.read_csv(metrics_path).set_index("model")
    trajectory = _load_trajectory()
    truth_x = trajectory["follower_x_true"].to_numpy(dtype=float)
    truth_v = trajectory["follower_v_true"].to_numpy(dtype=float)
    truth_a = trajectory["follower_a_true"].to_numpy(dtype=float)

    for model in MODEL_NAMES:
        simulation = _load_simulation(model)
        valid = simulation["valid"].astype(str).str.casefold().eq("true").to_numpy()
        common = valid & np.ones(len(trajectory), dtype=bool)

        position_error = simulation["x_sim"].to_numpy(dtype=float) - truth_x
        speed_error = simulation["v_sim"].to_numpy(dtype=float) - truth_v
        acceleration_error = simulation["a_sim"].to_numpy(dtype=float) - truth_a
        finite_acceleration = np.isfinite(acceleration_error) & common

        assert metrics.loc[model, "position_rmse_m"] == pytest.approx(
            float(np.sqrt(np.mean(position_error[common] ** 2))), rel=1e-6, abs=1e-6
        )
        assert metrics.loc[model, "speed_rmse_m_s"] == pytest.approx(
            float(np.sqrt(np.mean(speed_error[common] ** 2))), rel=1e-6, abs=1e-6
        )
        assert metrics.loc[model, "acceleration_rmse_m_s2"] == pytest.approx(
            float(np.sqrt(np.mean(acceleration_error[finite_acceleration] ** 2))),
            rel=1e-5,
            abs=1e-6,
        )
        assert metrics.loc[model, "minimum_net_gap_m"] == pytest.approx(
            float(np.nanmin(simulation["gap_sim"].to_numpy(dtype=float))), rel=0.0, abs=1e-5
        )
        last = int(np.flatnonzero(common)[-1])
        assert metrics.loc[model, "final_position_error_m"] == pytest.approx(
            float(position_error[last]), rel=0.0, abs=1e-5
        )


def test_gap_definition_uses_leader_length_not_follower_length() -> None:
    """s = x_L - x_F - l_L, verified on the prepared truth trajectory.

    Positions are stored rounded to six decimals, so the recomputed gap can
    differ from the stored one by up to ~2e-6 m.
    """

    trajectory = _load_trajectory()
    leader_length = trajectory["leader_length"].to_numpy(dtype=float)
    recomputed = (
        trajectory["leader_x"].to_numpy(dtype=float)
        - trajectory["follower_x_true"].to_numpy(dtype=float)
        - leader_length
    )
    assert np.allclose(
        recomputed, trajectory["gap_true"].to_numpy(dtype=float), rtol=0.0, atol=5e-6
    )
    assert np.all(leader_length > 0.0)


def test_report_error_source_numbers_are_reproducible() -> None:
    """The report's transient-vs-response RMSE split must match the raw errors."""

    report_path = EVALUATION_DIR / "experiment_report.md"
    errors_path = EVALUATION_DIR / "error_timeseries.csv"
    metrics_path = EVALUATION_DIR / "evaluation_metrics.csv"
    if not (report_path.is_file() and errors_path.is_file() and metrics_path.is_file()):
        pytest.skip("run stage 06 before the report consistency checks")

    report = report_path.read_text(encoding="utf-8")
    assert "### 误差来源分解" in report, "the report must document its error sources"
    errors = pd.read_csv(errors_path)
    metrics = pd.read_csv(metrics_path).set_index("model")

    for model in MODEL_NAMES:
        valid_end = float(metrics.loc[model, "common_valid_end_s"])
        model_errors = errors.loc[errors["model"] == model]
        model_errors = model_errors.loc[model_errors["t"] <= valid_end + 1e-9]
        within_early = model_errors.loc[model_errors["t"] <= 5.0]
        early_rmse = float(
            np.sqrt(np.mean(within_early["position_error_m"].to_numpy(dtype=float) ** 2))
        )
        # _format_number uses "%.4g", so the report must contain that rendering.
        assert f"{early_rmse:.4g}" in report, (
            f"{model} early-window RMSE {early_rmse:.4g} is missing from the report"
        )
