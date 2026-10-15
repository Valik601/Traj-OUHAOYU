"""Focused tests for evaluation alignment, event rules, and artifact export."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "06_evaluate_and_plot.py"
SPEC = importlib.util.spec_from_file_location("evaluate_and_plot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_consecutive_event_uses_run_onset_and_breaks_on_nan() -> None:
    times = np.arange(0.0, 0.8, 0.1)
    values = np.array([0.0, -0.6, np.nan, -0.7, -0.8, -0.9, 0.0, 0.0])

    onset = evaluation.find_consecutive_event(
        times,
        values,
        threshold=-0.5,
        consecutive_frames=3,
        maximum_frame_gap_s=0.15,
    )

    assert onset == pytest.approx(0.3)


def test_missing_response_is_nan_not_an_arbitrary_lag() -> None:
    times = np.arange(0.0, 1.0, 0.1)
    values = np.zeros_like(times)

    onset = evaluation.find_consecutive_event(
        times,
        values,
        threshold=-0.5,
        consecutive_frames=3,
        start_time_s=0.2,
        end_time_s=0.8,
    )

    assert math.isnan(onset)


def test_evaluate_model_reports_common_prefix_and_collision() -> None:
    t = np.arange(0.0, 0.6, 0.1)
    truth = pd.DataFrame(
        {
            "t": t,
            "leader_x": 20.0 + 8.0 * t,
            "leader_v": 8.0,
            "leader_a": 0.0,
            "follower_x_true": 10.0 + 8.0 * t,
            "follower_v_true": 8.0,
            "follower_a_true": 0.0,
            "leader_length": 5.0,
            "gap_true": 5.0,
        }
    )
    simulation = pd.DataFrame(
        {
            "t": t[:4],
            "x_sim": truth["follower_x_true"].iloc[:4],
            "v_sim": 8.0,
            "a_sim": 0.0,
            "gap_sim": [5.0, 4.0, 2.0, -0.1],
            "valid": True,
        }
    )

    metrics, errors = evaluation.evaluate_model(
        "OVM",
        truth,
        simulation,
        {"status": "collision", "completed": False, "failure_time_s": 0.3},
        time_tolerance_s=1e-8,
        collision_gap_m=0.0,
        braking_threshold_m_s2=-0.5,
        braking_consecutive_frames=3,
        response_search_window_s=2.0,
        leader_braking_onset_s=math.nan,
        observed_response_onset_s=math.nan,
        observed_response_lag_s=math.nan,
        maximum_frame_gap_s=0.15,
    )

    assert metrics["common_valid_samples"] == 4
    assert metrics["completion_rate"] == 4 / 6
    assert metrics["minimum_net_gap_m"] == -0.1
    assert metrics["collision"] is True
    assert metrics["completed"] is False
    assert math.isnan(metrics["response_time_lag_s"])
    assert len(errors) == 4


def test_smoke_run_exports_all_six_figures_and_composite(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    simulations_dir = tmp_path / "simulations"
    evaluation_dir = tmp_path / "evaluation"
    cases_dir = tmp_path / "cases"
    for directory in (prepared, simulations_dir, cases_dir):
        directory.mkdir(parents=True)

    t = np.arange(0.0, 2.1, 0.1)
    leader_a = np.zeros_like(t)
    leader_a[5:8] = -1.0
    leader_a[12:15] = 0.8
    follower_a = np.zeros_like(t)
    follower_a[7:10] = -0.8
    leader_x = 30.0 + 10.0 * t
    follower_x = 15.0 + 9.8 * t
    truth = pd.DataFrame(
        {
            "t": t,
            "frame_id": np.arange(len(t)),
            "time_s": 100.0 + t,
            "leader_id": 11,
            "follower_id": 22,
            "lane_id": 3,
            "leader_x": leader_x,
            "leader_v": 10.0,
            "leader_a": leader_a,
            "follower_x_true": follower_x,
            "follower_v_true": 9.8,
            "follower_a_true": follower_a,
            "leader_length": 5.0,
            "gap_true": leader_x - follower_x - 5.0,
        }
    )
    truth.to_csv(prepared / "trajectory_clean.csv", index=False)
    initial = {
        "leader": {
            "vehicle_id": 11,
            "frame_id": 0,
            "time_s": 100.0,
            "local_x_m": 3.5,
            "local_y_m": 30.0,
            "speed_m_s": 10.0,
            "acceleration_m_s2": 0.0,
            "vehicle_length_m": 5.0,
            "lane_id": 3,
        },
        "follower": {
            "vehicle_id": 22,
            "frame_id": 0,
            "time_s": 100.0,
            "local_x_m": 3.5,
            "local_y_m": 15.0,
            "speed_m_s": 9.8,
            "acceleration_m_s2": 0.0,
            "vehicle_length_m": 4.7,
            "lane_id": 3,
        },
        "relative": {
            "initial_gap_m": 10.0,
            "initial_relative_speed_m_s": 0.2,
            "initial_relative_acceleration_m_s2": 0.0,
        },
        "processing": {"smoothing_method": "synthetic smoke fixture"},
    }
    (prepared / "initial_conditions.json").write_text(json.dumps(initial), encoding="utf-8")
    (cases_dir / "selected_case.json").write_text(
        json.dumps(
            {
                "leader_id": 11,
                "follower_id": 22,
                "lane_id": 3,
                "start_time_s": 100.0,
                "end_time_s": 102.0,
                "selection_reason": "synthetic smoke test",
            }
        ),
        encoding="utf-8",
    )

    model_status = {}
    for offset, model in enumerate(evaluation.MODEL_ORDER):
        x_sim = follower_x + offset * 0.05
        simulation = pd.DataFrame(
            {
                "t": t,
                "x_sim": x_sim,
                "v_sim": 9.8 + offset * 0.02,
                "a_sim": follower_a,
                "gap_sim": leader_x - x_sim - 5.0,
                "valid": True,
                "status": "completed",
            }
        )
        simulation.to_csv(simulations_dir / f"simulation_{model}.csv", index=False)
        model_status[model] = {
            "status": "completed",
            "completed": True,
            "failure_time_s": None,
            "failure_reason": None,
            "parameters": {"fixture_parameter": offset + 1},
        }
    status = {
        "metadata": {"dt": 0.1, "integration_method": "smoke-test trapezoid"},
        "models": model_status,
    }
    (simulations_dir / "simulation_status.json").write_text(json.dumps(status), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: {name: smoke}\n", encoding="utf-8")
    config = {
        "paths": {
            "prepared_dir": str(prepared),
            "simulation_dir": str(simulations_dir),
            "evaluation_dir": str(evaluation_dir),
            "cases_dir": str(cases_dir),
        },
        "data": {"sampling_interval_s": 0.1, "time_grid_tolerance_s": 1e-8},
        "simulation": {"collision_gap_m": 0.0, "time_step_s": 0.1},
        "evaluation": {
            "braking_acceleration_threshold_m_s2": -0.5,
            "braking_consecutive_frames": 3,
            "response_search_window_s": 1.0,
            "png_dpi": 100,
            "figure_width_in": 6.0,
            "figure_height_in": 4.0,
        },
    }

    outputs = evaluation.run_evaluation(config, config_path)

    assert outputs["metrics"].is_file()
    assert outputs["report"].is_file()
    for stem in (
        "trajectory_selection",
        "position_comparison",
        "speed_response",
        "acceleration_response",
        "spacing_comparison",
        "position_error",
        "model_comparison",
    ):
        assert (evaluation_dir / f"{stem}.png").is_file()
        assert (evaluation_dir / f"{stem}.pdf").is_file()
