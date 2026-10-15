#!/usr/bin/env python3
"""Evaluate car-following simulations and create reproducible figures/report.

The script deliberately evaluates each model only over its *contiguous common
valid prefix* with the measured follower trajectory.  A model that collides or
otherwise fails early therefore keeps its failure status and completion rate;
its shorter RMSE is never presented as if it covered the full experiment.

All configuration paths are resolved from the project root (the parent of
``src``), independently of the current working directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger("evaluate_and_plot")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("OVM", "FVDM", "IDM")

# Okabe-Ito-derived colours plus redundant line styles and hatches.  Colour is
# never the sole model identifier in either line plots or bars.
MODEL_STYLES: dict[str, dict[str, Any]] = {
    "OVM": {"color": "#0072B2", "linestyle": "--", "hatch": "///"},
    "FVDM": {"color": "#D55E00", "linestyle": "-.", "hatch": "xxx"},
    "IDM": {"color": "#009E73", "linestyle": ":", "hatch": "..."},
}
TRUTH_STYLES: dict[str, dict[str, Any]] = {
    "leader": {"color": "#111111", "linestyle": "-"},
    "follower": {"color": "#666666", "linestyle": (0, (5, 2))},
}

TRUTH_REQUIRED_COLUMNS = {
    "t",
    "leader_x",
    "leader_v",
    "leader_a",
    "follower_x_true",
    "follower_v_true",
    "follower_a_true",
    "leader_length",
    "gap_true",
}
SIM_REQUIRED_COLUMNS = {"t", "x_sim", "v_sim", "a_sim", "gap_sim"}


class EvaluationError(RuntimeError):
    """Raised when an evaluation input violates the documented contract."""


def _case_insensitive_get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return ``mapping[key]`` without depending on key capitalisation."""

    if key in mapping:
        return mapping[key]
    folded = key.casefold()
    for candidate, value in mapping.items():
        if str(candidate).casefold() == folded:
            return value
    return default


def _config_value(
    config: Mapping[str, Any],
    section: str,
    names: Sequence[str],
    default: Any,
) -> Any:
    section_value = _case_insensitive_get(config, section, {})
    if not isinstance(section_value, Mapping):
        return default
    for name in names:
        value = _case_insensitive_get(section_value, name, None)
        if value is not None:
            return value
    return default


def resolve_from_project(path_value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a configured path from the project root."""

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_config(config_path: str | Path, project_root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], Path]:
    """Load YAML after resolving a relative ``--config`` from the project root."""

    resolved = resolve_from_project(config_path, project_root)
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise EvaluationError(f"Configuration root must be a mapping: {resolved}")
    return payload, resolved


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise EvaluationError(f"JSON root must be an object: {path}")
    return payload


def _numeric_frame(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise EvaluationError(f"{label} is missing required columns {missing}: {path}")
    if frame.empty:
        raise EvaluationError(f"{label} contains no rows: {path}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["t"].isna().any():
        raise EvaluationError(f"{label} contains non-numeric or missing t values: {path}")
    frame = frame.sort_values("t", kind="mergesort").reset_index(drop=True)
    if frame["t"].duplicated().any():
        examples = frame.loc[frame["t"].duplicated(keep=False), "t"].head(5).tolist()
        raise EvaluationError(f"{label} has duplicate time values, e.g. {examples}: {path}")
    if np.any(np.diff(frame["t"].to_numpy(dtype=float)) <= 0):
        raise EvaluationError(f"{label} time must increase strictly: {path}")
    return frame


def load_truth(path: Path) -> pd.DataFrame:
    return _numeric_frame(path, TRUTH_REQUIRED_COLUMNS, "prepared trajectory")


def load_simulation(path: Path, model: str) -> pd.DataFrame:
    frame = _numeric_frame(path, SIM_REQUIRED_COLUMNS, f"{model} simulation")
    if "valid" in frame.columns:
        frame["valid"] = _coerce_boolean_series(frame["valid"])
    else:
        frame["valid"] = True
    return frame


def _coerce_boolean_series(series: pd.Series) -> pd.Series:
    """Interpret common CSV boolean spellings; unknown/missing values are false."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    true_values = {"1", "true", "t", "yes", "y", "valid"}
    return series.fillna("").astype(str).str.strip().str.casefold().isin(true_values)


def find_simulation_file(simulation_dir: Path, model: str) -> Path:
    exact = simulation_dir / f"simulation_{model}.csv"
    if exact.is_file():
        return exact
    expected = exact.name.casefold()
    for candidate in simulation_dir.glob("simulation_*.csv"):
        if candidate.name.casefold() == expected:
            return candidate
    raise FileNotFoundError(f"Missing simulation for {model}: expected {exact}")


def extract_model_status(status_payload: Mapping[str, Any], model: str) -> dict[str, Any]:
    container = _case_insensitive_get(status_payload, "models", status_payload)
    if not isinstance(container, Mapping):
        return {}
    value = _case_insensitive_get(container, model, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _as_optional_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def find_consecutive_event(
    times: Sequence[float] | np.ndarray | pd.Series,
    values: Sequence[float] | np.ndarray | pd.Series,
    threshold: float,
    consecutive_frames: int,
    *,
    direction: str = "below",
    start_time_s: float | None = None,
    end_time_s: float | None = None,
    maximum_frame_gap_s: float | None = None,
) -> float:
    """Return the first run onset satisfying an explicit threshold rule.

    NaNs, samples outside the search interval, and time discontinuities break a
    run.  ``math.nan`` means that the event is not evaluable/detectable.
    """

    if consecutive_frames < 1:
        raise ValueError("consecutive_frames must be >= 1")
    if direction not in {"below", "above"}:
        raise ValueError("direction must be 'below' or 'above'")

    time_array = np.asarray(times, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if time_array.shape != value_array.shape:
        raise ValueError("times and values must have identical shapes")
    if time_array.ndim != 1:
        raise ValueError("times and values must be one-dimensional")

    run_start: int | None = None
    run_length = 0
    previous_index: int | None = None

    for index, (time_value, signal_value) in enumerate(zip(time_array, value_array)):
        in_window = (
            math.isfinite(time_value)
            and (start_time_s is None or time_value >= start_time_s - 1e-12)
            and (end_time_s is None or time_value <= end_time_s + 1e-12)
        )
        qualifies = (
            in_window
            and math.isfinite(signal_value)
            and (
                signal_value <= threshold
                if direction == "below"
                else signal_value >= threshold
            )
        )
        contiguous = True
        if qualifies and previous_index is not None and maximum_frame_gap_s is not None:
            contiguous = time_value - time_array[previous_index] <= maximum_frame_gap_s + 1e-12

        if qualifies and (run_length == 0 or contiguous):
            if run_length == 0:
                run_start = index
            run_length += 1
        elif qualifies:
            run_start = index
            run_length = 1
        else:
            run_start = None
            run_length = 0

        previous_index = index if qualifies else None
        if run_length >= consecutive_frames and run_start is not None:
            return float(time_array[run_start])

    return math.nan


def _infer_sampling_interval(times: pd.Series, fallback: float = 0.1) -> float:
    differences = np.diff(times.to_numpy(dtype=float))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    return float(np.median(differences)) if differences.size else float(fallback)


def align_common_valid_prefix(
    truth: pd.DataFrame,
    simulation: pd.DataFrame,
    tolerance_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align on time and retain the contiguous valid prefix from the first truth row."""

    sim = simulation.rename(columns={"t": "t_sim"}).copy()
    aligned = pd.merge_asof(
        truth.sort_values("t"),
        sim.sort_values("t_sim"),
        left_on="t",
        right_on="t_sim",
        direction="nearest",
        tolerance=tolerance_s,
    )
    finite_columns = [
        "t_sim",
        "leader_x",
        "leader_v",
        "leader_a",
        "follower_x_true",
        "follower_v_true",
        "follower_a_true",
        "x_sim",
        "v_sim",
        "a_sim",
        "gap_sim",
    ]
    finite_mask = np.ones(len(aligned), dtype=bool)
    for column in finite_columns:
        finite_mask &= np.isfinite(pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=float))
    finite_mask &= _coerce_boolean_series(aligned["valid"]).to_numpy(dtype=bool)

    invalid_indices = np.flatnonzero(~finite_mask)
    prefix_length = int(invalid_indices[0]) if invalid_indices.size else len(aligned)
    common = aligned.iloc[:prefix_length].copy().reset_index(drop=True)
    return common, aligned


def rmse(simulated: Iterable[float], observed: Iterable[float]) -> float:
    simulated_array = np.asarray(list(simulated), dtype=float)
    observed_array = np.asarray(list(observed), dtype=float)
    if simulated_array.shape != observed_array.shape:
        raise ValueError("RMSE arrays must have identical shapes")
    mask = np.isfinite(simulated_array) & np.isfinite(observed_array)
    if not np.any(mask):
        return math.nan
    return float(np.sqrt(np.mean(np.square(simulated_array[mask] - observed_array[mask]))))


def _status_text(model_status: Mapping[str, Any]) -> str:
    value = model_status.get("status", model_status.get("state", ""))
    return str(value).strip().casefold()


def _status_bool(model_status: Mapping[str, Any], name: str) -> bool | None:
    if name not in model_status:
        return None
    value = model_status[name]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        if value.strip().casefold() in {"true", "yes", "1", "completed"}:
            return True
        if value.strip().casefold() in {"false", "no", "0", "failed"}:
            return False
    return None


def evaluate_model(
    model: str,
    truth: pd.DataFrame,
    simulation: pd.DataFrame,
    model_status: Mapping[str, Any],
    *,
    time_tolerance_s: float,
    collision_gap_m: float,
    braking_threshold_m_s2: float,
    braking_consecutive_frames: int,
    response_search_window_s: float,
    leader_braking_onset_s: float,
    observed_response_onset_s: float,
    observed_response_lag_s: float,
    maximum_frame_gap_s: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    common, _ = align_common_valid_prefix(truth, simulation, time_tolerance_s)
    reference_samples = int(len(truth))
    common_samples = int(len(common))
    completion_rate = common_samples / reference_samples if reference_samples else math.nan

    if common_samples:
        position_rmse = rmse(common["x_sim"], common["follower_x_true"])
        speed_rmse = rmse(common["v_sim"], common["follower_v_true"])
        acceleration_rmse = rmse(common["a_sim"], common["follower_a_true"])
        common_start = float(common["t"].iloc[0])
        common_end = float(common["t"].iloc[-1])
        common_duration = common_end - common_start
        final_position_error = float(common["x_sim"].iloc[-1] - common["follower_x_true"].iloc[-1])
    else:
        position_rmse = speed_rmse = acceleration_rmse = math.nan
        common_start = common_end = common_duration = final_position_error = math.nan

    finite_gap = simulation.loc[np.isfinite(simulation["gap_sim"]), ["t", "gap_sim"]]
    if finite_gap.empty:
        minimum_gap = minimum_gap_time = math.nan
    else:
        minimum_index = finite_gap["gap_sim"].idxmin()
        minimum_gap = float(simulation.loc[minimum_index, "gap_sim"])
        minimum_gap_time = float(simulation.loc[minimum_index, "t"])

    status_value = _status_text(model_status)
    failure_reason = str(model_status.get("failure_reason") or "").strip()
    collision = (
        "collision" in status_value
        or "collision" in failure_reason.casefold()
        or (math.isfinite(minimum_gap) and minimum_gap <= collision_gap_m)
    )
    explicit_completed = _status_bool(model_status, "completed")
    inferred_completed = common_samples == reference_samples and not collision
    completed = inferred_completed if explicit_completed is None else explicit_completed and inferred_completed

    if collision:
        evaluation_status = status_value if "collision" in status_value else "collision_inferred"
    elif status_value in {"completed", "complete", "success"} and not completed:
        evaluation_status = "incomplete_coverage"
    elif status_value:
        evaluation_status = status_value
    elif completed:
        evaluation_status = "completed"
    else:
        evaluation_status = "incomplete"

    failure_time = _as_optional_float(
        model_status.get("failure_time_s", model_status.get("failure_time", math.nan))
    )
    if not math.isfinite(failure_time) and collision:
        collision_rows = finite_gap.loc[finite_gap["gap_sim"] <= collision_gap_m]
        if not collision_rows.empty:
            failure_time = float(collision_rows["t"].iloc[0])
    if not completed and not failure_reason:
        if collision:
            failure_reason = f"net gap reached collision boundary ({collision_gap_m:g} m)"
        elif status_value:
            failure_reason = status_value
        else:
            failure_reason = "simulation did not cover the full valid reference interval"

    if math.isfinite(leader_braking_onset_s) and common_samples:
        simulated_response_onset = find_consecutive_event(
            common["t"],
            common["a_sim"],
            braking_threshold_m_s2,
            braking_consecutive_frames,
            direction="below",
            start_time_s=leader_braking_onset_s,
            end_time_s=leader_braking_onset_s + response_search_window_s,
            maximum_frame_gap_s=maximum_frame_gap_s,
        )
    else:
        simulated_response_onset = math.nan

    if not math.isfinite(leader_braking_onset_s):
        response_lag = response_lag_error = math.nan
        response_evaluable = False
        response_reason = "leader braking event was not detected"
    elif not math.isfinite(simulated_response_onset):
        response_lag = response_lag_error = math.nan
        response_evaluable = False
        response_reason = (
            f"no {model} braking onset met the threshold within "
            f"{response_search_window_s:g} s after the leader event"
        )
    else:
        response_lag = float(simulated_response_onset - leader_braking_onset_s)
        response_lag_error = (
            float(response_lag - observed_response_lag_s)
            if math.isfinite(observed_response_lag_s)
            else math.nan
        )
        response_evaluable = True
        response_reason = ""

    errors = common[["t"]].copy()
    errors["model"] = model
    errors["position_error_m"] = common["x_sim"] - common["follower_x_true"]
    errors["speed_error_m_s"] = common["v_sim"] - common["follower_v_true"]
    errors["acceleration_error_m_s2"] = common["a_sim"] - common["follower_a_true"]

    metrics = {
        "model": model,
        "position_rmse_m": position_rmse,
        "speed_rmse_m_s": speed_rmse,
        "acceleration_rmse_m_s2": acceleration_rmse,
        "minimum_net_gap_m": minimum_gap,
        "minimum_gap_time_s": minimum_gap_time,
        "final_position_error_m": final_position_error,
        "final_position_absolute_error_m": abs(final_position_error)
        if math.isfinite(final_position_error)
        else math.nan,
        "common_valid_start_s": common_start,
        "common_valid_end_s": common_end,
        "common_valid_duration_s": common_duration,
        "common_valid_samples": common_samples,
        "reference_samples": reference_samples,
        "completion_rate": completion_rate,
        "completion_percent": completion_rate * 100 if math.isfinite(completion_rate) else math.nan,
        "status": evaluation_status,
        "completed": bool(completed),
        "collision": bool(collision),
        "failure_time_s": failure_time,
        "failure_reason": failure_reason,
        "leader_braking_onset_s": leader_braking_onset_s,
        "observed_response_onset_s": observed_response_onset_s,
        "observed_response_lag_s": observed_response_lag_s,
        "simulated_response_onset_s": simulated_response_onset,
        "response_time_lag_s": response_lag,
        "response_lag_error_s": response_lag_error,
        "response_lag_evaluable": bool(response_evaluable),
        "response_lag_reason": response_reason,
    }
    return metrics, errors


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _export_figure(fig: plt.Figure, output_base: Path, dpi: int, title: str) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, facecolor="white", metadata={"Title": title})
    fig.savefig(pdf_path, facecolor="white", metadata={"Title": title})
    plt.close(fig)
    return [png_path, pdf_path]


def _plot_model_line(ax: plt.Axes, simulation: pd.DataFrame, model: str, column: str) -> None:
    style = MODEL_STYLES[model]
    valid = _coerce_boolean_series(simulation["valid"]).to_numpy(dtype=bool)
    values = pd.to_numeric(simulation[column], errors="coerce").to_numpy(dtype=float)
    plotted_values = values.copy()
    plotted_values[~valid] = np.nan
    ax.plot(
        simulation["t"],
        plotted_values,
        label=model,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.8,
    )
    failure_rows = (~valid) & np.isfinite(values)
    if np.any(failure_rows):
        ax.scatter(
            simulation.loc[failure_rows, "t"],
            values[failure_rows],
            marker="X",
            s=38,
            color=style["color"],
            edgecolor="#222222",
            linewidth=0.45,
            zorder=5,
        )


def _identity(initial: Mapping[str, Any], truth: pd.DataFrame) -> dict[str, Any]:
    leader = initial.get("leader", {}) if isinstance(initial.get("leader", {}), Mapping) else {}
    follower = initial.get("follower", {}) if isinstance(initial.get("follower", {}), Mapping) else {}

    def first_truth(name: str, default: Any = "unknown") -> Any:
        if name not in truth.columns:
            return default
        values = truth[name].dropna()
        return values.iloc[0] if not values.empty else default

    return {
        "leader_id": leader.get("vehicle_id", first_truth("leader_id")),
        "follower_id": follower.get("vehicle_id", first_truth("follower_id")),
        "lane_id": leader.get("lane_id", follower.get("lane_id", first_truth("lane_id"))),
    }


def _add_event_lines(
    ax: plt.Axes,
    braking_time_s: float,
    reacceleration_time_s: float,
    observed_response_time_s: float = math.nan,
    *,
    labels: bool,
) -> None:
    if math.isfinite(braking_time_s):
        ax.axvline(
            braking_time_s,
            color="#CC3311",
            linestyle=(0, (3, 2)),
            linewidth=1.1,
            alpha=0.9,
            label="Leader braking onset" if labels else None,
        )
    if math.isfinite(reacceleration_time_s):
        ax.axvline(
            reacceleration_time_s,
            color="#AA4499",
            linestyle=(0, (1, 2)),
            linewidth=1.1,
            alpha=0.9,
            label="Leader re-acceleration onset" if labels else None,
        )
    if math.isfinite(observed_response_time_s):
        ax.axvline(
            observed_response_time_s,
            color="#4477AA",
            linestyle=(0, (5, 1, 1, 1)),
            linewidth=1.0,
            alpha=0.8,
            label="Measured follower braking onset" if labels else None,
        )


def plot_trajectory_selection(
    truth: pd.DataFrame,
    identity: Mapping[str, Any],
    output_dir: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.plot(
        truth["t"],
        truth["leader_x"],
        label=f"Leader {identity['leader_id']}",
        **TRUTH_STYLES["leader"],
    )
    ax.plot(
        truth["t"],
        truth["follower_x_true"],
        label=f"Follower {identity['follower_id']}",
        **TRUTH_STYLES["follower"],
    )
    start_time, end_time = float(truth["t"].iloc[0]), float(truth["t"].iloc[-1])
    ax.axvline(start_time, color="#999999", linestyle=":", label="Selected start/end")
    ax.axvline(end_time, color="#999999", linestyle=":")
    for x_column, marker in (("leader_x", "o"), ("follower_x_true", "s")):
        ax.scatter(
            [start_time, end_time],
            [truth[x_column].iloc[0], truth[x_column].iloc[-1]],
            marker=marker,
            s=28,
            facecolor="white",
            edgecolor="#222222",
            zorder=4,
        )
    ax.set(
        xlabel="Elapsed time (s)",
        ylabel="Longitudinal front position (m)",
        title=(
            "Selected car-following trajectory | "
            f"lane {identity['lane_id']}, leader {identity['leader_id']}, "
            f"follower {identity['follower_id']}"
        ),
    )
    ax.legend(ncol=2)
    return _export_figure(fig, output_dir / "trajectory_selection", dpi, "Vehicle Trajectory Selection")


def plot_position_comparison(
    truth: pd.DataFrame,
    simulations: Mapping[str, pd.DataFrame],
    output_dir: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.plot(truth["t"], truth["leader_x"], label="Measured leader", **TRUTH_STYLES["leader"])
    ax.plot(
        truth["t"],
        truth["follower_x_true"],
        label="Measured follower",
        **TRUTH_STYLES["follower"],
    )
    for model in MODEL_ORDER:
        _plot_model_line(ax, simulations[model], model, "x_sim")
    ax.set(
        xlabel="Elapsed time (s)",
        ylabel="Longitudinal front position (m)",
        title="Measured and simulated longitudinal positions",
    )
    ax.legend(ncol=2)
    return _export_figure(fig, output_dir / "position_comparison", dpi, "Longitudinal Position Comparison")


def plot_speed_response(
    truth: pd.DataFrame,
    simulations: Mapping[str, pd.DataFrame],
    braking_time_s: float,
    reacceleration_time_s: float,
    observed_response_time_s: float,
    output_dir: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.plot(truth["t"], truth["leader_v"], label="Measured leader", **TRUTH_STYLES["leader"])
    ax.plot(
        truth["t"],
        truth["follower_v_true"],
        label="Measured follower",
        **TRUTH_STYLES["follower"],
    )
    for model in MODEL_ORDER:
        _plot_model_line(ax, simulations[model], model, "v_sim")
    _add_event_lines(
        ax,
        braking_time_s,
        reacceleration_time_s,
        observed_response_time_s,
        labels=True,
    )
    ax.set(xlabel="Elapsed time (s)", ylabel="Speed (m/s)", title="Velocity response")
    ax.legend(ncol=2)
    return _export_figure(fig, output_dir / "speed_response", dpi, "Velocity Response")


def plot_acceleration_response(
    truth: pd.DataFrame,
    simulations: Mapping[str, pd.DataFrame],
    braking_time_s: float,
    reacceleration_time_s: float,
    observed_response_time_s: float,
    output_dir: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.plot(truth["t"], truth["leader_a"], label="Measured leader", **TRUTH_STYLES["leader"])
    ax.plot(
        truth["t"],
        truth["follower_a_true"],
        label="Measured follower",
        **TRUTH_STYLES["follower"],
    )
    for model in MODEL_ORDER:
        _plot_model_line(ax, simulations[model], model, "a_sim")
    _add_event_lines(
        ax,
        braking_time_s,
        reacceleration_time_s,
        observed_response_time_s,
        labels=True,
    )
    ax.set(
        xlabel="Elapsed time (s)",
        ylabel="Acceleration (m/s²)",
        title="Acceleration and braking response",
    )
    ax.legend(ncol=2)
    return _export_figure(fig, output_dir / "acceleration_response", dpi, "Acceleration Response")


def plot_gap_comparison(
    truth: pd.DataFrame,
    simulations: Mapping[str, pd.DataFrame],
    collision_gap_m: float,
    output_dir: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.plot(
        truth["t"],
        truth["gap_true"],
        label="Measured net gap",
        **TRUTH_STYLES["follower"],
    )
    finite_truth = truth.loc[np.isfinite(truth["gap_true"])]
    if not finite_truth.empty:
        truth_minimum_index = finite_truth["gap_true"].idxmin()
        ax.scatter(
            truth.loc[truth_minimum_index, "t"],
            truth.loc[truth_minimum_index, "gap_true"],
            marker="v",
            s=36,
            facecolor="white",
            edgecolor=TRUTH_STYLES["follower"]["color"],
            linewidth=1.0,
            zorder=4,
        )
    for model in MODEL_ORDER:
        simulation = simulations[model]
        _plot_model_line(ax, simulation, model, "gap_sim")
        finite = simulation.loc[np.isfinite(simulation["gap_sim"])]
        if not finite.empty:
            index = finite["gap_sim"].idxmin()
            style = MODEL_STYLES[model]
            ax.scatter(
                simulation.loc[index, "t"],
                simulation.loc[index, "gap_sim"],
                color=style["color"],
                marker="v",
                s=36,
                edgecolor="#222222",
                linewidth=0.4,
                zorder=4,
            )
    ax.axhline(
        collision_gap_m,
        color="#CC3311",
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        label=f"Collision boundary ({collision_gap_m:g} m)",
    )
    ax.set(xlabel="Elapsed time (s)", ylabel="Net gap (m)", title="Net-gap evolution")
    ax.legend(ncol=2)
    return _export_figure(fig, output_dir / "spacing_comparison", dpi, "Net Gap Evolution")


def _metric_bars(ax: plt.Axes, metrics: pd.DataFrame, column: str, ylabel: str, title: str) -> None:
    values = metrics.set_index("model").reindex(MODEL_ORDER)[column].to_numpy(dtype=float)
    safe_values = np.where(np.isfinite(values), values, 0.0)
    bars = ax.bar(
        np.arange(len(MODEL_ORDER)),
        safe_values,
        color=[MODEL_STYLES[model]["color"] for model in MODEL_ORDER],
        edgecolor="#222222",
        linewidth=0.7,
    )
    ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set(ylabel=ylabel, title=title)
    ax.set_ylim(bottom=0)
    for model, bar, value in zip(MODEL_ORDER, bars, values):
        bar.set_hatch(MODEL_STYLES[model]["hatch"])
        row = metrics.loc[metrics["model"] == model].iloc[0]
        if not bool(row["completed"]):
            bar.set_alpha(0.55)
        label = "N/A" if not math.isfinite(value) else f"{value:.3g}"
        coverage = _as_optional_float(row["completion_percent"])
        status_suffix = "" if bool(row["completed"]) else f"\n{coverage:.1f}% / {row['status']}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label + status_suffix,
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_error_comparison(
    errors: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> list[Path]:
    error_width = max(figure_size[0], 12.0)
    error_height = max(figure_size[1], 8.5)
    fig, axes = plt.subplots(2, 3, figsize=(error_width, error_height), constrained_layout=True)
    time_specs = (
        ("position_error_m", "Position error (m)", "Position error vs time"),
        ("speed_error_m_s", "Speed error (m/s)", "Speed error vs time"),
        ("acceleration_error_m_s2", "Acceleration error (m/s²)", "Acceleration error vs time"),
    )
    for ax, (column, ylabel, title) in zip(axes[0], time_specs):
        ax.axhline(0, color="#999999", linewidth=0.8)
        for model in MODEL_ORDER:
            subset = errors.loc[errors["model"] == model]
            style = MODEL_STYLES[model]
            ax.plot(
                subset["t"],
                subset[column],
                color=style["color"],
                linestyle=style["linestyle"],
                label=model,
            )
            failure_time = _as_optional_float(
                metrics.loc[metrics["model"] == model, "failure_time_s"].iloc[0]
            )
            if math.isfinite(failure_time):
                ax.axvline(failure_time, color=style["color"], linestyle=":", alpha=0.45)
        ax.set(xlabel="Elapsed time (s)", ylabel=ylabel, title=title)
    axes[0, 0].legend()

    bar_specs = (
        ("position_rmse_m", "RMSE (m)", "Position RMSE"),
        ("speed_rmse_m_s", "RMSE (m/s)", "Speed RMSE"),
        ("acceleration_rmse_m_s2", "RMSE (m/s²)", "Acceleration RMSE"),
    )
    for ax, (column, ylabel, title) in zip(axes[1], bar_specs):
        _metric_bars(ax, metrics, column, ylabel, title)
    fig.suptitle("Error comparison (faded bars denote incomplete/failed simulations)", fontsize=13)
    return _export_figure(fig, output_dir / "position_error", dpi, "Error Comparison")


def plot_model_comparison(
    truth: pd.DataFrame,
    simulations: Mapping[str, pd.DataFrame],
    errors: pd.DataFrame,
    metrics: pd.DataFrame,
    braking_time_s: float,
    reacceleration_time_s: float,
    observed_response_time_s: float,
    collision_gap_m: float,
    output_dir: Path,
    figure_size: tuple[float, float],
    dpi: int,
) -> list[Path]:
    width = max(figure_size[0], 12.0)
    height = max(figure_size[1], 9.0)
    fig, axes = plt.subplots(3, 2, figsize=(width, height), constrained_layout=True)

    axes[0, 0].plot(truth["t"], truth["leader_x"], label="Measured leader", **TRUTH_STYLES["leader"])
    axes[0, 0].plot(
        truth["t"], truth["follower_x_true"], label="Measured follower", **TRUTH_STYLES["follower"]
    )
    for model in MODEL_ORDER:
        _plot_model_line(axes[0, 0], simulations[model], model, "x_sim")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Position (m)", title="Longitudinal position")

    axes[0, 1].plot(truth["t"], truth["leader_v"], label="Measured leader", **TRUTH_STYLES["leader"])
    axes[0, 1].plot(
        truth["t"], truth["follower_v_true"], label="Measured follower", **TRUTH_STYLES["follower"]
    )
    for model in MODEL_ORDER:
        _plot_model_line(axes[0, 1], simulations[model], model, "v_sim")
    _add_event_lines(
        axes[0, 1],
        braking_time_s,
        reacceleration_time_s,
        observed_response_time_s,
        labels=False,
    )
    axes[0, 1].set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Velocity response")

    axes[1, 0].axhline(0, color="#BBBBBB", linewidth=0.8)
    axes[1, 0].plot(truth["t"], truth["leader_a"], label="Measured leader", **TRUTH_STYLES["leader"])
    axes[1, 0].plot(
        truth["t"], truth["follower_a_true"], label="Measured follower", **TRUTH_STYLES["follower"]
    )
    for model in MODEL_ORDER:
        _plot_model_line(axes[1, 0], simulations[model], model, "a_sim")
    _add_event_lines(
        axes[1, 0],
        braking_time_s,
        reacceleration_time_s,
        observed_response_time_s,
        labels=False,
    )
    axes[1, 0].set(xlabel="Time (s)", ylabel="Acceleration (m/s²)", title="Acceleration")

    axes[1, 1].plot(
        truth["t"], truth["gap_true"], label="Measured gap", **TRUTH_STYLES["follower"]
    )
    for model in MODEL_ORDER:
        _plot_model_line(axes[1, 1], simulations[model], model, "gap_sim")
    axes[1, 1].axhline(collision_gap_m, color="#CC3311", linestyle=(0, (3, 2)), linewidth=1.1)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Net gap (m)", title="Net gap")

    axes[2, 0].axhline(0, color="#999999", linewidth=0.8)
    for model in MODEL_ORDER:
        subset = errors.loc[errors["model"] == model]
        style = MODEL_STYLES[model]
        axes[2, 0].plot(
            subset["t"],
            subset["position_error_m"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=model,
        )
    axes[2, 0].set(xlabel="Time (s)", ylabel="Position error (m)", title="Follower position error")

    axes[2, 1].axis("off")
    table_rows: list[list[str]] = []
    for model in MODEL_ORDER:
        row = metrics.loc[metrics["model"] == model].iloc[0]
        table_rows.append(
            [
                model,
                _format_number(row["position_rmse_m"], 3),
                _format_number(row["speed_rmse_m_s"], 3),
                _format_number(row["minimum_net_gap_m"], 3),
                f"{row['completion_percent']:.1f}%",
                str(row["status"]),
            ]
        )
    table = axes[2, 1].table(
        cellText=table_rows,
        colLabels=["Model", "x RMSE\n(m)", "v RMSE\n(m/s)", "min gap\n(m)", "coverage", "status"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    axes[2, 1].set_title("Evaluation summary")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5)
    fig.suptitle("NGSIM I-80 car-following model comparison", fontsize=14)
    return _export_figure(fig, output_dir / "model_comparison", dpi, "Model Comparison")


def _format_number(value: Any, digits: int = 4, suffix: str = "") -> str:
    number = _as_optional_float(value)
    return f"{number:.{digits}g}{suffix}" if math.isfinite(number) else "NaN（不可评价）"


def _display_path(path: Path) -> str:
    """Prefer a project-relative path while remaining usable in isolated tests."""

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(output)


def _initial_state_rows(initial: Mapping[str, Any], truth: pd.DataFrame) -> list[list[str]]:
    leader = initial.get("leader", {}) if isinstance(initial.get("leader", {}), Mapping) else {}
    follower = initial.get("follower", {}) if isinstance(initial.get("follower", {}), Mapping) else {}

    def item(record: Mapping[str, Any], key: str, fallback: Any = math.nan) -> Any:
        return record.get(key, fallback)

    def used(
        record: Mapping[str, Any],
        key: str,
        fallback_key: str,
        truth_key: str,
    ) -> Any:
        """Prefer the state that actually initialises the simulation."""

        value = record.get(key, None)
        if value is None:
            value = record.get(fallback_key, None)
        return fallback[truth_key] if value is None else value

    fallback = truth.iloc[0]
    # Seven significant digits keep the table numerically checkable:
    # x_L(0) - x_F(0) - l_L must reproduce the reported initial net gap.
    ndigits = 7
    return [
        ["Vehicle ID", item(leader, "vehicle_id", fallback.get("leader_id", "unknown")), item(follower, "vehicle_id", fallback.get("follower_id", "unknown"))],
        ["Frame ID", item(leader, "frame_id", fallback.get("frame_id", "unknown")), item(follower, "frame_id", fallback.get("frame_id", "unknown"))],
        ["Initial time (s)", _format_number(item(leader, "time_s", fallback.get("time_s", fallback["t"])), 8), _format_number(item(follower, "time_s", fallback.get("time_s", fallback["t"])), 8)],
        ["Initial lateral position (m)", _format_number(item(leader, "local_x_m"), ndigits), _format_number(item(follower, "local_x_m"), ndigits)],
        ["Initial longitudinal front position (m)", _format_number(used(leader, "initial_longitudinal_position_m", "prepared_longitudinal_position_m", "leader_x"), ndigits), _format_number(used(follower, "initial_longitudinal_position_m", "prepared_longitudinal_position_m", "follower_x_true"), ndigits)],
        ["Initial speed (m/s)", _format_number(used(leader, "initial_speed_m_s", "prepared_speed_m_s", "leader_v"), ndigits), _format_number(used(follower, "initial_speed_m_s", "prepared_speed_m_s", "follower_v_true"), ndigits)],
        ["Initial acceleration (m/s²)", _format_number(used(leader, "initial_acceleration_m_s2", "prepared_acceleration_m_s2", "leader_a"), ndigits), _format_number(used(follower, "initial_acceleration_m_s2", "prepared_acceleration_m_s2", "follower_a_true"), ndigits)],
        # Untouched source-frame observation, retained for traceability.
        ["Observed source speed (m/s)", _format_number(item(leader, "source_frame_speed_m_s", item(leader, "raw_speed_m_s")), ndigits), _format_number(item(follower, "source_frame_speed_m_s", item(follower, "raw_speed_m_s")), ndigits)],
        ["Observed source acceleration (m/s²)", _format_number(item(leader, "source_frame_acceleration_m_s2", item(leader, "raw_acceleration_m_s2")), ndigits), _format_number(item(follower, "source_frame_acceleration_m_s2", item(follower, "raw_acceleration_m_s2")), ndigits)],
        ["Vehicle length (m)", _format_number(item(leader, "vehicle_length_m", fallback["leader_length"]), ndigits), _format_number(item(follower, "vehicle_length_m"), ndigits)],
        ["Lane ID", item(leader, "lane_id", fallback.get("lane_id", "unknown")), item(follower, "lane_id", fallback.get("lane_id", "unknown"))],
    ]


def _parameters_by_model(
    config: Mapping[str, Any], status_payload: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    configured_models = _case_insensitive_get(config, "models", {})
    result: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        status_parameters = extract_model_status(status_payload, model).get("parameters", {})
        if isinstance(status_parameters, Mapping) and status_parameters:
            result[model] = dict(status_parameters)
            continue
        config_parameters = (
            _case_insensitive_get(configured_models, model, {})
            if isinstance(configured_models, Mapping)
            else {}
        )
        result[model] = dict(config_parameters) if isinstance(config_parameters, Mapping) else {}
    return result


def write_experiment_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    truth_path: Path,
    initial_path: Path,
    status_path: Path,
    truth: pd.DataFrame,
    initial: Mapping[str, Any],
    selected_case: Mapping[str, Any],
    status_payload: Mapping[str, Any],
    metrics: pd.DataFrame,
    identity: Mapping[str, Any],
    errors: pd.DataFrame | None = None,
    braking_threshold_m_s2: float,
    braking_consecutive_frames: int,
    response_search_window_s: float,
    reacceleration_threshold_m_s2: float,
    leader_braking_onset_s: float,
    observed_response_onset_s: float,
    observed_response_lag_s: float,
    leader_reacceleration_onset_s: float,
    collision_gap_m: float,
    generated_files: Sequence[Path],
) -> None:
    relative = initial.get("relative", {}) if isinstance(initial.get("relative", {}), Mapping) else {}
    processing = initial.get("processing", {}) if isinstance(initial.get("processing", {}), Mapping) else {}
    parameters = _parameters_by_model(config, status_payload)
    metadata = status_payload.get("metadata", {}) if isinstance(status_payload.get("metadata", {}), Mapping) else {}

    absolute_start = (
        selected_case.get("start_time_s")
        if selected_case
        else (truth["time_s"].iloc[0] if "time_s" in truth.columns else truth["t"].iloc[0])
    )
    absolute_end = (
        selected_case.get("end_time_s")
        if selected_case
        else (truth["time_s"].iloc[-1] if "time_s" in truth.columns else truth["t"].iloc[-1])
    )
    duration = float(truth["t"].iloc[-1] - truth["t"].iloc[0])
    selection_reason = selected_case.get(
        "selection_reason",
        "由上游候选排序与轨迹复核选定；本评价步骤不重新筛选车辆对。",
    )

    initial_rows = _initial_state_rows(initial, truth)
    metric_rows = []
    for model in MODEL_ORDER:
        row = metrics.loc[metrics["model"] == model].iloc[0]
        response_value = (
            _format_number(row["response_time_lag_s"], 4)
            if bool(row["response_lag_evaluable"])
            else f"NaN（{row['response_lag_reason']}）"
        )
        metric_rows.append(
            [
                model,
                _format_number(row["position_rmse_m"], 4),
                _format_number(row["speed_rmse_m_s"], 4),
                _format_number(row["acceleration_rmse_m_s2"], 4),
                _format_number(row["minimum_net_gap_m"], 4),
                response_value,
                _format_number(row["final_position_error_m"], 4),
                _format_number(row["common_valid_duration_s"], 4),
                f"{row['completion_percent']:.2f}%",
                str(row["status"]),
            ]
        )

    parameter_rows = []
    for model in MODEL_ORDER:
        parameter_text = ", ".join(f"`{key}={value}`" for key, value in parameters[model].items()) or "未记录"
        parameter_rows.append([model, parameter_text])

    failure_lines = []
    for model in MODEL_ORDER:
        row = metrics.loc[metrics["model"] == model].iloc[0]
        if bool(row["completed"]):
            failure_lines.append(f"- **{model}**：完整运行；未检测到碰撞。")
        else:
            when = _format_number(row["failure_time_s"], 4, " s")
            reason = row["failure_reason"] or "未提供明确原因"
            collision_text = "检测到碰撞；" if bool(row["collision"]) else "未检测到碰撞边界穿越；"
            failure_lines.append(
                f"- **{model}**：{collision_text}状态 `{row['status']}`，失败时间 {when}，"
                f"共同有效覆盖率 {row['completion_percent']:.2f}%，原因：{reason}。"
            )

    # Error-source decomposition: how much of each position misfit comes from the
    # initial transient (before the measured leader braking) versus the response
    # to the real gap/speed disturbance.
    source_rows: list[list[str]] = []
    source_lines: list[str] = []
    observed_initial_acceleration = _as_optional_float(
        (initial.get("follower", {}) or {}).get("initial_acceleration_m_s2")
    )
    if errors is not None and not errors.empty:
        early_window_s = 5.0
        initial_gap = _as_optional_float(initial.get("relative", {}).get("initial_gap_m"))
        for model in MODEL_ORDER:
            row = metrics.loc[metrics["model"] == model].iloc[0]
            model_errors = errors.loc[errors["model"] == model]
            model_errors = model_errors.loc[
                model_errors["t"] <= float(row["common_valid_end_s"]) + 1e-9
            ]
            position_error = model_errors["position_error_m"].to_numpy(dtype=float)
            finite = np.isfinite(position_error)
            early = finite & (model_errors["t"].to_numpy(dtype=float) <= early_window_s)
            early_count = int(early.sum())
            total_count = int(finite.sum())
            squared_total = float(np.sum(position_error[finite] ** 2))
            squared_early = float(np.sum(position_error[early] ** 2))
            early_rmse = math.sqrt(squared_early / max(1, early_count))
            late_rmse = (
                math.sqrt((squared_total - squared_early) / max(1, total_count - early_count))
                if total_count > early_count
                else math.nan
            )
            share = squared_early / squared_total if squared_total > 0 else math.nan
            duration_fraction = early_count / total_count if total_count else math.nan
            model_status = status_payload.get("models", {}).get(model, {})
            initial_acceleration = _as_optional_float(
                (model_status.get("initial_state", {}) or {}).get("a_m_s2")
            )
            never_closed = bool(
                initial_gap is not None
                and math.isfinite(initial_gap)
                # The metric is rounded to six decimals, hence the 1e-5 tolerance.
                and float(row["minimum_net_gap_m"]) >= initial_gap - 1e-5
            )
            lag_error = (
                _as_optional_float(row["response_lag_error_s"])
                if bool(row["response_lag_evaluable"])
                else math.nan
            )
            source_rows.append(
                [
                    model,
                    _format_number(initial_acceleration, 4),
                    _format_number(observed_initial_acceleration, 4),
                    _format_number(early_rmse, 4),
                    _format_number(late_rmse, 4),
                    (
                        f"{100.0 * share:.1f}%"
                        if math.isfinite(share)
                        else "不可评价"
                    ),
                    "是" if never_closed else "否",
                    (
                        _format_number(lag_error, 4)
                        if math.isfinite(lag_error)
                        else "不可评价"
                    ),
                ]
            )
            share_text = f"{100.0 * share:.1f}%" if math.isfinite(share) else "不可评价"
            fraction_text = (
                f"{100.0 * duration_fraction:.1f}%" if math.isfinite(duration_fraction) else "不可评价"
            )
            if math.isfinite(early_rmse) and math.isfinite(late_rmse) and early_rmse > 1.25 * late_rmse:
                source_lines.append(
                    f"- **{model}** 以起步瞬态为主：0–{early_window_s:g} s 的 RMSE 为"
                    f" {_format_number(early_rmse, 4)} m，明显高于其后的"
                    f" {_format_number(late_rmse, 4)} m；该窗口只占时长的 {fraction_text}，"
                    f"却贡献了 {share_text} 的位置平方误差。"
                )
            elif math.isfinite(early_rmse) and math.isfinite(late_rmse):
                source_lines.append(
                    f"- **{model}** 的误差更多来自后段对前车扰动的响应：0–{early_window_s:g} s"
                    f" RMSE 为 {_format_number(early_rmse, 4)} m，其后为"
                    f" {_format_number(late_rmse, 4)} m；前段占时长 {fraction_text}，"
                    f"贡献 {share_text} 的位置平方误差。"
                )
            if never_closed and initial_gap is not None:
                source_lines.append(
                    f"- **{model}** 的最小净间距等于初始净间距 {_format_number(initial_gap, 4)} m"
                    "（在六位小数舍入容差内），说明整段仿真从未把间距压缩到初始值以下，"
                    "没有复现真实后车先接近再拉开的间距过程。"
                )
            if math.isfinite(lag_error):
                direction = "晚" if lag_error > 0 else "早"
                source_lines.append(
                    f"- **{model}** 的制动响应比真实后车{direction} {abs(lag_error):.1f} s"
                    "（阈值与连续帧规则见上一节，改变阈值会改变该数值）。"
                )
    if observed_initial_acceleration is not None and math.isfinite(observed_initial_acceleration):
        model_initial = [
            _as_optional_float(
                (status_payload.get("models", {}).get(model, {}).get("initial_state", {}) or {}).get(
                    "a_m_s2"
                )
            )
            for model in MODEL_ORDER
        ]
        readable = ", ".join(
            f"{model} {value:.2f} m/s²"
            for model, value in zip(MODEL_ORDER, model_initial)
            if value is not None and math.isfinite(value)
        )
        source_lines.append(
            f"- 三模型在 t=0 的加速度由各自方程给出（{readable}），而真实后车观测初始加速度为"
            f" {observed_initial_acceleration:.2f} m/s²。该差异来自各自的平衡态关系，"
            "不是把真值当作初始条件。"
        )

    completed_rows = metrics.loc[metrics["completed"]]
    comparison_lines = []
    if completed_rows.empty:
        comparison_lines.append("- 没有模型完整覆盖参考时段，因此不进行跨模型优劣排名。")
    else:
        for column, label in (
            ("position_rmse_m", "位置 RMSE"),
            ("speed_rmse_m_s", "速度 RMSE"),
            ("acceleration_rmse_m_s2", "加速度 RMSE"),
        ):
            finite = completed_rows.loc[np.isfinite(completed_rows[column])]
            if not finite.empty:
                best = finite.loc[finite[column].idxmin()]
                comparison_lines.append(
                    f"- 在**完整运行模型**中，{label}最低的是 {best['model']} "
                    f"（{_format_number(best[column], 4)}）；这是单一片段上的描述性结果，不代表泛化性能。"
                )
    if (~metrics["completed"]).any():
        comparison_lines.append(
            "- 未完成模型的 RMSE 仅基于其共同有效前缀，不能与完整时段 RMSE 直接排名；覆盖率和失败状态必须同时阅读。"
        )

    processing_lines = []
    if processing:
        for key, value in processing.items():
            processing_lines.append(f"- `{key}`：`{value}`")
    else:
        preparation_config = _case_insensitive_get(config, "preparation", {})
        if isinstance(preparation_config, Mapping):
            for key, value in preparation_config.items():
                processing_lines.append(f"- 配置 `{key}`：`{value}`")
    if not processing_lines:
        processing_lines.append("- 初始化文件未提供额外处理元数据；应回查上游准备步骤日志。")

    generated_relative = []
    for generated in sorted(set(generated_files)):
        try:
            generated_relative.append(f"- `{generated.relative_to(PROJECT_ROOT).as_posix()}`")
        except ValueError:
            generated_relative.append(f"- `{generated}`")

    dt = metadata.get(
        "dt",
        metadata.get(
            "time_step_s",
            _config_value(config, "simulation", ("time_step_s", "dt"), _infer_sampling_interval(truth["t"])),
        ),
    )
    integration = metadata.get("integration_method", "速度显式更新 + 梯形位置积分")
    observed_lag_text = (
        _format_number(observed_response_lag_s, 4, " s")
        if math.isfinite(observed_response_lag_s)
        else "NaN（真实后车在规则窗口内未检测到有效制动响应）"
    )

    report = f"""# NGSIM I-80 跟驰模型复现与对比实验报告

生成时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec="seconds")}

## 1. 数据集与评价范围

本实验使用 NGSIM I-80 Eastbound（Emeryville, California）2005-04-13 16:00–16:15 PDT 数据，标称采样率 10 Hz。输入为 SI 版本，位置、速度、加速度分别采用 m、m/s、m/s²，未执行二次单位换算。`local_y_m`/标准化 `x` 表示车辆前端中心纵向位置；净间距定义为前车前端位置减后车前端位置再减前车长度。

本报告只评价上游已选定并清洗的单个跟驰片段，不重新筛选车辆对，也不修改原始或清洗轨迹。

- Leader ID：**{identity['leader_id']}**
- Follower ID：**{identity['follower_id']}**
- Lane ID：**{identity['lane_id']}**
- 数据时间段：{absolute_start}–{absolute_end} s；标准化时长：{duration:.3f} s；样本数：{len(truth)}
- 筛选说明：{selection_reason}

## 2. 初始条件

{_markdown_table(["State", "Leader", "Follower"], initial_rows)}

- 初始净间距：{_format_number(relative.get('initial_gap_m', truth['gap_true'].iloc[0]), 6)} m
- 初始相对速度 \\(v_L-v_F\\)：{_format_number(relative.get('initial_relative_speed_m_s', truth['leader_v'].iloc[0] - truth['follower_v_true'].iloc[0]), 6)} m/s
- 初始相对加速度 \\(a_L-a_F\\)：{_format_number(relative.get('initial_relative_acceleration_m_s2', truth['leader_a'].iloc[0] - truth['follower_a_true'].iloc[0]), 6)} m/s²

表中 `Initial ...` 行为三模型实际使用的初始化状态，与 `{_display_path(truth_path)}` 首行一致，并与上方初始净间距、相对速度满足 `s(0)=x_L(0)-x_F(0)-l_L`；`Observed source ...` 行为未改动的原始观测，仅用于追溯平滑前后的差异（原始位置见初始化 JSON）。

真实后车初始加速度仅作为观测值；各模型初始加速度由自身方程计算。

## 3. 数据处理与可追溯性

评价真值来自 `{_display_path(truth_path)}`，初始化记录来自 `{_display_path(initial_path)}`，仿真状态来自 `{_display_path(status_path)}`。配置文件为 `{_display_path(config_path)}`。绘图直接使用这些 CSV 中的真实计算数据，没有插值补齐失败后的模型轨迹，也没有跨缺失区间连线。

{chr(10).join(processing_lines)}

## 4. 模型与参数

- **OVM**：\\(a_F=\\kappa[V(s)-v_F]\\)，其中 \\(V(s)\\) 为归一化双曲正切最优速度函数。
- **FVDM**：\\(a_F=\\kappa[V(s)-v_F]+\\lambda(v_L-v_F)\\)，与 OVM 共用同一形式的 \\(V(s)\\)。
- **IDM**：\\(a_F=a[1-(v_F/v_0)^\\delta-(s^*/s)^2]\\)，\\(s^*=s_0+\\max(0,v_FT+v_F(v_F-v_L)/(2\\sqrt{{ab}}))\\)。

{_markdown_table(["Model", "本次运行参数"], parameter_rows)}

这些参数是本次运行记录的配置/状态参数；若未经独立训练片段标定，则不得称为真实最优参数。

## 5. 数值积分设置

- 时间步长：{dt} s
- 积分方法：{integration}
- 外部输入：三种模型均使用相同的真实前车位置与速度轨迹。
- 后车初始化：三种模型均使用相同的真实后车初始位置与速度；仿真开始后不使用真实后车状态纠偏。
- 碰撞边界：模拟净间距 ≤ {collision_gap_m:g} m。

## 6. 制动响应事件规则

Leader 制动事件定义为加速度 ≤ {braking_threshold_m_s2:g} m/s² 且连续至少 {braking_consecutive_frames} 帧，事件时刻取该连续序列的首帧。Follower/模型响应使用相同阈值和连续帧规则，并只在 Leader 事件后 {response_search_window_s:g} s 内搜索；响应时滞为响应首帧减 Leader 事件首帧。无法检测 Leader 事件或窗口内响应时输出 `NaN`，不填入任意数值。

- Leader 制动起点：{_format_number(leader_braking_onset_s, 4, ' s')}
- 真实 Follower 响应起点：{_format_number(observed_response_onset_s, 4, ' s')}
- 真实 Follower 响应时滞：{observed_lag_text}
- 图中重新加速事件规则：Leader 加速度 ≥ {reacceleration_threshold_m_s2:g} m/s² 且连续至少 {braking_consecutive_frames} 帧，并在制动事件之后搜索；检测时刻：{_format_number(leader_reacceleration_onset_s, 4, ' s')}。

## 7. 评价结果

RMSE 均在模型与真值从片段首帧开始的**连续共同有效前缀**上计算。最终位置误差是共同有效终点处的有符号误差 \\(x_{{sim}}-x_{{true}}\\)。最小净间距使用模型文件内全部有限记录（包括已记录的失败/碰撞点）。

{_markdown_table(["Model", "位置 RMSE (m)", "速度 RMSE (m/s)", "加速度 RMSE (m/s²)", "最小净间距 (m)", "响应时滞 (s)", "最终位置误差 (m)", "共同有效时长 (s)", "完成率", "状态"], metric_rows)}

### 描述性比较

{chr(10).join(comparison_lines)}

### 碰撞与失效分析

{chr(10).join(failure_lines)}

### 误差来源分解

下表把每个模型的位置平方误差拆成“0–5 s 起步瞬态”和“其后对前车扰动的响应”两部分，并按分段 RMSE、是否曾压缩初始净间距、响应时滞误差比较。数值全部由本轮仿真与真值直接计算。

{_markdown_table(["Model", "模型初始加速度 (m/s²)", "真实后车初始加速度 (m/s²)", "0–5 s 位置 RMSE (m)", "5 s 后位置 RMSE (m)", "0–5 s 误差占比", "最小净间距未低于初始值", "响应时滞误差 (s)"], source_rows)}

{chr(10).join(source_lines)}

## 8. 图表说明

模型用颜色和线型双重编码；PNG 按配置导出高分辨率栅格图，PDF 保留线条和文本的矢量表示。误差图中淡化的柱表示未完成/失败模型，柱上同时标出覆盖率与状态。所有坐标轴明确给出物理单位，零净间距/配置碰撞边界在间距图中单独标记。

{chr(10).join(generated_relative)}

## 9. 研究局限

1. 结果来自单一车辆对和单一时间窗，不能直接推广到其他驾驶人、车道或交通状态。
2. 未经独立片段标定的文献参考/示例参数只能用于复现流程；不能据此宣称某一模型具有普遍优势。
3. NGSIM 轨迹存在测量噪声，位置平滑及数值微分会影响速度、加速度和制动事件时刻；响应时滞还会随阈值、连续帧数和搜索窗变化。
4. 模型提前失效时，RMSE 仅覆盖失效前的共同有效区间；短区间误差可能看似较小，因此必须结合完成率、最小间距和失效原因解释。
5. 碰撞判据使用一维净间距边界，不包含横向重叠、车辆姿态、反应感知误差或车辆动力学约束。
6. 当前比较不含独立训练/测试划分的参数标定，也未量化参数不确定性或跨事件泛化误差。

## 10. 参考资料

- Bando, M. et al. (1995). Dynamical model of traffic congestion and numerical simulation. *Physical Review E*, 51, 1035. https://doi.org/10.1103/PhysRevE.51.1035
- Jiang, R., Wu, Q. S., & Zhu, Z. J. (2001). Full velocity difference model for a car-following theory. *Physical Review E*, 64, 017101. https://doi.org/10.1103/PhysRevE.64.017101
- Treiber, M., Hennecke, A., & Helbing, D. (2000). Congested traffic states in empirical observations and microscopic simulations. *Physical Review E*, 62, 1805. https://doi.org/10.1103/PhysRevE.62.1805
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
"""
    path.write_text(report, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def run_evaluation(config: Mapping[str, Any], config_path: Path) -> dict[str, Path]:
    prepared_dir = resolve_from_project(
        _config_value(config, "paths", ("prepared_dir",), "outputs/04_prepared")
    )
    simulation_dir = resolve_from_project(
        _config_value(config, "paths", ("simulation_dir",), "outputs/05_simulation")
    )
    output_dir = resolve_from_project(
        _config_value(config, "paths", ("evaluation_dir",), "outputs/06_evaluation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_path = prepared_dir / "trajectory_clean.csv"
    initial_path = prepared_dir / "initial_conditions.json"
    status_path = simulation_dir / "simulation_status.json"
    truth = load_truth(truth_path)
    initial = _read_json(initial_path)
    status_payload = _read_json(status_path)

    selected_case_path = resolve_from_project(
        _config_value(config, "paths", ("cases_dir",), "outputs/03_cases")
    ) / "selected_case.json"
    selected_case = _read_json(selected_case_path, required=False)

    simulations = {
        model: load_simulation(find_simulation_file(simulation_dir, model), model)
        for model in MODEL_ORDER
    }

    fallback_dt = float(_config_value(config, "data", ("sampling_interval_s",), 0.1))
    dt = _infer_sampling_interval(truth["t"], fallback_dt)
    time_tolerance = float(
        _config_value(config, "data", ("time_grid_tolerance_s",), max(1e-8, dt * 1e-5))
    )
    braking_threshold = float(
        _config_value(
            config,
            "evaluation",
            ("braking_acceleration_threshold_m_s2", "braking_accel_threshold_m_s2"),
            -0.5,
        )
    )
    braking_frames = int(
        _config_value(
            config,
            "evaluation",
            ("braking_consecutive_frames", "response_consecutive_frames"),
            3,
        )
    )
    response_window = float(
        _config_value(config, "evaluation", ("response_search_window_s",), 8.0)
    )
    reacceleration_threshold = float(
        _config_value(
            config,
            "evaluation",
            ("reacceleration_acceleration_threshold_m_s2", "reaccel_accel_threshold_m_s2"),
            max(0.3, abs(braking_threshold)),
        )
    )
    collision_gap = float(
        _config_value(config, "simulation", ("collision_gap_m",), 0.0)
    )
    dpi = int(_config_value(config, "evaluation", ("png_dpi",), 300))
    figure_size = (
        float(_config_value(config, "evaluation", ("figure_width_in",), 12.0)),
        float(_config_value(config, "evaluation", ("figure_height_in",), 8.0)),
    )

    if braking_threshold >= 0:
        raise EvaluationError("Braking acceleration threshold must be negative")
    if braking_frames < 1:
        raise EvaluationError("Braking consecutive-frame count must be >= 1")
    if response_window <= 0:
        raise EvaluationError("Response search window must be positive")
    if time_tolerance <= 0:
        raise EvaluationError("Time-grid tolerance must be positive")
    if dpi < 72:
        raise EvaluationError("PNG DPI must be at least 72")

    maximum_frame_gap = dt * 1.5
    leader_braking_onset = find_consecutive_event(
        truth["t"],
        truth["leader_a"],
        braking_threshold,
        braking_frames,
        direction="below",
        maximum_frame_gap_s=maximum_frame_gap,
    )
    if math.isfinite(leader_braking_onset):
        event_search_end = leader_braking_onset + response_window
        observed_response_onset = find_consecutive_event(
            truth["t"],
            truth["follower_a_true"],
            braking_threshold,
            braking_frames,
            direction="below",
            start_time_s=leader_braking_onset,
            end_time_s=event_search_end,
            maximum_frame_gap_s=maximum_frame_gap,
        )
        leader_reacceleration_onset = find_consecutive_event(
            truth["t"],
            truth["leader_a"],
            reacceleration_threshold,
            braking_frames,
            direction="above",
            start_time_s=leader_braking_onset + braking_frames * dt,
            maximum_frame_gap_s=maximum_frame_gap,
        )
    else:
        observed_response_onset = leader_reacceleration_onset = math.nan
    observed_response_lag = (
        float(observed_response_onset - leader_braking_onset)
        if math.isfinite(observed_response_onset) and math.isfinite(leader_braking_onset)
        else math.nan
    )

    metric_records: list[dict[str, Any]] = []
    error_frames: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        model_metrics, model_errors = evaluate_model(
            model,
            truth,
            simulations[model],
            extract_model_status(status_payload, model),
            time_tolerance_s=time_tolerance,
            collision_gap_m=collision_gap,
            braking_threshold_m_s2=braking_threshold,
            braking_consecutive_frames=braking_frames,
            response_search_window_s=response_window,
            leader_braking_onset_s=leader_braking_onset,
            observed_response_onset_s=observed_response_onset,
            observed_response_lag_s=observed_response_lag,
            maximum_frame_gap_s=maximum_frame_gap,
        )
        metric_records.append(model_metrics)
        error_frames.append(model_errors)

    metrics = pd.DataFrame(metric_records)
    errors = pd.concat(error_frames, ignore_index=True)
    metrics_path = output_dir / "evaluation_metrics.csv"
    errors_path = output_dir / "error_timeseries.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.10g")
    errors.to_csv(errors_path, index=False, float_format="%.10g")

    _set_plot_style()
    identity = _identity(initial, truth)
    generated: list[Path] = [metrics_path, errors_path]
    generated.extend(plot_trajectory_selection(truth, identity, output_dir, figure_size, dpi))
    generated.extend(plot_position_comparison(truth, simulations, output_dir, figure_size, dpi))
    generated.extend(
        plot_speed_response(
            truth,
            simulations,
            leader_braking_onset,
            leader_reacceleration_onset,
            observed_response_onset,
            output_dir,
            figure_size,
            dpi,
        )
    )
    generated.extend(
        plot_acceleration_response(
            truth,
            simulations,
            leader_braking_onset,
            leader_reacceleration_onset,
            observed_response_onset,
            output_dir,
            figure_size,
            dpi,
        )
    )
    generated.extend(
        plot_gap_comparison(truth, simulations, collision_gap, output_dir, figure_size, dpi)
    )
    generated.extend(plot_error_comparison(errors, metrics, output_dir, figure_size, dpi))
    generated.extend(
        plot_model_comparison(
            truth,
            simulations,
            errors,
            metrics,
            leader_braking_onset,
            leader_reacceleration_onset,
            observed_response_onset,
            collision_gap,
            output_dir,
            figure_size,
            dpi,
        )
    )

    report_path = output_dir / "experiment_report.md"
    # Include the report and metadata in the generated-file inventory before
    # rendering the report, so its provenance section is self-contained.
    metadata_path = output_dir / "evaluation_metadata.json"
    generated.extend([report_path, metadata_path])
    write_experiment_report(
        report_path,
        config=config,
        config_path=config_path,
        truth_path=truth_path,
        initial_path=initial_path,
        status_path=status_path,
        truth=truth,
        initial=initial,
        selected_case=selected_case,
        status_payload=status_payload,
        metrics=metrics,
        identity=identity,
        errors=errors,
        braking_threshold_m_s2=braking_threshold,
        braking_consecutive_frames=braking_frames,
        response_search_window_s=response_window,
        reacceleration_threshold_m_s2=reacceleration_threshold,
        leader_braking_onset_s=leader_braking_onset,
        observed_response_onset_s=observed_response_onset,
        observed_response_lag_s=observed_response_lag,
        leader_reacceleration_onset_s=leader_reacceleration_onset,
        collision_gap_m=collision_gap,
        generated_files=generated,
    )

    metadata_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "config": str(config_path),
            "trajectory": str(truth_path),
            "initial_conditions": str(initial_path),
            "simulation_status": str(status_path),
            "simulations": {
                model: str(find_simulation_file(simulation_dir, model)) for model in MODEL_ORDER
            },
        },
        "alignment": {
            "method": "nearest timestamp match followed by contiguous valid prefix",
            "time_tolerance_s": time_tolerance,
            "reference_samples": len(truth),
        },
        "response_event_rule": {
            "braking_acceleration_threshold_m_s2": braking_threshold,
            "consecutive_frames": braking_frames,
            "response_search_window_s": response_window,
            "reacceleration_acceleration_threshold_m_s2": reacceleration_threshold,
            "leader_braking_onset_s": leader_braking_onset,
            "observed_follower_response_onset_s": observed_response_onset,
            "observed_follower_response_lag_s": observed_response_lag,
            "leader_reacceleration_onset_s": leader_reacceleration_onset,
        },
        "collision_gap_m": collision_gap,
        "figure_export": {
            "png_dpi": dpi,
            "figure_width_in": figure_size[0],
            "figure_height_in": figure_size[1],
            "formats": ["png", "pdf"],
            "colour_redundancy": "model colour plus line style; bars also use hatching",
        },
        "outputs": [str(item) for item in generated],
        "metrics": metrics.to_dict(orient="records"),
    }
    metadata_path.write_text(
        json.dumps(_json_safe(metadata_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Evaluation complete: %s", output_dir)
    return {
        "metrics": metrics_path,
        "errors": errors_path,
        "report": report_path,
        "metadata": metadata_path,
        "model_comparison_png": output_dir / "model_comparison.png",
        "model_comparison_pdf": output_dir / "model_comparison.pdf",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate OVM/FVDM/IDM simulations and create figures/report."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="YAML configuration path; relative paths are resolved from the project root.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        config, config_path = load_config(args.config)
        outputs = run_evaluation(config, config_path)
    except (EvaluationError, FileNotFoundError, json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 1
    LOGGER.info("Metrics: %s", outputs["metrics"])
    LOGGER.info("Report: %s", outputs["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
