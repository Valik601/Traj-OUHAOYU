"""Simulate the circle-antipode experiment and compare it with observations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_circle import (
    CENTER_RADIUS_M,
    FPS,
    add_behavior_flags,
    add_interaction_risk,
    add_kinematics,
    extract_episodes,
    load_trajectories,
    pedestrian_summary,
)
from model import ModelParameters, PredictiveVelocityModel, SimulationResult


def prepare_observed(path: Path) -> pd.DataFrame:
    return add_kinematics(load_trajectories(path))


def initial_state(observed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = observed.sort_values(["frame", "pedestrian"]).groupby("pedestrian", sort=True).first().reset_index()
    pedestrian_ids = first["pedestrian"].to_numpy(int)
    positions = first[["x_smooth", "y_smooth"]].to_numpy(float)
    velocities = first[["vx", "vy"]].to_numpy(float)
    speed = np.linalg.norm(velocities, axis=1)
    velocities[speed > 1.2] *= (1.2 / speed[speed > 1.2])[:, None]
    goals = -positions
    frames = np.sort(observed["frame"].unique()).astype(int)
    return pedestrian_ids, positions, velocities, goals, frames


def enrich_simulation(raw: pd.DataFrame) -> pd.DataFrame:
    columns = ["pedestrian", "frame", "time", "x", "y"]
    if "hard_core_correction_m" in raw.columns:
        columns.append("hard_core_correction_m")
    data = raw[columns].copy()
    data["x_cm"] = data["x"] * 100
    data["y_cm"] = data["y"] * 100
    data["height_cm"] = np.nan
    data = add_kinematics(data)
    data = add_interaction_risk(data)
    data = add_behavior_flags(data)
    return data


def minimum_pair_distances(data: pd.DataFrame, x_column: str, y_column: str) -> pd.Series:
    rows: list[tuple[int, float]] = []
    for frame, group in data.groupby("frame", sort=True):
        positions = group[[x_column, y_column]].to_numpy(float)
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, np.inf)
        rows.append((int(frame), float(distance.min())))
    return pd.Series(dict(rows), name="minimum_pair_distance_m")


def route_counts(summary: pd.DataFrame) -> dict[str, int]:
    counts = summary["route_side"].value_counts()
    return {side: int(counts.get(side, 0)) for side in ["right", "left", "central"]}


def dataset_metrics(
    name: str,
    data: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    goals: np.ndarray,
) -> dict[str, float | int | str]:
    final = data.sort_values(["pedestrian", "frame"]).groupby("pedestrian", sort=True).last()
    final_positions = final[["x_smooth", "y_smooth"]].to_numpy(float)
    final_error = np.linalg.norm(final_positions - goals, axis=1)
    pair_distance = minimum_pair_distances(data, "x_smooth", "y_smooth")
    center = data[data["radius"] <= CENTER_RADIUS_M]
    counts = route_counts(summary)
    classified = counts["right"] + counts["left"]
    result = {
        "dataset": name,
        "completion_rate_0.5m": float(np.mean(final_error <= 0.5)),
        "median_final_goal_error_m": float(np.median(final_error)),
        "median_speed_m_s": float(data["speed"].median()),
        "center_median_speed_m_s": float(center["speed"].median()),
        "minimum_pair_distance_m": float(pair_distance.min()),
        "close_contact_frame_rate_lt_0.6m": float(np.mean(pair_distance < 0.6)),
        "right_routes": counts["right"],
        "left_routes": counts["left"],
        "central_routes": counts["central"],
        "right_share_excluding_central": float(counts["right"] / classified) if classified else math.nan,
        "conflict_events": int((events["event_type"] == "conflict").sum()),
        "deceleration_events": int((events["event_type"] == "deceleration").sum()),
        "turning_events": int((events["event_type"] == "turning").sum()),
        "avoidance_events": int((events["event_type"] == "avoidance").sum()),
    }
    if "hard_core_correction_m" in data.columns:
        result["hard_core_activation_rate"] = float((data["hard_core_correction_m"] > 1e-6).mean())
        result["maximum_hard_core_correction_m"] = float(data["hard_core_correction_m"].max())
    else:
        result["hard_core_activation_rate"] = math.nan
        result["maximum_hard_core_correction_m"] = math.nan
    return result


def comparison_errors(observed: pd.DataFrame, simulated: pd.DataFrame) -> dict[str, float]:
    merged = observed[["pedestrian", "frame", "x_smooth", "y_smooth", "speed"]].merge(
        simulated[["pedestrian", "frame", "x_smooth", "y_smooth", "speed"]],
        on=["pedestrian", "frame"],
        suffixes=("_observed", "_simulated"),
    )
    position_error = np.hypot(
        merged["x_smooth_simulated"] - merged["x_smooth_observed"],
        merged["y_smooth_simulated"] - merged["y_smooth_observed"],
    )
    median_observed = merged.groupby("frame")["speed_observed"].median()
    median_simulated = merged.groupby("frame")["speed_simulated"].median()
    return {
        "time_aligned_ADE_m": float(position_error.mean()),
        "time_aligned_median_position_error_m": float(position_error.median()),
        "median_speed_curve_RMSE_m_s": float(np.sqrt(np.mean((median_simulated - median_observed) ** 2))),
    }


def plot_trajectory_comparison(
    observed: pd.DataFrame, model: pd.DataFrame, baseline: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    colors = ["#4C78A8", "#009E73", "#999999"]
    for ax, data, title, color in zip(
        axes,
        [observed, model, baseline],
        ["Observed", "Predictive model", "Straight-goal baseline"],
        colors,
    ):
        for _, track in data.groupby("pedestrian", sort=True):
            ax.plot(track["x_smooth"], track["y_smooth"], color=color, alpha=0.42, lw=0.65)
        ax.add_patch(plt.Circle((0, 0), 10, fill=False, ls="--", lw=1, color="#444444"))
        ax.add_patch(plt.Circle((0, 0), 2, color="#D55E00", alpha=0.06, lw=0))
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)", xlim=(-11, 11), ylim=(-11, 11))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    fig.suptitle("Circle-antipode trajectory comparison", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_speed_comparison(
    observed: pd.DataFrame, model: pd.DataFrame, baseline: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    palette = {"Observed": "#4C78A8", "Model": "#009E73", "Baseline": "#999999"}
    for label, data in [("Observed", observed), ("Model", model), ("Baseline", baseline)]:
        stats = data.groupby("time")["speed"].quantile([0.25, 0.5, 0.75]).unstack()
        axes[0].plot(stats.index, stats[0.5], color=palette[label], lw=2, label=label)
        if label != "Baseline":
            axes[0].fill_between(stats.index, stats[0.25], stats[0.75], color=palette[label], alpha=0.12)
    axes[0].set(title="Median speed over time (bands: IQR)", xlabel="time (s)", ylabel="speed (m/s)")
    axes[0].legend(frameon=False)

    bins = np.arange(0, 10.51, 0.5)
    for label, data in [("Observed", observed), ("Model", model), ("Baseline", baseline)]:
        copy = data.copy()
        copy["radius_bin"] = pd.cut(copy["radius"], bins=bins, include_lowest=True)
        radial = copy.groupby("radius_bin", observed=True)["speed"].median()
        centers = np.array([interval.mid for interval in radial.index])
        axes[1].plot(centers, radial, color=palette[label], lw=2, marker="o", ms=2.5, label=label)
    axes[1].axvspan(0, 2, color="#D55E00", alpha=0.07)
    axes[1].set(title="Median speed by centre distance", xlabel="radius (m)", ylabel="speed (m/s)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_avoidance_process(
    observed: pd.DataFrame, model: pd.DataFrame, decisions: pd.DataFrame, output: Path
) -> int:
    active = decisions[decisions["action"].isin(["brake", "turn", "brake_and_turn"])]
    pedestrian = int(active["pedestrian"].value_counts().index[0]) if not active.empty else int(decisions["pedestrian"].iloc[0])
    obs = observed[observed["pedestrian"] == pedestrian].sort_values("frame")
    sim = model[model["pedestrian"] == pedestrian].sort_values("frame")
    dec = decisions[decisions["pedestrian"] == pedestrian].sort_values("frame")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True)
    axes[0].plot(obs["time"], obs["speed"], color="#4C78A8", label="observed")
    axes[0].plot(sim["time"], sim["speed"], color="#009E73", label="model")
    axes[0].set(ylabel="speed (m/s)", title=f"Avoidance process example: pedestrian {pedestrian}")
    axes[0].legend(frameon=False)
    realised_turn = sim["turn_rate_deg_s"].where(sim["speed"] > 0.30).clip(-180, 180)
    axes[1].plot(sim["time"], realised_turn, color="#E69F00", label="realised turn rate")
    axes[1].plot(dec["frame"].sub(dec["frame"].min()).div(FPS), dec["selected_turn_deg"], color="#D55E00", alpha=0.65, label="selected heading offset")
    axes[1].set(ylabel="degrees/s or candidate degrees", ylim=(-180, 180))
    axes[1].legend(frameon=False)
    finite = dec["predicted_separation"].replace(np.inf, np.nan)
    axes[2].plot(dec["frame"].sub(dec["frame"].min()).div(FPS), finite, color="#7A5195", label="predicted closest separation")
    action = dec["action"].isin(["brake", "turn", "brake_and_turn"])
    axes[2].scatter(
        dec.loc[action, "frame"].sub(dec["frame"].min()).div(FPS),
        finite[action], s=9, color="#D55E00", alpha=0.6, label="avoidance action"
    )
    axes[2].axhline(0.78, color="#555555", ls="--", lw=1, label="comfort clearance")
    axes[2].set(xlabel="time (s)", ylabel="distance (m)", ylim=(0, 2.2))
    axes[2].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pedestrian


def run_stability_trials(
    base_parameters: ModelParameters,
    pedestrian_ids: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    frames: np.ndarray,
    trials: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        params = replace(base_parameters, seed=1000 + trial)
        perturbed_positions = positions + rng.normal(0.0, 0.02, positions.shape)
        perturbed_velocities = velocities + rng.normal(0.0, 0.03, velocities.shape)
        perturbed_goals = -perturbed_positions
        desired = np.clip(params.desired_speed * rng.normal(1.0, 0.04, len(positions)), 1.55, 2.25)
        result = PredictiveVelocityModel(params).simulate(
            pedestrian_ids, perturbed_positions, perturbed_velocities, perturbed_goals, frames,
            desired_speeds=desired, use_avoidance=True
        )
        data = result.trajectories
        final = data.sort_values(["pedestrian", "frame"]).groupby("pedestrian", sort=True).last()
        final_error = np.linalg.norm(final[["x", "y"]].to_numpy() - perturbed_goals, axis=1)
        pair_distance = minimum_pair_distances(data, "x", "y")
        choice_counts = result.route_choices["route_side"].value_counts()
        classified = int(choice_counts.get("right", 0) + choice_counts.get("left", 0))
        rows.append(
            {
                "trial": trial,
                "seed": params.seed,
                "completion_rate_0.5m": float(np.mean(final_error <= 0.5)),
                "median_final_goal_error_m": float(np.median(final_error)),
                "minimum_pair_distance_m": float(pair_distance.min()),
                "close_contact_frame_rate_lt_0.6m": float(np.mean(pair_distance < 0.6)),
                "right_share_excluding_central": float(choice_counts.get("right", 0) / classified),
                "mean_selected_speed_m_s": float(result.decisions["selected_speed"].mean()),
                "avoidance_action_rate": float(result.decisions["action"].isin(["brake", "turn", "brake_and_turn"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def stability_summary(trials: pd.DataFrame) -> dict[str, object]:
    numeric = [column for column in trials.columns if column not in {"trial", "seed"}]
    summary: dict[str, object] = {}
    for column in numeric:
        summary[column] = {
            "minimum": float(trials[column].min()),
            "median": float(trials[column].median()),
            "maximum": float(trials[column].max()),
        }
    gates = {
        "all_trials_complete_at_least_90_percent": bool((trials["completion_rate_0.5m"] >= 0.90).all()),
        "no_near_coincident_tracks_below_0.35m": bool((trials["minimum_pair_distance_m"] >= 0.35).all()),
        "right_share_between_55_and_90_percent": bool(
            trials["right_share_excluding_central"].between(0.55, 0.90).all()
        ),
    }
    summary["gates"] = gates
    summary["stable"] = bool(all(gates.values()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=script_dir.parent / "datasets" / "circle-10m-64-1.txt")
    parser.add_argument("--output", type=Path, default=script_dir / "results" / "simulation")
    parser.add_argument("--stability-trials", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    observed = prepare_observed(args.input.resolve())
    pedestrian_ids, positions, velocities, goals, frames = initial_state(observed)
    parameters = ModelParameters()
    model_result = PredictiveVelocityModel(parameters).simulate(
        pedestrian_ids, positions, velocities, goals, frames, use_avoidance=True
    )
    baseline_parameters = replace(parameters, route_offset=0.0, seed=parameters.seed)
    baseline_result = PredictiveVelocityModel(baseline_parameters).simulate(
        pedestrian_ids, positions, velocities, goals, frames, use_avoidance=False
    )
    model_data = enrich_simulation(model_result.trajectories)
    baseline_data = enrich_simulation(baseline_result.trajectories)
    observed_risk = add_behavior_flags(add_interaction_risk(observed))
    observed_events = extract_episodes(observed_risk)
    model_events = extract_episodes(model_data)
    baseline_events = extract_episodes(baseline_data)
    observed_summary = pedestrian_summary(observed_risk, observed_events)
    model_summary = pedestrian_summary(model_data, model_events)
    baseline_summary = pedestrian_summary(baseline_data, baseline_events)

    metrics = pd.DataFrame(
        [
            dataset_metrics("observed", observed_risk, observed_events, observed_summary, goals),
            dataset_metrics("predictive_model", model_data, model_events, model_summary, goals),
            dataset_metrics("straight_baseline", baseline_data, baseline_events, baseline_summary, goals),
        ]
    )
    model_errors = comparison_errors(observed_risk, model_data)
    baseline_errors = comparison_errors(observed_risk, baseline_data)
    stability = run_stability_trials(
        parameters, pedestrian_ids, positions, velocities, goals, frames, args.stability_trials
    )
    stability_report = stability_summary(stability)

    model_result.trajectories.to_csv(args.output / "simulation_trajectory.csv", index=False, float_format="%.6f")
    model_result.decisions.to_csv(args.output / "simulation_decisions.csv", index=False, float_format="%.6f")
    model_result.route_choices.to_csv(args.output / "simulation_route_choices.csv", index=False, float_format="%.6f")
    model_events.to_csv(args.output / "simulation_behavior_events.csv", index=False, float_format="%.6f")
    metrics.to_csv(args.output / "simulation_metrics.csv", index=False, float_format="%.6f")
    stability.to_csv(args.output / "stability_trials.csv", index=False, float_format="%.6f")

    plot_trajectory_comparison(observed_risk, model_data, baseline_data, args.output / "trajectory_comparison.png")
    plot_speed_comparison(observed_risk, model_data, baseline_data, args.output / "speed_comparison.png")
    example_pedestrian = plot_avoidance_process(
        observed_risk, model_data, model_result.decisions, args.output / "avoidance_process.png"
    )
    report = {
        "model": "predictive candidate-velocity model with biased tactical routing",
        "parameters": asdict(parameters),
        "comparison_metrics": metrics.to_dict(orient="records"),
        "trajectory_errors": {
            "predictive_model": model_errors,
            "straight_baseline": baseline_errors,
        },
        "stability": stability_report,
        "avoidance_example_pedestrian": example_pedestrian,
        "known_failure_checks": {
            "near_coincident_tracks_below_0.35m": bool(metrics.loc[metrics["dataset"] == "predictive_model", "minimum_pair_distance_m"].iloc[0] < 0.35),
            "incomplete_agents_detected": bool(metrics.loc[metrics["dataset"] == "predictive_model", "completion_rate_0.5m"].iloc[0] < 1.0),
            "right_bias_outside_observed_95ci_target_rough_range": bool(
                not 0.55 <= metrics.loc[metrics["dataset"] == "predictive_model", "right_share_excluding_central"].iloc[0] <= 0.90
            ),
        },
    }
    (args.output / "simulation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
