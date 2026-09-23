"""Analyze the 10 m / 64 pedestrian circle-antipode experiment.

The input format is::

    pedestrian_id frame x_cm y_cm height_cm

The experiment is sampled at 25 Hz.  All generated tables and figures are
written next to this script by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.ndimage import binary_closing, binary_dilation, binary_opening
from scipy.signal import savgol_filter
from scipy.stats import binomtest, wilcoxon


FPS = 25.0
SMOOTH_WINDOW = 15  # 0.6 s; suppresses head-marker jitter while retaining avoidance motion.
BODY_DIAMETER_M = 0.60
CONFLICT_SEPARATION_M = 0.80
CONFLICT_HORIZON_S = 2.0
CENTER_RADIUS_M = 2.0


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_trajectories(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["pedestrian", "frame", "x_cm", "y_cm", "height_cm"],
    ).sort_values(["pedestrian", "frame"])
    data["x"] = data["x_cm"] / 100.0
    data["y"] = data["y_cm"] / 100.0
    data["time"] = (data["frame"] - data["frame"].min()) / FPS
    return data.reset_index(drop=True)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def add_kinematics(data: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for pedestrian, track in data.groupby("pedestrian", sort=True):
        track = track.copy().sort_values("frame")
        n = len(track)
        window = min(SMOOTH_WINDOW, n if n % 2 else n - 1)
        window = max(window, 5)
        x = track["x"].to_numpy(float)
        y = track["y"].to_numpy(float)
        track["x_smooth"] = savgol_filter(x, window, 2)
        track["y_smooth"] = savgol_filter(y, window, 2)
        vx = savgol_filter(x, window, 2, deriv=1, delta=1 / FPS)
        vy = savgol_filter(y, window, 2, deriv=1, delta=1 / FPS)
        speed = np.hypot(vx, vy)
        acceleration = savgol_filter(speed, window, 2, deriv=1, delta=1 / FPS)
        heading = np.unwrap(np.arctan2(vy, vx))
        turn_rate = savgol_filter(heading, window, 2, deriv=1, delta=1 / FPS)

        start = np.array([track["x_smooth"].iloc[0], track["y_smooth"].iloc[0]])
        goal = -start
        direction = goal - start
        straight_distance = float(np.linalg.norm(direction))
        unit = direction / straight_distance
        right_normal = np.array([unit[1], -unit[0]])
        pos = track[["x_smooth", "y_smooth"]].to_numpy(float)
        offset = pos - start

        track["vx"] = vx
        track["vy"] = vy
        track["speed"] = speed
        track["longitudinal_acceleration"] = acceleration
        track["heading"] = wrap_angle(heading)
        track["turn_rate_deg_s"] = np.degrees(turn_rate)
        track["radius"] = np.hypot(pos[:, 0], pos[:, 1])
        track["progress"] = offset @ unit / straight_distance
        track["right_lateral_deviation"] = offset @ right_normal
        track["distance_to_start"] = np.linalg.norm(pos - start, axis=1)
        track["distance_to_goal"] = np.linalg.norm(pos - goal, axis=1)
        track["goal_x"] = goal[0]
        track["goal_y"] = goal[1]
        pieces.append(track)
    return pd.concat(pieces, ignore_index=True)


def add_interaction_risk(data: pd.DataFrame) -> pd.DataFrame:
    """Add nearest-neighbour and constant-velocity closest-approach measures."""
    outputs: list[pd.DataFrame] = []
    for frame, group in data.groupby("frame", sort=True):
        group = group.copy().sort_values("pedestrian")
        pos = group[["x_smooth", "y_smooth"]].to_numpy(float)
        vel = group[["vx", "vy"]].to_numpy(float)
        n = len(group)
        delta_pos = pos[None, :, :] - pos[:, None, :]
        delta_vel = vel[None, :, :] - vel[:, None, :]
        distance = np.linalg.norm(delta_pos, axis=2)
        np.fill_diagonal(distance, np.inf)

        speed_sq = np.einsum("ijk,ijk->ij", delta_vel, delta_vel)
        dot = np.einsum("ijk,ijk->ij", delta_pos, delta_vel)
        time_to_cpa = np.divide(
            -dot,
            speed_sq,
            out=np.full((n, n), np.inf),
            where=speed_sq > 1e-8,
        )
        valid_future = (time_to_cpa > 0.05) & (time_to_cpa <= CONFLICT_HORIZON_S)
        clipped_t = np.clip(time_to_cpa, 0.0, CONFLICT_HORIZON_S)
        closest_vector = delta_pos + delta_vel * clipped_t[:, :, None]
        predicted_separation = np.linalg.norm(closest_vector, axis=2)
        predicted_separation[~valid_future] = np.inf
        np.fill_diagonal(predicted_separation, np.inf)

        nearest_idx = np.argmin(distance, axis=1)
        risk_idx = np.argmin(predicted_separation, axis=1)
        rows = np.arange(n)
        nearest_distance = distance[rows, nearest_idx]
        min_predicted_separation = predicted_separation[rows, risk_idx]
        min_time_to_cpa = time_to_cpa[rows, risk_idx]
        no_future_pair = ~np.isfinite(min_predicted_separation)
        min_time_to_cpa[no_future_pair] = np.inf
        risk_idx[no_future_pair] = nearest_idx[no_future_pair]

        conflict = (
            (min_predicted_separation < CONFLICT_SEPARATION_M)
            & (min_time_to_cpa <= CONFLICT_HORIZON_S)
        ) | (nearest_distance < BODY_DIAMETER_M)

        group["nearest_neighbor"] = group["pedestrian"].to_numpy()[nearest_idx]
        group["risk_neighbor"] = group["pedestrian"].to_numpy()[risk_idx]
        group["nearest_distance"] = nearest_distance
        group["predicted_separation"] = min_predicted_separation
        group["time_to_cpa"] = min_time_to_cpa
        group["conflict_raw"] = conflict
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True).sort_values(["pedestrian", "frame"]).reset_index(drop=True)


def clean_boolean(mask: np.ndarray, close_frames: int = 5, open_frames: int = 3) -> np.ndarray:
    result = binary_closing(mask, structure=np.ones(close_frames, dtype=bool))
    result = binary_opening(result, structure=np.ones(open_frames, dtype=bool))
    return result.astype(bool)


def add_behavior_flags(data: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for _, track in data.groupby("pedestrian", sort=True):
        track = track.copy().sort_values("frame")
        moving = track["speed"].to_numpy() > 0.30
        conflict = clean_boolean(track["conflict_raw"].to_numpy(bool), 5, 3)
        deceleration = clean_boolean(
            (track["longitudinal_acceleration"].to_numpy() < -0.60) & moving,
            5,
            3,
        )
        turning = clean_boolean(
            (np.abs(track["turn_rate_deg_s"].to_numpy()) > 35.0) & moving,
            5,
            3,
        )
        nearby_conflict = binary_dilation(conflict, structure=np.ones(31, dtype=bool))
        avoidance = clean_boolean(nearby_conflict & (deceleration | turning), 7, 3)
        track["conflict"] = conflict
        track["deceleration"] = deceleration
        track["turning"] = turning
        track["avoidance"] = avoidance
        outputs.append(track)
    return pd.concat(outputs, ignore_index=True)


def mode_or_nan(series: pd.Series) -> float:
    mode = series.dropna().mode()
    return float(mode.iloc[0]) if not mode.empty else math.nan


def extract_episodes(data: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "conflict": ("conflict", 3),
        "deceleration": ("deceleration", 4),
        "turning": ("turning", 3),
        "avoidance": ("avoidance", 3),
    }
    rows: list[dict[str, float | int | str]] = []
    event_id = 0
    for pedestrian, track in data.groupby("pedestrian", sort=True):
        track = track.copy().sort_values("frame").reset_index(drop=True)
        for event_type, (column, minimum_frames) in definitions.items():
            mask = track[column].to_numpy(bool)
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            for start_idx, end_idx in zip(starts, ends):
                if end_idx - start_idx + 1 < minimum_frames:
                    continue
                episode = track.iloc[start_idx : end_idx + 1]
                if event_type in {"conflict", "avoidance"}:
                    finite_prediction = episode["predicted_separation"].replace(np.inf, np.nan)
                    peak_index = (
                        finite_prediction.idxmin()
                        if finite_prediction.notna().any()
                        else episode["nearest_distance"].idxmin()
                    )
                elif event_type == "deceleration":
                    peak_index = episode["longitudinal_acceleration"].idxmin()
                else:
                    peak_index = episode["turn_rate_deg_s"].abs().idxmax()
                peak = episode.loc[peak_index]
                event_id += 1
                finite_sep = episode["predicted_separation"].replace(np.inf, np.nan)
                finite_ttc = episode["time_to_cpa"].replace(np.inf, np.nan)
                rows.append(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "pedestrian": int(pedestrian),
                        "neighbor": mode_or_nan(episode["risk_neighbor"]),
                        "start_frame": int(episode["frame"].iloc[0]),
                        "end_frame": int(episode["frame"].iloc[-1]),
                        "start_time_s": float(episode["time"].iloc[0]),
                        "end_time_s": float(episode["time"].iloc[-1]),
                        "duration_s": float((len(episode) - 1) / FPS),
                        "x_m": float(peak["x_smooth"]),
                        "y_m": float(peak["y_smooth"]),
                        "radius_m": float(peak["radius"]),
                        "minimum_radius_m": float(episode["radius"].min()),
                        "min_speed_m_s": float(episode["speed"].min()),
                        "max_speed_m_s": float(episode["speed"].max()),
                        "speed_change_m_s": float(episode["speed"].iloc[-1] - episode["speed"].iloc[0]),
                        "max_deceleration_m_s2": float(episode["longitudinal_acceleration"].min()),
                        "max_abs_turn_rate_deg_s": float(episode["turn_rate_deg_s"].abs().max()),
                        "net_heading_change_deg": float(
                            np.degrees(
                                wrap_angle(
                                    np.array([episode["heading"].iloc[-1] - episode["heading"].iloc[0]])
                                )[0]
                            )
                        ),
                        "nearest_distance_m": float(episode["nearest_distance"].min()),
                        "predicted_separation_m": float(finite_sep.min()) if finite_sep.notna().any() else math.nan,
                        "time_to_cpa_s": float(finite_ttc.min()) if finite_ttc.notna().any() else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def pedestrian_summary(data: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for pedestrian, track in data.groupby("pedestrian", sort=True):
        track = track.copy().sort_values("frame").reset_index(drop=True)
        depart_candidates = np.flatnonzero(track["distance_to_start"].to_numpy() >= 0.50)
        depart = int(depart_candidates[0]) if len(depart_candidates) else 0
        arrival_candidates = np.flatnonzero(
            (np.arange(len(track)) > depart) & (track["distance_to_goal"].to_numpy() <= 0.50)
        )
        arrival = int(arrival_candidates[0]) if len(arrival_candidates) else len(track) - 1
        active = track.iloc[depart : arrival + 1]
        points = active[["x_smooth", "y_smooth"]].to_numpy(float)
        route_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        straight_distance = float(np.linalg.norm(points[-1] - points[0]))
        central = track[(track["progress"] >= 0.15) & (track["progress"] <= 0.85)]
        signed_detour = float(central["right_lateral_deviation"].median())
        if signed_detour > 0.15:
            side = "right"
        elif signed_detour < -0.15:
            side = "left"
        else:
            side = "central"
        ped_events = events[events["pedestrian"] == pedestrian]
        center_samples = track["radius"] <= CENTER_RADIUS_M
        approach_samples = (track["radius"] >= 5.0) & (track["radius"] <= 8.0)
        rows.append(
            {
                "pedestrian": int(pedestrian),
                "departure_time_s": float(track["time"].iloc[depart]),
                "arrival_time_s": float(track["time"].iloc[arrival]),
                "travel_time_s": float(track["time"].iloc[arrival] - track["time"].iloc[depart]),
                "route_length_m": route_length,
                "straight_distance_m": straight_distance,
                "detour_ratio": route_length / straight_distance,
                "route_side": side,
                "signed_right_deviation_m": signed_detour,
                "peak_abs_deviation_m": float(central["right_lateral_deviation"].abs().max()),
                "mean_speed_m_s": float(active["speed"].mean()),
                "min_speed_m_s": float(active["speed"].min()),
                "center_median_speed_m_s": float(track.loc[track["radius"] <= CENTER_RADIUS_M, "speed"].median()),
                "approach_median_speed_m_s": float(
                    track.loc[approach_samples, "speed"].median()
                ),
                "center_conflict_rate": float(track.loc[center_samples, "conflict"].mean()),
                "approach_conflict_rate": float(track.loc[approach_samples, "conflict"].mean()),
                "center_avoidance_rate": float(track.loc[center_samples, "avoidance"].mean()),
                "approach_avoidance_rate": float(track.loc[approach_samples, "avoidance"].mean()),
                "conflict_events": int((ped_events["event_type"] == "conflict").sum()),
                "deceleration_events": int((ped_events["event_type"] == "deceleration").sum()),
                "turning_events": int((ped_events["event_type"] == "turning").sum()),
                "avoidance_events": int((ped_events["event_type"] == "avoidance").sum()),
            }
        )
    return pd.DataFrame(rows)


def event_summary(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby("event_type", as_index=False)
        .agg(
            events=("event_id", "size"),
            pedestrians=("pedestrian", "nunique"),
            median_duration_s=("duration_s", "median"),
            median_radius_m=("radius_m", "median"),
            median_min_speed_m_s=("min_speed_m_s", "median"),
            median_nearest_distance_m=("nearest_distance_m", "median"),
        )
        .sort_values("event_type")
    )


def plot_trajectory_speed(data: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    all_speeds = data["speed"].to_numpy(float)
    norm = plt.Normalize(0, min(2.2, float(np.quantile(all_speeds, 0.99))))
    collection = None
    for _, track in data.groupby("pedestrian", sort=True):
        xy = track[["x_smooth", "y_smooth"]].to_numpy(float)
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        collection = LineCollection(segments, cmap="viridis", norm=norm, linewidth=1.15, alpha=0.88)
        collection.set_array(track["speed"].to_numpy(float)[:-1])
        ax.add_collection(collection)
        ax.scatter(xy[0, 0], xy[0, 1], s=7, color="#202020", alpha=0.55, zorder=3)
    boundary = plt.Circle((0, 0), 10, fill=False, ls="--", lw=1.2, color="#555555")
    center = plt.Circle((0, 0), CENTER_RADIUS_M, color="#D55E00", alpha=0.07, lw=0)
    ax.add_patch(boundary)
    ax.add_patch(center)
    if collection is not None:
        cbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("speed (m/s)")
    ax.set(
        title="Circle-antipode trajectories coloured by speed",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(-11, 11),
        ylim=(-11, 11),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.text(-10.6, -10.6, "dots: starts   orange disk: r ≤ 2 m", fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_speed_changes(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5))
    ax = axes[0]
    for _, track in data.groupby("pedestrian", sort=True):
        ax.plot(track["time"], track["speed"], color="#808080", lw=0.45, alpha=0.18)
    time_stats = data.groupby("time")["speed"].quantile([0.25, 0.5, 0.75]).unstack()
    ax.fill_between(time_stats.index, time_stats[0.25], time_stats[0.75], color="#0072B2", alpha=0.22, label="IQR")
    ax.plot(time_stats.index, time_stats[0.5], color="#0072B2", lw=2.0, label="median")
    ax.set(title="Speed changes over time", xlabel="time (s)", ylabel="speed (m/s)", ylim=(0, 2.6))
    ax.legend(frameon=False)

    ax = axes[1]
    bins = np.arange(0, 10.51, 0.5)
    temp = data.copy()
    temp["radius_bin"] = pd.cut(temp["radius"], bins=bins, include_lowest=True)
    radial = temp.groupby("radius_bin", observed=True)["speed"].quantile([0.25, 0.5, 0.75]).unstack()
    centers = np.array([interval.mid for interval in radial.index])
    ax.fill_between(centers, radial[0.25], radial[0.75], color="#009E73", alpha=0.22, label="IQR")
    ax.plot(centers, radial[0.5], color="#009E73", lw=2.0, marker="o", ms=3, label="median")
    ax.axvspan(0, CENTER_RADIUS_M, color="#D55E00", alpha=0.08, label="central conflict zone")
    ax.set(title="Speed by distance from centre", xlabel="radius (m)", ylabel="speed (m/s)", xlim=(0, 10.5), ylim=(0, 2.1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_behavior_events(data: pd.DataFrame, events: pd.DataFrame, output: Path) -> None:
    colors = {
        "conflict": "#D55E00",
        "deceleration": "#E69F00",
        "turning": "#0072B2",
        "avoidance": "#009E73",
    }
    labels = {
        "conflict": "Conflict",
        "deceleration": "Deceleration",
        "turning": "Turning",
        "avoidance": "Avoidance",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.0), sharex=True, sharey=True)
    for ax, event_type in zip(axes.ravel(), colors):
        for _, track in data.groupby("pedestrian", sort=True):
            ax.plot(track["x_smooth"], track["y_smooth"], color="#BEBEBE", lw=0.35, alpha=0.22)
        selected = events[events["event_type"] == event_type]
        size = np.clip(selected["duration_s"].to_numpy(float) * 30 + 12, 12, 90)
        ax.scatter(selected["x_m"], selected["y_m"], s=size, color=colors[event_type], alpha=0.68, edgecolor="white", linewidth=0.35)
        ax.add_patch(plt.Circle((0, 0), 10, fill=False, ls="--", lw=0.8, color="#666666"))
        ax.add_patch(plt.Circle((0, 0), CENTER_RADIUS_M, fill=False, ls=":", lw=1.0, color="#D55E00"))
        ax.set_title(f"{labels[event_type]} episodes (n={len(selected)})")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle("Detected behaviour episodes; marker size = duration", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_findings(data: pd.DataFrame, pedestrians: pd.DataFrame, events: pd.DataFrame) -> dict[str, object]:
    classified = pedestrians[pedestrians["route_side"] != "central"]
    right = int((classified["route_side"] == "right").sum())
    left = int((classified["route_side"] == "left").sum())
    side_test = binomtest(right, right + left, 0.5, alternative="greater") if right + left else None

    paired = pedestrians[["center_median_speed_m_s", "approach_median_speed_m_s"]].dropna()
    speed_test = wilcoxon(
        paired["center_median_speed_m_s"],
        paired["approach_median_speed_m_s"],
        alternative="less",
    )
    conflict_paired = pedestrians[["center_conflict_rate", "approach_conflict_rate"]].dropna()
    conflict_test = wilcoxon(
        conflict_paired["center_conflict_rate"],
        conflict_paired["approach_conflict_rate"],
        alternative="greater",
        method="approx",
    )
    avoidance_paired = pedestrians[["center_avoidance_rate", "approach_avoidance_rate"]].dropna()
    avoidance_test = wilcoxon(
        avoidance_paired["center_avoidance_rate"],
        avoidance_paired["approach_avoidance_rate"],
        alternative="greater",
        method="approx",
    )
    center_events = events.assign(in_center=events["radius_m"] <= CENTER_RADIUS_M)
    event_center_share = center_events.groupby("event_type")["in_center"].mean().to_dict()
    return {
        "samples": int(len(data)),
        "pedestrians": int(data["pedestrian"].nunique()),
        "frames": int(data["frame"].nunique()),
        "duration_s": float(data["time"].max() - data["time"].min()),
        "median_speed_m_s": float(data["speed"].median()),
        "right_routes": right,
        "left_routes": left,
        "central_routes": int((pedestrians["route_side"] == "central").sum()),
        "right_share_excluding_central": right / (right + left) if right + left else math.nan,
        "right_preference_binomial_p": float(side_test.pvalue) if side_test else math.nan,
        "center_median_speed_m_s": float(paired["center_median_speed_m_s"].median()),
        "approach_median_speed_m_s": float(paired["approach_median_speed_m_s"].median()),
        "paired_speed_pedestrians": int(len(paired)),
        "paired_speed_wilcoxon_p": float(speed_test.pvalue),
        "center_conflict_rate": float(conflict_paired["center_conflict_rate"].median()),
        "approach_conflict_rate": float(conflict_paired["approach_conflict_rate"].median()),
        "paired_conflict_wilcoxon_p": float(conflict_test.pvalue),
        "center_avoidance_rate": float(avoidance_paired["center_avoidance_rate"].median()),
        "approach_avoidance_rate": float(avoidance_paired["approach_avoidance_rate"].median()),
        "paired_avoidance_wilcoxon_p": float(avoidance_test.pvalue),
        "event_counts": {str(k): int(v) for k, v in events["event_type"].value_counts().sort_index().items()},
        "event_center_share": {str(k): float(v) for k, v in event_center_share.items()},
        "thresholds": {
            "fps": FPS,
            "smoothing_window_frames": SMOOTH_WINDOW,
            "body_diameter_m": BODY_DIAMETER_M,
            "conflict_predicted_separation_m": CONFLICT_SEPARATION_M,
            "conflict_horizon_s": CONFLICT_HORIZON_S,
            "deceleration_m_s2": -0.60,
            "turn_rate_deg_s": 35.0,
            "center_radius_m": CENTER_RADIUS_M,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir.parent / "datasets" / "circle-10m-64-1.txt",
    )
    parser.add_argument("--output", type=Path, default=script_dir / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    configure_plotting()
    trajectories = load_trajectories(args.input.resolve())
    trajectories = add_kinematics(trajectories)
    trajectories = add_interaction_risk(trajectories)
    trajectories = add_behavior_flags(trajectories)
    events = extract_episodes(trajectories)
    pedestrians = pedestrian_summary(trajectories, events)
    summary = event_summary(events)
    findings = compute_findings(trajectories, pedestrians, events)

    export_columns = [
        "pedestrian", "frame", "time", "x", "y", "x_smooth", "y_smooth",
        "vx", "vy", "speed", "longitudinal_acceleration", "heading",
        "turn_rate_deg_s", "radius", "progress", "right_lateral_deviation",
        "nearest_neighbor", "risk_neighbor", "nearest_distance",
        "predicted_separation", "time_to_cpa", "conflict", "deceleration",
        "turning", "avoidance",
    ]
    trajectories[export_columns].to_csv(args.output / "trajectory_kinematics.csv", index=False, float_format="%.6f")
    events.to_csv(args.output / "behavior_events.csv", index=False, float_format="%.6f")
    pedestrians.to_csv(args.output / "pedestrian_summary.csv", index=False, float_format="%.6f")
    summary.to_csv(args.output / "event_summary.csv", index=False, float_format="%.6f")
    (args.output / "analysis_summary.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_trajectory_speed(trajectories, args.output / "trajectory_speed.png")
    plot_speed_changes(trajectories, args.output / "speed_changes.png")
    plot_behavior_events(trajectories, events, args.output / "behavior_events.png")

    print(json.dumps(findings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
