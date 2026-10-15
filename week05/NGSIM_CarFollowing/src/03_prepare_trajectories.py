"""STEP 3: align, optionally smooth, and initialize the selected trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from common import (
    PAIR_COLUMNS,
    configure_logging,
    default_config_path,
    ensure_output_dir,
    load_config,
    read_ngsim_csv,
    require_config_section,
    write_csv_atomic,
    write_json_atomic,
)


LOGGER = configure_logging("prepare_trajectories")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare aligned real leader/follower trajectories and initial state."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(__file__),
        help="YAML configuration path (default: project config.yaml)",
    )
    return parser.parse_args()


def _validate_window(
    selected: pd.DataFrame,
    selected_case: dict[str, Any],
    sampling_interval_s: float,
    tolerance_s: float,
) -> pd.DataFrame:
    missing = sorted(set(PAIR_COLUMNS).difference(selected.columns))
    if missing:
        raise ValueError(f"selected_case_raw.csv is missing columns: {missing}")
    ordered = selected.sort_values("frame_id", kind="mergesort").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("selected_case_raw.csv is empty")
    if ordered["frame_id"].duplicated().any():
        raise ValueError("selected case contains duplicate frames")
    expected_frames = np.arange(
        int(ordered["frame_id"].iloc[0]),
        int(ordered["frame_id"].iloc[-1]) + 1,
        dtype=np.int64,
    )
    if not np.array_equal(ordered["frame_id"].to_numpy(dtype=np.int64), expected_frames):
        raise ValueError("selected case contains missing frames")
    if len(ordered) > 1 and not np.allclose(
        np.diff(ordered["time_s"].to_numpy(dtype=float)),
        sampling_interval_s,
        atol=tolerance_s,
        rtol=0,
    ):
        raise ValueError("selected case is not sampled continuously at the configured interval")
    if ordered["lane_id"].nunique() != 1:
        raise ValueError("selected vehicles do not remain in one lane")
    if ordered[["leader_id", "follower_id"]].nunique().max() != 1:
        raise ValueError("selected case changes leader or follower")
    if not ordered["gap_m"].gt(0).all():
        raise ValueError("selected raw case contains a nonpositive net gap")

    expected_values = {
        "leader_id": int(ordered["leader_id"].iloc[0]),
        "follower_id": int(ordered["follower_id"].iloc[0]),
        "lane_id": int(ordered["lane_id"].iloc[0]),
        "start_frame": int(ordered["frame_id"].iloc[0]),
        "end_frame": int(ordered["frame_id"].iloc[-1]),
    }
    mismatches = {
        key: (selected_case.get(key), value)
        for key, value in expected_values.items()
        if int(selected_case.get(key, -1)) != value
    }
    if mismatches:
        raise ValueError(
            f"selected_case.json and selected_case_raw.csv disagree: {mismatches}"
        )
    return ordered


def _standardized_raw(selected: pd.DataFrame) -> pd.DataFrame:
    relative_time = selected["time_s"] - selected["time_s"].iloc[0]
    raw = pd.DataFrame(
        {
            "t": relative_time,
            "frame_id": selected["frame_id"],
            "time_s": selected["time_s"],
            "global_time_ms": selected["global_time_ms"],
            "leader_id": selected["leader_id"],
            "follower_id": selected["follower_id"],
            "lane_id": selected["lane_id"],
            "leader_local_x_m": selected["leader_local_x_m"],
            "follower_local_x_m": selected["follower_local_x_m"],
            "leader_x": selected["leader_y"],
            "leader_v": selected["leader_speed_m_s"],
            "leader_a": selected["leader_acceleration_m_s2"],
            "follower_x_true": selected["follower_y"],
            "follower_v_true": selected["follower_speed_m_s"],
            "follower_a_true": selected["follower_acceleration_m_s2"],
            "leader_length": selected["leader_length"],
        }
    )
    raw["gap_true"] = (
        raw["leader_x"] - raw["follower_x_true"] - raw["leader_length"]
    )
    return raw


def _load_vehicle_context(
    si_csv_path: Path,
    selected: pd.DataFrame,
    padding_frames: int,
) -> pd.DataFrame:
    """Load both individual tracks around the window to avoid SG edge artifacts."""

    leader_id = int(selected["leader_id"].iloc[0])
    follower_id = int(selected["follower_id"].iloc[0])
    start_frame = int(selected["frame_id"].iloc[0]) - padding_frames
    end_frame = int(selected["frame_id"].iloc[-1]) + padding_frames
    observations = read_ngsim_csv(si_csv_path)
    context = observations.loc[
        observations["vehicle_id"].isin([leader_id, follower_id])
        & observations["frame_id"].between(start_frame, end_frame)
    ].copy()
    leader = context.loc[context["vehicle_id"].eq(leader_id)].copy()
    follower = context.loc[context["vehicle_id"].eq(follower_id)].copy()
    merged = leader.merge(
        follower,
        on="frame_id",
        how="inner",
        suffixes=("_leader", "_follower"),
        validate="one_to_one",
    ).sort_values("frame_id", kind="mergesort")
    if merged.empty:
        raise ValueError("No aligned leader/follower samples are available for smoothing")
    if not np.allclose(
        merged["time_s_leader"], merged["time_s_follower"], atol=1e-9, rtol=0
    ) or not np.array_equal(
        merged["global_time_ms_leader"].to_numpy(),
        merged["global_time_ms_follower"].to_numpy(),
    ):
        raise ValueError("Leader and follower source timestamps do not align")

    pair_like = pd.DataFrame(
        {
            "frame_id": merged["frame_id"],
            "time_s": merged["time_s_leader"],
            "global_time_ms": merged["global_time_ms_leader"],
            "lane_id": merged["lane_id_follower"],
            "leader_id": merged["vehicle_id_leader"],
            "follower_id": merged["vehicle_id_follower"],
            "leader_y": merged["local_y_m_leader"],
            "follower_y": merged["local_y_m_follower"],
            "leader_local_x_m": merged["local_x_m_leader"],
            "follower_local_x_m": merged["local_x_m_follower"],
            "leader_length": merged["vehicle_length_m_leader"],
            "follower_length": merged["vehicle_length_m_follower"],
            "gap_m": (
                merged["local_y_m_leader"]
                - merged["local_y_m_follower"]
                - merged["vehicle_length_m_leader"]
            ),
            "leader_speed_m_s": merged["speed_m_s_leader"],
            "follower_speed_m_s": merged["speed_m_s_follower"],
            "relative_speed_m_s": (
                merged["speed_m_s_leader"] - merged["speed_m_s_follower"]
            ),
            "leader_acceleration_m_s2": merged["acceleration_m_s2_leader"],
            "follower_acceleration_m_s2": merged["acceleration_m_s2_follower"],
            "relative_acceleration_m_s2": (
                merged["acceleration_m_s2_leader"]
                - merged["acceleration_m_s2_follower"]
            ),
            "leader_class": merged["vehicle_class_leader"],
            "follower_class": merged["vehicle_class_follower"],
        }
    )
    return pair_like.reset_index(drop=True)


def _prepare_clean(
    raw: pd.DataFrame,
    preparation: dict[str, Any],
    sampling_interval_s: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean = raw.copy()
    smoothing_enabled = bool(preparation.get("smoothing_enabled", False))
    use_smoothed_kinematics = bool(
        preparation.get("use_smoothed_kinematics", smoothing_enabled)
    )
    window_length = int(preparation.get("savgol_window_length", 11))
    polyorder = int(preparation.get("savgol_polyorder", 3))

    if smoothing_enabled:
        if window_length <= 0 or window_length % 2 == 0:
            raise ValueError("preparation.savgol_window_length must be a positive odd integer")
        if polyorder < 0 or polyorder >= window_length:
            raise ValueError(
                "preparation.savgol_polyorder must satisfy 0 <= polyorder < window_length"
            )
        if window_length > len(raw):
            raise ValueError(
                "Savitzky-Golay window is longer than the selected trajectory; "
                "change the configuration explicitly"
            )
        if use_smoothed_kinematics and polyorder < 2:
            raise ValueError(
                "Savitzky-Golay polyorder must be at least 2 to derive acceleration"
            )

        leader_position = raw["leader_x"].to_numpy(dtype=float)
        follower_position = raw["follower_x_true"].to_numpy(dtype=float)
        clean["leader_x"] = savgol_filter(
            leader_position, window_length, polyorder, deriv=0, mode="interp"
        )
        clean["follower_x_true"] = savgol_filter(
            follower_position, window_length, polyorder, deriv=0, mode="interp"
        )
        if use_smoothed_kinematics:
            clean["leader_v"] = savgol_filter(
                leader_position,
                window_length,
                polyorder,
                deriv=1,
                delta=sampling_interval_s,
                mode="interp",
            )
            clean["leader_a"] = savgol_filter(
                leader_position,
                window_length,
                polyorder,
                deriv=2,
                delta=sampling_interval_s,
                mode="interp",
            )
            clean["follower_v_true"] = savgol_filter(
                follower_position,
                window_length,
                polyorder,
                deriv=1,
                delta=sampling_interval_s,
                mode="interp",
            )
            clean["follower_a_true"] = savgol_filter(
                follower_position,
                window_length,
                polyorder,
                deriv=2,
                delta=sampling_interval_s,
                mode="interp",
            )

    clean["gap_true"] = (
        clean["leader_x"] - clean["follower_x_true"] - clean["leader_length"]
    )
    numeric = [
        "t",
        "leader_x",
        "leader_v",
        "leader_a",
        "follower_x_true",
        "follower_v_true",
        "follower_a_true",
        "leader_length",
        "gap_true",
    ]
    if not np.isfinite(clean[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Prepared trajectory contains nonfinite values")
    processing = {
        "smoothing_enabled": smoothing_enabled,
        "method": "Savitzky-Golay position smoothing" if smoothing_enabled else "none",
        "savgol_window_length": window_length if smoothing_enabled else None,
        "savgol_polyorder": polyorder if smoothing_enabled else None,
        "use_smoothed_kinematics": use_smoothed_kinematics if smoothing_enabled else False,
        "kinematics_method": (
            "first and second Savitzky-Golay derivatives of smoothed longitudinal position"
            if smoothing_enabled and use_smoothed_kinematics
            else "reported NGSIM speed and acceleration retained"
        ),
        "sampling_interval_s": sampling_interval_s,
        "raw_values_preserved_in": "trajectory_raw.csv",
        "no_clipping_or_imputation": True,
    }
    return clean, processing


def _vehicle_initial_state(
    raw: pd.Series,
    clean: pd.Series,
    *,
    role: str,
) -> dict[str, Any]:
    is_leader = role == "leader"
    identifier = int(raw["leader_id"] if is_leader else raw["follower_id"])
    lateral = float(
        raw["leader_local_x_m"] if is_leader else raw["follower_local_x_m"]
    )
    raw_position = float(raw["leader_x"] if is_leader else raw["follower_x_true"])
    raw_speed = float(raw["leader_v"] if is_leader else raw["follower_v_true"])
    raw_acceleration = float(raw["leader_a"] if is_leader else raw["follower_a_true"])
    prepared_position = float(
        clean["leader_x"] if is_leader else clean["follower_x_true"]
    )
    prepared_speed = float(
        clean["leader_v"] if is_leader else clean["follower_v_true"]
    )
    prepared_acceleration = float(
        clean["leader_a"] if is_leader else clean["follower_a_true"]
    )
    vehicle_length = (
        float(raw["leader_length"])
        if is_leader
        else None
    )
    # ``initial_*`` is the state that actually initialises the simulations and
    # matches the first row of trajectory_clean.csv.  It therefore carries the
    # prepared values, which is what the relative block and the reported initial
    # net gap are computed from.  The untouched source-frame observation is kept
    # under ``raw_*`` and in the source-named fields below.
    state = {
        "vehicle_id": identifier,
        "frame_id": int(raw["frame_id"]),
        "time_s": float(raw["time_s"]),
        "initial_time_s": float(raw["time_s"]),
        "local_x_m": lateral,
        "local_y_m": raw_position,
        "speed_m_s": raw_speed,
        "acceleration_m_s2": raw_acceleration,
        "vehicle_length_m": vehicle_length,
        "lane_id": int(raw["lane_id"]),
        "initial_lateral_position_m": lateral,
        "initial_longitudinal_position_m": prepared_position,
        "initial_speed_m_s": prepared_speed,
        "initial_acceleration_m_s2": prepared_acceleration,
        "initial_state_basis": (
            "prepared trajectory (trajectory_clean.csv first row); "
            "source-frame observation retained in raw_* and in "
            "local_y_m/speed_m_s/acceleration_m_s2"
        ),
        "source_frame_longitudinal_position_m": raw_position,
        "source_frame_speed_m_s": raw_speed,
        "source_frame_acceleration_m_s2": raw_acceleration,
        "raw_longitudinal_position_m": raw_position,
        "raw_speed_m_s": raw_speed,
        "raw_acceleration_m_s2": raw_acceleration,
        "prepared_longitudinal_position_m": prepared_position,
        "prepared_speed_m_s": prepared_speed,
        "prepared_acceleration_m_s2": prepared_acceleration,
    }
    return state


def _save_comparison(raw: pd.DataFrame, clean: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    specifications = [
        ("leader_x", "Leader position", "Position (m)"),
        ("follower_x_true", "Follower position", "Position (m)"),
        ("leader_v", "Leader speed", "Speed (m/s)"),
        ("follower_v_true", "Follower speed", "Speed (m/s)"),
        ("leader_a", "Leader acceleration", "Acceleration (m/s²)"),
        ("follower_a_true", "Follower acceleration", "Acceleration (m/s²)"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel(), specifications):
        axis.plot(raw["t"], raw[column], label="Raw", linewidth=1.2, alpha=0.8)
        axis.plot(clean["t"], clean[column], label="Prepared", linewidth=1.3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Time from case start (s)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["_paths"]
    data_config = require_config_section(config, "data")
    preparation = dict(require_config_section(config, "preparation"))
    output_dir = ensure_output_dir(config, "prepared_dir")
    cases_dir = Path(paths["cases_dir"])
    raw_case_path = cases_dir / "selected_case_raw.csv"
    case_json_path = cases_dir / "selected_case.json"
    if not raw_case_path.is_file() or not case_json_path.is_file():
        raise FileNotFoundError(
            "Selected-case artifacts are missing; run 02_select_following_case.py first"
        )
    with case_json_path.open("r", encoding="utf-8") as handle:
        selected_case = json.load(handle)
    selected = pd.read_csv(raw_case_path)
    dt = float(data_config.get("sampling_interval_s", 0.1))
    tolerance = float(data_config.get("time_grid_tolerance_s", 1e-6))
    selected = _validate_window(selected, selected_case, dt, tolerance)

    raw = _standardized_raw(selected)
    smoothing_enabled = bool(preparation.get("smoothing_enabled", False))
    configured_window = int(preparation.get("savgol_window_length", 11))
    padding_frames = configured_window // 2 if smoothing_enabled else 0
    if smoothing_enabled:
        context_pairs = _load_vehicle_context(
            Path(paths["si_csv"]), selected, padding_frames
        )
        context_raw = _standardized_raw(context_pairs)
        context_clean, processing = _prepare_clean(context_raw, preparation, dt)
        clean = context_clean.loc[
            context_clean["frame_id"].between(
                int(raw["frame_id"].iloc[0]), int(raw["frame_id"].iloc[-1])
            )
        ].reset_index(drop=True)
        if not np.array_equal(
            clean["frame_id"].to_numpy(dtype=np.int64),
            raw["frame_id"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("Smoothing context did not preserve every selected frame")
        # Keep the public case clock anchored at zero, independent of context padding.
        clean["t"] = raw["t"].to_numpy()
        processing["smoothing_context"] = {
            "source": "individual raw SI trajectories",
            "padding_frames_requested_each_side": padding_frames,
            "context_start_frame": int(context_raw["frame_id"].iloc[0]),
            "context_end_frame": int(context_raw["frame_id"].iloc[-1]),
            "selected_start_frame": int(raw["frame_id"].iloc[0]),
            "selected_end_frame": int(raw["frame_id"].iloc[-1]),
        }
    else:
        clean, processing = _prepare_clean(raw, preparation, dt)
    if not clean["gap_true"].gt(0).all():
        raise ValueError(
            "Configured smoothing produced a nonpositive net gap inside the selected "
            "window; no silent fallback was applied"
        )
    maximum_speed = float(data_config.get("maximum_plausible_speed_m_s", 60.0))
    maximum_abs_acceleration = float(
        data_config.get("maximum_plausible_abs_acceleration_m_s2", 15.0)
    )
    minimum_prepared_speed = float(
        min(clean["leader_v"].min(), clean["follower_v_true"].min())
    )
    maximum_prepared_speed = float(
        max(clean["leader_v"].max(), clean["follower_v_true"].max())
    )
    maximum_prepared_abs_acceleration = float(
        max(clean["leader_a"].abs().max(), clean["follower_a_true"].abs().max())
    )
    if minimum_prepared_speed < 0:
        raise ValueError(
            f"Prepared trajectory contains negative speed ({minimum_prepared_speed:.6f} "
            "m/s); no clipping or fallback was applied"
        )
    if maximum_prepared_speed > maximum_speed:
        raise ValueError(
            f"Prepared trajectory speed {maximum_prepared_speed:.6f} m/s exceeds the "
            f"configured plausibility limit {maximum_speed:.6f} m/s"
        )
    if maximum_prepared_abs_acceleration > maximum_abs_acceleration:
        raise ValueError(
            "Prepared trajectory acceleration magnitude "
            f"{maximum_prepared_abs_acceleration:.6f} m/s^2 exceeds the configured "
            f"plausibility limit {maximum_abs_acceleration:.6f} m/s^2"
        )
    write_csv_atomic(raw, output_dir / "trajectory_raw.csv")
    write_csv_atomic(clean, output_dir / "trajectory_clean.csv")

    raw_first = raw.iloc[0]
    clean_first = clean.iloc[0]
    leader = _vehicle_initial_state(raw_first, clean_first, role="leader")
    follower = _vehicle_initial_state(raw_first, clean_first, role="follower")
    follower_length = float(selected["follower_length"].iloc[0])
    follower["vehicle_length_m"] = follower_length
    initial_conditions = {
        "leader": leader,
        "follower": follower,
        "relative": {
            "initial_gap_m": float(clean_first["gap_true"]),
            "initial_relative_speed_m_s": float(
                clean_first["leader_v"] - clean_first["follower_v_true"]
            ),
            "initial_relative_acceleration_m_s2": float(
                clean_first["leader_a"] - clean_first["follower_a_true"]
            ),
            "raw_initial_gap_m": float(raw_first["gap_true"]),
            "raw_initial_relative_speed_m_s": float(
                raw_first["leader_v"] - raw_first["follower_v_true"]
            ),
            "raw_initial_relative_acceleration_m_s2": float(
                raw_first["leader_a"] - raw_first["follower_a_true"]
            ),
        },
        "processing": processing,
        "model_initialization_note": (
            "The observed follower acceleration is retained for comparison only. OVM, "
            "FVDM, and IDM must compute their initial accelerations from their equations."
        ),
    }
    write_json_atomic(initial_conditions, output_dir / "initial_conditions.json")
    _save_comparison(raw, clean, output_dir / "raw_vs_clean.png")

    # Verify -- rather than assume -- that the documented initial state really is
    # the first row of the prepared trajectory and that the reported relative
    # quantities follow from the documented definitions.
    initial_state_checks = {
        "leader_position_matches_trajectory_clean": bool(
            np.isclose(
                leader["initial_longitudinal_position_m"], float(clean_first["leader_x"]), atol=0.0
            )
        ),
        "leader_speed_matches_trajectory_clean": bool(
            np.isclose(leader["initial_speed_m_s"], float(clean_first["leader_v"]), atol=0.0)
        ),
        "leader_acceleration_matches_trajectory_clean": bool(
            np.isclose(leader["initial_acceleration_m_s2"], float(clean_first["leader_a"]), atol=0.0)
        ),
        "follower_position_matches_trajectory_clean": bool(
            np.isclose(
                follower["initial_longitudinal_position_m"],
                float(clean_first["follower_x_true"]),
                atol=0.0,
            )
        ),
        "follower_speed_matches_trajectory_clean": bool(
            np.isclose(
                follower["initial_speed_m_s"], float(clean_first["follower_v_true"]), atol=0.0
            )
        ),
        "follower_acceleration_matches_trajectory_clean": bool(
            np.isclose(
                follower["initial_acceleration_m_s2"],
                float(clean_first["follower_a_true"]),
                atol=0.0,
            )
        ),
        "initial_gap_equals_positions_and_leader_length": bool(
            np.isclose(
                initial_conditions["relative"]["initial_gap_m"],
                leader["initial_longitudinal_position_m"]
                - follower["initial_longitudinal_position_m"]
                - float(clean_first["leader_length"]),
                atol=1e-9,
            )
        ),
        "initial_relative_speed_equals_position_derived_speeds": bool(
            np.isclose(
                initial_conditions["relative"]["initial_relative_speed_m_s"],
                leader["initial_speed_m_s"] - follower["initial_speed_m_s"],
                atol=1e-9,
            )
        ),
    }
    if not all(initial_state_checks.values()):
        raise ValueError(
            "initial state is inconsistent with trajectory_clean.csv: "
            + ", ".join(sorted(key for key, ok in initial_state_checks.items() if not ok))
        )

    summary = {
        "status": "prepared",
        "leader_id": leader["vehicle_id"],
        "follower_id": follower["vehicle_id"],
        "lane_id": leader["lane_id"],
        "start_frame": int(clean["frame_id"].iloc[0]),
        "end_frame": int(clean["frame_id"].iloc[-1]),
        "duration_s": float(clean["t"].iloc[-1]),
        "frame_count": int(len(clean)),
        "processing": processing,
        "raw_gap_range_m": [float(raw["gap_true"].min()), float(raw["gap_true"].max())],
        "prepared_gap_range_m": [
            float(clean["gap_true"].min()),
            float(clean["gap_true"].max()),
        ],
        "prepared_physical_checks": {
            "minimum_speed_m_s": minimum_prepared_speed,
            "maximum_speed_m_s": maximum_prepared_speed,
            "maximum_abs_acceleration_m_s2": maximum_prepared_abs_acceleration,
            "configured_maximum_speed_m_s": maximum_speed,
            "configured_maximum_abs_acceleration_m_s2": maximum_abs_acceleration,
            "passed": True,
        },
        "alignment_checks": {
            "one_row_per_frame": True,
            "continuous_0.1s_grid": True,
            "common_leader_follower_time_index": True,
            "prepared_initial_state_matches_trajectory_clean_first_row": True,
            "initial_state_consistency": initial_state_checks,
        },
    }
    write_json_atomic(summary, output_dir / "preparation_summary.json")
    LOGGER.info(
        "Prepared %d aligned frames for L=%d F=%d; initial prepared gap %.3f m",
        len(clean),
        leader["vehicle_id"],
        follower["vehicle_id"],
        initial_conditions["relative"]["initial_gap_m"],
    )
    return initial_conditions


def main() -> None:
    args = parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
