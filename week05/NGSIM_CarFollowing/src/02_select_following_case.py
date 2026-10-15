"""STEP 2: rank continuous passenger-car following events and select one case."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from common import (
    PAIR_COLUMNS,
    add_continuous_segment_ids,
    configure_logging,
    default_config_path,
    ensure_output_dir,
    load_config,
    require_config_section,
    write_csv_atomic,
    write_json_atomic,
)


LOGGER = configure_logging("select_following_case")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank stable NGSIM following windows and select one real case."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(__file__),
        help="YAML configuration path (default: project config.yaml)",
    )
    return parser.parse_args()


def _first_sustained_index(mask: np.ndarray, count: int) -> int | None:
    if count <= 1:
        matches = np.flatnonzero(mask)
        return int(matches[0]) if matches.size else None
    if mask.size < count:
        return None
    runs = np.convolve(mask.astype(np.int8), np.ones(count, dtype=np.int8), mode="valid")
    matches = np.flatnonzero(runs == count)
    return int(matches[0]) if matches.size else None


def _window_metrics(
    window: pd.DataFrame,
    *,
    sampling_interval_s: float,
    braking_threshold_m_s2: float,
    braking_consecutive_frames: int = 3,
    response_search_window_s: float = 8.0,
) -> dict[str, Any]:
    ordered = window.sort_values("frame_id", kind="mergesort")
    leader_acceleration = ordered["leader_acceleration_m_s2"].to_numpy(dtype=float)
    follower_acceleration = ordered["follower_acceleration_m_s2"].to_numpy(dtype=float)
    leader_event = _first_sustained_index(
        leader_acceleration <= braking_threshold_m_s2,
        braking_consecutive_frames,
    )
    response_lag_s: float | None = None
    if leader_event is not None:
        search_samples = max(1, int(round(response_search_window_s / sampling_interval_s)))
        search_end = min(len(ordered), leader_event + search_samples + 1)
        follower_event_relative = _first_sustained_index(
            follower_acceleration[leader_event:search_end] <= braking_threshold_m_s2,
            braking_consecutive_frames,
        )
        if follower_event_relative is not None:
            response_lag_s = float(follower_event_relative * sampling_interval_s)

    gaps = ordered["gap_m"].to_numpy(dtype=float)
    leader_speed = ordered["leader_speed_m_s"].to_numpy(dtype=float)
    follower_speed = ordered["follower_speed_m_s"].to_numpy(dtype=float)
    leader_position_step_speed = np.diff(
        ordered["leader_y"].to_numpy(dtype=float)
    ) / sampling_interval_s
    follower_position_step_speed = np.diff(
        ordered["follower_y"].to_numpy(dtype=float)
    ) / sampling_interval_s
    leader_interval_speed = 0.5 * (leader_speed[:-1] + leader_speed[1:])
    follower_interval_speed = 0.5 * (follower_speed[:-1] + follower_speed[1:])
    leader_speed_position_rmse = float(
        np.sqrt(np.mean(np.square(leader_position_step_speed - leader_interval_speed)))
    )
    follower_speed_position_rmse = float(
        np.sqrt(np.mean(np.square(follower_position_step_speed - follower_interval_speed)))
    )
    kinematic_quality_score = math.exp(
        -(leader_speed_position_rmse + follower_speed_position_rmse) / 4.0
    )

    if {
        "leader_prepared_v_est",
        "leader_prepared_a_est",
        "follower_prepared_v_est",
        "follower_prepared_a_est",
    }.issubset(ordered.columns):
        estimated_leader_speed = ordered["leader_prepared_v_est"].to_numpy(dtype=float)
        estimated_leader_acceleration = ordered["leader_prepared_a_est"].to_numpy(
            dtype=float
        )
        estimated_follower_speed = ordered["follower_prepared_v_est"].to_numpy(
            dtype=float
        )
        estimated_follower_acceleration = ordered[
            "follower_prepared_a_est"
        ].to_numpy(dtype=float)
    else:
        estimated_leader_speed = leader_speed
        estimated_leader_acceleration = leader_acceleration
        estimated_follower_speed = follower_speed
        estimated_follower_acceleration = follower_acceleration
    gap_mean = float(np.mean(gaps))
    gap_cv = float(np.std(gaps, ddof=0) / gap_mean) if gap_mean > 0 else math.inf
    follower_deceleration_strength = max(0.0, -float(np.min(follower_acceleration)))
    response_strength = follower_deceleration_strength
    if response_lag_s is not None:
        response_strength *= math.exp(-response_lag_s / max(response_search_window_s, 1e-9))
    else:
        response_strength = 0.0

    return {
        "leader_id": int(ordered["leader_id"].iloc[0]),
        "follower_id": int(ordered["follower_id"].iloc[0]),
        "lane_id": int(ordered["lane_id"].iloc[0]),
        "start_frame": int(ordered["frame_id"].iloc[0]),
        "end_frame": int(ordered["frame_id"].iloc[-1]),
        "start_time_s": float(ordered["time_s"].iloc[0]),
        "end_time_s": float(ordered["time_s"].iloc[-1]),
        "duration_s": float(ordered["time_s"].iloc[-1] - ordered["time_s"].iloc[0]),
        "frame_count": int(len(ordered)),
        "leader_speed_range_m_s": float(
            ordered["leader_speed_m_s"].max() - ordered["leader_speed_m_s"].min()
        ),
        "leader_robust_speed_range_m_s": float(
            np.quantile(leader_speed, 0.95) - np.quantile(leader_speed, 0.05)
        ),
        "leader_prepared_speed_range_m_s": float(
            np.max(estimated_leader_speed) - np.min(estimated_leader_speed)
        ),
        "follower_speed_range_m_s": float(
            ordered["follower_speed_m_s"].max()
            - ordered["follower_speed_m_s"].min()
        ),
        "follower_robust_speed_range_m_s": float(
            np.quantile(follower_speed, 0.95) - np.quantile(follower_speed, 0.05)
        ),
        "leader_prepared_min_speed_m_s": float(np.min(estimated_leader_speed)),
        "leader_prepared_max_speed_m_s": float(np.max(estimated_leader_speed)),
        "follower_prepared_min_speed_m_s": float(np.min(estimated_follower_speed)),
        "follower_prepared_max_speed_m_s": float(np.max(estimated_follower_speed)),
        "leader_prepared_max_abs_acceleration_m_s2": float(
            np.max(np.abs(estimated_leader_acceleration))
        ),
        "follower_prepared_max_abs_acceleration_m_s2": float(
            np.max(np.abs(estimated_follower_acceleration))
        ),
        "leader_speed_position_rmse_m_s": leader_speed_position_rmse,
        "follower_speed_position_rmse_m_s": follower_speed_position_rmse,
        "kinematic_quality_score": float(kinematic_quality_score),
        "leader_negative_position_step_count": int(
            np.count_nonzero(leader_position_step_speed < -1e-9)
        ),
        "follower_negative_position_step_count": int(
            np.count_nonzero(follower_position_step_speed < -1e-9)
        ),
        "leader_min_acceleration_m_s2": float(np.min(leader_acceleration)),
        "follower_min_acceleration_m_s2": float(np.min(follower_acceleration)),
        "leader_braking_frame_count": int(
            np.count_nonzero(leader_acceleration <= braking_threshold_m_s2)
        ),
        "follower_braking_frame_count": int(
            np.count_nonzero(follower_acceleration <= braking_threshold_m_s2)
        ),
        "braking_response_lag_s": response_lag_s,
        "follower_response_strength": float(response_strength),
        "minimum_gap_m": float(np.min(gaps)),
        "maximum_gap_m": float(np.max(gaps)),
        "mean_gap_m": gap_mean,
        "gap_standard_deviation_m": float(np.std(gaps, ddof=0)),
        "gap_coefficient_of_variation": gap_cv,
    }


def _add_prepared_kinematic_estimates(
    segment: pd.DataFrame,
    preparation: Mapping[str, Any],
    sampling_interval_s: float,
) -> pd.DataFrame:
    """Estimate STEP 3 kinematics once per segment for safe window selection."""

    estimated = segment.copy()
    smoothing_enabled = bool(preparation.get("smoothing_enabled", False))
    use_smoothed_kinematics = bool(
        preparation.get("use_smoothed_kinematics", smoothing_enabled)
    )
    window_length = int(preparation.get("savgol_window_length", 11))
    polyorder = int(preparation.get("savgol_polyorder", 3))
    can_filter = (
        smoothing_enabled
        and use_smoothed_kinematics
        and window_length > 0
        and window_length % 2 == 1
        and polyorder >= 2
        and polyorder < window_length
        and len(estimated) >= window_length
    )
    if can_filter:
        for role, position_column in (
            ("leader", "leader_y"),
            ("follower", "follower_y"),
        ):
            position = estimated[position_column].to_numpy(dtype=float)
            estimated[f"{role}_prepared_v_est"] = savgol_filter(
                position,
                window_length,
                polyorder,
                deriv=1,
                delta=sampling_interval_s,
                mode="interp",
            )
            estimated[f"{role}_prepared_a_est"] = savgol_filter(
                position,
                window_length,
                polyorder,
                deriv=2,
                delta=sampling_interval_s,
                mode="interp",
            )
    else:
        estimated["leader_prepared_v_est"] = estimated["leader_speed_m_s"]
        estimated["leader_prepared_a_est"] = estimated["leader_acceleration_m_s2"]
        estimated["follower_prepared_v_est"] = estimated["follower_speed_m_s"]
        estimated["follower_prepared_a_est"] = estimated[
            "follower_acceleration_m_s2"
        ]
    return estimated


def _window_starts(length: int, minimum_count: int, maximum_count: int, step: int) -> list[int]:
    if length < minimum_count:
        return []
    if length <= maximum_count:
        return [0]
    final_start = length - maximum_count
    starts = list(range(0, final_start + 1, max(step, 1)))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _robust_unit_scale(values: pd.Series) -> pd.Series:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    denominator = float(finite.quantile(0.95))
    if denominator <= 1e-12:
        denominator = float(finite.max())
    if denominator <= 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values.fillna(0.0) / denominator).clip(lower=0.0, upper=1.0)


def _score_candidates(candidates: pd.DataFrame, selection: Mapping[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    scored = candidates.copy()
    maximum_duration = float(selection["maximum_duration_s"])
    scored["duration_score"] = (scored["duration_s"] / maximum_duration).clip(0, 1)
    scored["speed_range_score"] = _robust_unit_scale(
        scored["leader_prepared_speed_range_m_s"]
    )
    scored["deceleration_score"] = _robust_unit_scale(
        -scored["leader_min_acceleration_m_s2"]
    )
    scored["follower_response_score"] = _robust_unit_scale(
        scored["follower_response_strength"]
    )
    stable_gap = (1.0 - scored["gap_coefficient_of_variation"].clip(0, 1)).fillna(0)
    positive_margin = (
        scored["minimum_gap_m"] / max(float(selection["minimum_gap_m"]) * 4.0, 1.0)
    ).clip(0, 1)
    scored["gap_quality_score"] = 0.6 * stable_gap + 0.4 * positive_margin

    weights = selection.get("score_weights", {})
    component_weights = {
        "duration_score": float(weights.get("duration", 0.15)),
        "speed_range_score": float(weights.get("speed_range", 0.30)),
        "deceleration_score": float(weights.get("deceleration", 0.25)),
        "follower_response_score": float(weights.get("follower_response", 0.15)),
        "gap_quality_score": float(weights.get("gap_quality", 0.15)),
    }
    weight_sum = sum(component_weights.values())
    if weight_sum <= 0:
        raise ValueError("selection.score_weights must have a positive sum")
    scored["base_selection_score"] = sum(
        scored[column] * weight for column, weight in component_weights.items()
    ) / weight_sum
    # Use position/reported-speed agreement as a transparent quality multiplier.
    # Hard physical validity is handled separately by the eligibility flags.
    scored["selection_score"] = (
        scored["base_selection_score"] * scored["kinematic_quality_score"]
    )
    return scored


def _extract_exact_window(
    pairs: pd.DataFrame,
    leader_id: int,
    follower_id: int,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    selected = pairs.loc[
        pairs["leader_id"].eq(leader_id)
        & pairs["follower_id"].eq(follower_id)
        & pairs["frame_id"].between(start_frame, end_frame)
    ].sort_values("frame_id", kind="mergesort")
    return selected.reset_index(drop=True)


def _validate_manual_case(
    pairs: pd.DataFrame,
    manual: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    sampling_interval_s: float,
    time_tolerance_s: float,
    braking_threshold_m_s2: float,
    braking_consecutive_frames: int,
    response_search_window_s: float,
    preparation: Mapping[str, Any],
    maximum_plausible_speed_m_s: float,
    maximum_plausible_abs_acceleration_m_s2: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["leader_id", "follower_id", "start_frame", "end_frame"]
    missing = [key for key in required if key not in manual]
    if missing:
        raise ValueError(f"selection.manual_case is missing required keys: {missing}")
    leader_id, follower_id, start_frame, end_frame = (
        int(manual[key]) for key in required
    )
    if end_frame < start_frame:
        raise ValueError("selection.manual_case end_frame must be >= start_frame")
    window = _extract_exact_window(
        pairs, leader_id, follower_id, start_frame, end_frame
    )
    expected_frames = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    observed_frames = window["frame_id"].to_numpy(dtype=np.int64)
    errors: list[str] = []
    if len(window) != len(expected_frames) or not np.array_equal(
        observed_frames, expected_frames
    ):
        errors.append("the requested pair is not directly adjacent in every requested frame")
    if window["frame_id"].duplicated().any():
        errors.append("duplicate pair records exist within the requested frame range")
    if window["lane_id"].nunique() != 1:
        errors.append("the pair does not remain in one lane")
    passenger_class = int(selection["passenger_vehicle_class"])
    if not (
        window["leader_class"].eq(passenger_class).all()
        and window["follower_class"].eq(passenger_class).all()
    ):
        errors.append("both vehicles are not passenger cars throughout the window")
    if len(window) > 1 and not np.allclose(
        np.diff(window["time_s"].to_numpy(dtype=float)),
        sampling_interval_s,
        atol=time_tolerance_s,
        rtol=0,
    ):
        errors.append("timestamps are not continuous at the configured sampling interval")
    if not window["gap_m"].gt(0).all():
        errors.append("the requested window contains a nonpositive net gap")
    if not window["gap_m"].between(
        float(selection["minimum_gap_m"]),
        float(selection["maximum_gap_m"]),
        inclusive="both",
    ).all():
        errors.append("the requested window violates configured gap bounds")

    if window.empty:
        duration = float("nan")
        speed_range = float("nan")
        minimum_acceleration = float("nan")
    else:
        duration = float(window["time_s"].iloc[-1] - window["time_s"].iloc[0])
        speed_range = float(
            window["leader_speed_m_s"].max() - window["leader_speed_m_s"].min()
        )
        minimum_acceleration = float(window["leader_acceleration_m_s2"].min())
    if not (
        float(selection["minimum_duration_s"]) - time_tolerance_s
        <= duration
        <= float(selection["maximum_duration_s"]) + time_tolerance_s
    ):
        errors.append("the requested duration is outside configured 20-40 s bounds")
    if speed_range < float(selection["minimum_leader_speed_range_m_s"]):
        errors.append("leader speed variation is below the configured minimum")
    if minimum_acceleration > float(selection["minimum_leader_deceleration_m_s2"]):
        errors.append("leader braking is weaker than the configured requirement")
    if errors:
        raise ValueError("Invalid selection.manual_case: " + "; ".join(errors))

    padding = int(preparation.get("savgol_window_length", 11)) // 2
    lane_id = int(window["lane_id"].iloc[0])
    context = pairs.loc[
        pairs["leader_id"].eq(leader_id)
        & pairs["follower_id"].eq(follower_id)
        & pairs["lane_id"].eq(lane_id)
        & pairs["frame_id"].between(start_frame - padding, end_frame + padding)
    ].sort_values("frame_id", kind="mergesort")
    context = _add_prepared_kinematic_estimates(context, preparation, sampling_interval_s)
    estimated_window = context.loc[
        context["frame_id"].between(start_frame, end_frame)
    ]
    metrics = _window_metrics(
        estimated_window,
        sampling_interval_s=sampling_interval_s,
        braking_threshold_m_s2=braking_threshold_m_s2,
        braking_consecutive_frames=braking_consecutive_frames,
        response_search_window_s=response_search_window_s,
    )
    prepared_errors: list[str] = []
    if min(
        metrics["leader_prepared_min_speed_m_s"],
        metrics["follower_prepared_min_speed_m_s"],
    ) < 0:
        prepared_errors.append("prepared kinematics contain negative speed")
    if max(
        metrics["leader_prepared_max_speed_m_s"],
        metrics["follower_prepared_max_speed_m_s"],
    ) > maximum_plausible_speed_m_s:
        prepared_errors.append("prepared kinematics exceed the configured speed limit")
    if max(
        metrics["leader_prepared_max_abs_acceleration_m_s2"],
        metrics["follower_prepared_max_abs_acceleration_m_s2"],
    ) > maximum_plausible_abs_acceleration_m_s2:
        prepared_errors.append("prepared kinematics exceed the configured acceleration limit")
    if prepared_errors:
        raise ValueError(
            "Invalid selection.manual_case after configured trajectory preparation: "
            + "; ".join(prepared_errors)
        )
    return window, metrics


def _save_candidate_preview(
    pairs: pd.DataFrame,
    displayed_candidates: pd.DataFrame,
    selected_key: tuple[int, int, int, int],
    destination: Path,
) -> None:
    rows: list[tuple[pd.Series, pd.DataFrame]] = []
    for _, candidate in displayed_candidates.head(4).iterrows():
        key = (
            int(candidate["leader_id"]),
            int(candidate["follower_id"]),
            int(candidate["start_frame"]),
            int(candidate["end_frame"]),
        )
        window = _extract_exact_window(pairs, *key)
        if not window.empty:
            rows.append((candidate, window))
    if not rows:
        raise ValueError("No candidate data available for preview")

    figure, axes = plt.subplots(
        len(rows), 3, figsize=(15, 3.5 * len(rows)), squeeze=False, constrained_layout=True
    )
    for row_index, (candidate, window) in enumerate(rows):
        relative_time = window["time_s"] - window["time_s"].iloc[0]
        key = (
            int(candidate["leader_id"]),
            int(candidate["follower_id"]),
            int(candidate["start_frame"]),
            int(candidate["end_frame"]),
        )
        selected_marker = " [SELECTED]" if key == selected_key else ""
        title = (
            f"L={key[0]}, F={key[1]}, lane={int(candidate['lane_id'])}, "
            f"score={float(candidate.get('selection_score', float('nan'))):.3f}"
            f"{selected_marker}"
        )
        axes[row_index, 0].plot(relative_time, window["leader_y"], label="Leader")
        axes[row_index, 0].plot(relative_time, window["follower_y"], label="Follower")
        axes[row_index, 0].set_ylabel("Longitudinal position (m)")
        axes[row_index, 0].set_title(title)
        axes[row_index, 0].legend(loc="best")

        axes[row_index, 1].plot(
            relative_time, window["leader_speed_m_s"], label="Leader"
        )
        axes[row_index, 1].plot(
            relative_time, window["follower_speed_m_s"], label="Follower"
        )
        axes[row_index, 1].set_ylabel("Speed (m/s)")
        axes[row_index, 1].legend(loc="best")

        axes[row_index, 2].plot(relative_time, window["gap_m"], color="#F28E2B")
        axes[row_index, 2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[row_index, 2].set_ylabel("Net gap (m)")
        for axis in axes[row_index]:
            axis.set_xlabel("Time from window start (s)")
            axis.grid(alpha=0.25)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def select_case(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["_paths"]
    selection = require_config_section(config, "selection")
    data_config = require_config_section(config, "data")
    preparation = require_config_section(config, "preparation")
    output_dir = ensure_output_dir(config, "cases_dir")
    pairs_path = Path(paths["pairs_dir"]) / "frame_pairs.csv"
    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Pair table not found: {pairs_path}. Run 01_find_following_pairs.py first."
        )
    pairs = pd.read_csv(pairs_path)
    missing_pair_columns = sorted(set(PAIR_COLUMNS).difference(pairs.columns))
    if missing_pair_columns:
        raise ValueError(f"frame_pairs.csv is missing columns: {missing_pair_columns}")

    dt = float(data_config.get("sampling_interval_s", 0.1))
    tolerance = float(data_config.get("time_grid_tolerance_s", 1e-6))
    minimum_duration = float(selection["minimum_duration_s"])
    maximum_duration = float(selection["maximum_duration_s"])
    if not 0 < minimum_duration <= maximum_duration:
        raise ValueError("selection duration bounds must satisfy 0 < minimum <= maximum")
    minimum_count = int(math.ceil((minimum_duration - tolerance) / dt)) + 1
    maximum_count = int(math.floor((maximum_duration + tolerance) / dt)) + 1
    step_count = max(1, int(round(float(selection.get("window_step_s", 1.0)) / dt)))
    passenger_class = int(selection["passenger_vehicle_class"])
    preferred_lanes = {int(value) for value in selection["preferred_lanes"]}
    minimum_gap = float(selection["minimum_gap_m"])
    maximum_gap = float(selection["maximum_gap_m"])
    maximum_plausible_speed = float(
        data_config.get("maximum_plausible_speed_m_s", 60.0)
    )
    maximum_plausible_abs_acceleration = float(
        data_config.get("maximum_plausible_abs_acceleration_m_s2", 15.0)
    )

    row_eligible = (
        pairs["leader_class"].eq(passenger_class)
        & pairs["follower_class"].eq(passenger_class)
        & pairs["lane_id"].isin(preferred_lanes)
        & pairs["gap_m"].between(minimum_gap, maximum_gap, inclusive="both")
        & pairs["gap_m"].gt(0)
    )
    selection_rows = pairs.loc[row_eligible].copy()
    LOGGER.info(
        "Segmenting %d passenger-car adjacency records in preferred lanes",
        len(selection_rows),
    )
    segmented = add_continuous_segment_ids(selection_rows, dt, tolerance)

    evaluation = config.get("evaluation", {})
    braking_threshold = float(
        evaluation.get(
            "braking_acceleration_threshold_m_s2",
            selection.get("minimum_leader_deceleration_m_s2", -0.5),
        )
    )
    braking_consecutive_frames = int(evaluation.get("braking_consecutive_frames", 3))
    response_search_window = float(evaluation.get("response_search_window_s", 8.0))

    candidate_records: list[dict[str, Any]] = []
    segment_keys = ["leader_id", "follower_id", "lane_id", "segment_number"]
    for segment_key, segment in segmented.groupby(
        segment_keys, sort=True, observed=True
    ):
        ordered_segment = segment.sort_values("frame_id", kind="mergesort")
        ordered_segment = _add_prepared_kinematic_estimates(
            ordered_segment, preparation, dt
        )
        starts = _window_starts(
            len(ordered_segment), minimum_count, maximum_count, step_count
        )
        for start in starts:
            stop = min(start + maximum_count, len(ordered_segment))
            window = ordered_segment.iloc[start:stop]
            metrics = _window_metrics(
                window,
                sampling_interval_s=dt,
                braking_threshold_m_s2=braking_threshold,
                braking_consecutive_frames=braking_consecutive_frames,
                response_search_window_s=response_search_window,
            )
            metrics["segment_id"] = str(window["segment_id"].iloc[0])
            candidate_records.append(metrics)

    candidates = pd.DataFrame(candidate_records)
    if candidates.empty:
        write_csv_atomic(candidates, output_dir / "candidate_cases.csv")
        raise RuntimeError(
            "No continuous passenger-car relationship reached the configured minimum duration"
        )
    candidates["meets_speed_variation"] = candidates[
        "leader_prepared_speed_range_m_s"
    ].ge(float(selection["minimum_leader_speed_range_m_s"]))
    candidates["meets_leader_deceleration"] = candidates[
        "leader_min_acceleration_m_s2"
    ].le(float(selection["minimum_leader_deceleration_m_s2"]))
    candidates["has_observable_follower_braking_response"] = candidates[
        "braking_response_lag_s"
    ].notna()
    candidates["prepared_kinematics_valid"] = (
        candidates["leader_prepared_min_speed_m_s"].ge(0)
        & candidates["follower_prepared_min_speed_m_s"].ge(0)
        & candidates["leader_prepared_max_speed_m_s"].le(maximum_plausible_speed)
        & candidates["follower_prepared_max_speed_m_s"].le(maximum_plausible_speed)
        & candidates["leader_prepared_max_abs_acceleration_m_s2"].le(
            maximum_plausible_abs_acceleration
        )
        & candidates["follower_prepared_max_abs_acceleration_m_s2"].le(
            maximum_plausible_abs_acceleration
        )
    )
    candidates["eligible"] = (
        candidates["meets_speed_variation"]
        & candidates["meets_leader_deceleration"]
        & candidates["prepared_kinematics_valid"]
    )
    candidates = _score_candidates(candidates, selection)
    candidates.sort_values(
        [
            "eligible",
            "selection_score",
            "leader_prepared_speed_range_m_s",
            "leader_id",
            "follower_id",
            "start_frame",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
        inplace=True,
        ignore_index=True,
    )
    candidates.insert(0, "candidate_rank", np.arange(1, len(candidates) + 1))

    eligible_candidates = candidates.loc[candidates["eligible"]].copy()
    candidate_limit = int(selection.get("candidate_limit", 100))
    delivered_candidates = eligible_candidates.head(candidate_limit).copy()
    write_csv_atomic(delivered_candidates, output_dir / "candidate_cases.csv")
    if eligible_candidates.empty:
        write_json_atomic(
            {
                "status": "no_eligible_case",
                "continuous_windows_evaluated": int(len(candidates)),
                "failure_counts": {
                    "below_speed_variation_threshold": int(
                        (~candidates["meets_speed_variation"]).sum()
                    ),
                    "without_required_leader_deceleration": int(
                        (~candidates["meets_leader_deceleration"]).sum()
                    ),
                    "invalid_prepared_kinematics": int(
                        (~candidates["prepared_kinematics_valid"]).sum()
                    ),
                },
            },
            output_dir / "case_selection_summary.json",
        )
        raise RuntimeError(
            "Continuous windows exist, but none satisfy the configured speed-variation "
            "and braking requirements; thresholds were not relaxed"
        )

    manual_case = selection.get("manual_case")
    if manual_case is not None:
        if not isinstance(manual_case, Mapping):
            raise ValueError("selection.manual_case must be null or a mapping")
        selected_window, selected_metrics = _validate_manual_case(
            pairs,
            manual_case,
            selection,
            sampling_interval_s=dt,
            time_tolerance_s=tolerance,
            braking_threshold_m_s2=braking_threshold,
            braking_consecutive_frames=braking_consecutive_frames,
            response_search_window_s=response_search_window,
            preparation=preparation,
            maximum_plausible_speed_m_s=maximum_plausible_speed,
            maximum_plausible_abs_acceleration_m_s2=maximum_plausible_abs_acceleration,
        )
        selection_mode = "manual_validated"
        selection_reason = (
            "User-configured manual case; accepted only after validating direct adjacency, "
            "single-lane continuity, passenger classes, 0.1 s sampling, configured gap and "
            "duration bounds, leader speed variation, and leader braking."
        )
        match = candidates.loc[
            candidates["leader_id"].eq(selected_metrics["leader_id"])
            & candidates["follower_id"].eq(selected_metrics["follower_id"])
            & candidates["start_frame"].eq(selected_metrics["start_frame"])
            & candidates["end_frame"].eq(selected_metrics["end_frame"])
        ]
        if not match.empty:
            selected_metrics.update(match.iloc[0].to_dict())
        else:
            temporary = _score_candidates(pd.DataFrame([selected_metrics]), selection)
            selected_metrics.update(temporary.iloc[0].to_dict())
    else:
        selected_row = eligible_candidates.iloc[0]
        selected_metrics = selected_row.to_dict()
        selected_window = _extract_exact_window(
            pairs,
            int(selected_row["leader_id"]),
            int(selected_row["follower_id"]),
            int(selected_row["start_frame"]),
            int(selected_row["end_frame"]),
        )
        selection_mode = "automatic_rank_1"
        selection_reason = (
            "Highest deterministic composite score among continuous direct passenger-car "
            "pairs in preferred mainline lanes that satisfied configured duration, positive "
            "gap, leader speed-variation, and braking requirements."
        )

    selected_window = selected_window.sort_values("frame_id", kind="mergesort").reset_index(
        drop=True
    )
    write_csv_atomic(selected_window, output_dir / "selected_case_raw.csv")
    selected_key = (
        int(selected_metrics["leader_id"]),
        int(selected_metrics["follower_id"]),
        int(selected_metrics["start_frame"]),
        int(selected_metrics["end_frame"]),
    )

    preview_candidates = delivered_candidates.copy()
    manual_preview_row = pd.DataFrame([selected_metrics])
    if manual_case is not None:
        preview_candidates = pd.concat(
            [manual_preview_row, preview_candidates], ignore_index=True, sort=False
        ).drop_duplicates(
            ["leader_id", "follower_id", "start_frame", "end_frame"], keep="first"
        )
    _save_candidate_preview(
        pairs,
        preview_candidates,
        selected_key,
        output_dir / "candidate_preview.png",
    )

    metrics_for_json = {
        key: value
        for key, value in selected_metrics.items()
        if key
        not in {
            "leader_id",
            "follower_id",
            "lane_id",
            "start_frame",
            "end_frame",
            "start_time_s",
            "end_time_s",
            "duration_s",
        }
    }
    selected_case = {
        "leader_id": int(selected_metrics["leader_id"]),
        "follower_id": int(selected_metrics["follower_id"]),
        "lane_id": int(selected_metrics["lane_id"]),
        "start_frame": int(selected_metrics["start_frame"]),
        "end_frame": int(selected_metrics["end_frame"]),
        "start_time_s": float(selected_metrics["start_time_s"]),
        "end_time_s": float(selected_metrics["end_time_s"]),
        "duration_s": float(selected_metrics["duration_s"]),
        "selection_reason": selection_reason,
        "selection_mode": selection_mode,
        "metrics": metrics_for_json,
        "validated_constraints": {
            "direct_adjacent_pair_every_frame": True,
            "both_vehicle_classes": passenger_class,
            "constant_lane": True,
            "continuous_frames_and_timestamps": True,
            "strictly_positive_gap": True,
            "configured_gap_range_m": [minimum_gap, maximum_gap],
            "configured_duration_range_s": [minimum_duration, maximum_duration],
            "minimum_leader_speed_range_m_s": float(
                selection["minimum_leader_speed_range_m_s"]
            ),
            "required_leader_minimum_acceleration_m_s2": float(
                selection["minimum_leader_deceleration_m_s2"]
            ),
            "maximum_prepared_speed_m_s": maximum_plausible_speed,
            "maximum_prepared_abs_acceleration_m_s2": maximum_plausible_abs_acceleration,
        },
    }
    write_json_atomic(selected_case, output_dir / "selected_case.json")
    write_json_atomic(
        {
            "status": "selected",
            "selection_mode": selection_mode,
            "pair_observations_read": int(len(pairs)),
            "row_eligible_observations": int(row_eligible.sum()),
            "continuous_segments": int(
                segmented[segment_keys].drop_duplicates().shape[0]
            ),
            "windows_evaluated": int(len(candidates)),
            "eligible_windows": int(len(eligible_candidates)),
            "candidate_rows_written": int(len(delivered_candidates)),
            "thresholds_were_relaxed": False,
        },
        output_dir / "case_selection_summary.json",
    )
    LOGGER.info(
        "Selected real case L=%d F=%d lane=%d frames=%d-%d (%.1f s)",
        selected_case["leader_id"],
        selected_case["follower_id"],
        selected_case["lane_id"],
        selected_case["start_frame"],
        selected_case["end_frame"],
        selected_case["duration_s"],
    )
    return selected_case


def main() -> None:
    args = parse_args()
    select_case(load_config(args.config))


if __name__ == "__main__":
    main()
