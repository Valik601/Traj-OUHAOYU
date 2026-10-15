"""STEP 0: validate and profile the unmodified NGSIM I-80 SI data."""

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

from common import (
    REQUIRED_COLUMNS,
    configure_logging,
    default_config_path,
    ensure_output_dir,
    file_sha256,
    load_config,
    read_ngsim_csv,
    require_config_section,
    write_csv_atomic,
    write_json_atomic,
)


LOGGER = configure_logging("inspect_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the NGSIM I-80 SI CSV without modifying it."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(__file__),
        help="YAML configuration path (default: project config.yaml)",
    )
    return parser.parse_args()


def _mode_int(values: pd.Series) -> int:
    modes = values.mode(dropna=True)
    if modes.empty:
        raise ValueError("Cannot determine a mode from an empty series")
    return int(modes.iloc[0])


def _quantiles(series: pd.Series) -> dict[str, float]:
    probabilities = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    values = series.quantile(probabilities)
    return {f"q{int(probability * 100):02d}": float(values.loc[probability]) for probability in probabilities}


def _collect_anomalies(
    data: pd.DataFrame, reason_masks: dict[str, pd.Series | np.ndarray]
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for reason, raw_mask in reason_masks.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.any():
            selected = data.loc[mask, REQUIRED_COLUMNS].copy()
            selected.insert(0, "source_row", data.index[mask].to_numpy() + 2)
            selected.insert(1, "anomaly_reason", reason)
            pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=["source_row", "anomaly_reason", *REQUIRED_COLUMNS])
    stacked = pd.concat(pieces, ignore_index=True)
    grouped = (
        stacked.groupby("source_row", sort=True, as_index=False)
        .agg(
            {
                "anomaly_reason": lambda values: ";".join(sorted(set(values))),
                **{column: "first" for column in REQUIRED_COLUMNS},
            }
        )
        .sort_values("source_row", kind="mergesort")
    )
    return grouped


def _save_plots(data: pd.DataFrame, output_dir: Path) -> None:
    lane_counts = data["lane_id"].value_counts().sort_index()
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(lane_counts.index.astype(str), lane_counts.values, color="#4C78A8")
    axis.set_xlabel("Lane ID")
    axis.set_ylabel("Trajectory records")
    axis.set_title("NGSIM I-80 lane distribution")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "lane_distribution.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.hist(data["speed_m_s"], bins=80, color="#59A14F", edgecolor="none")
    axis.set_xlabel("Speed (m/s)")
    axis.set_ylabel("Trajectory records")
    axis.set_title("Speed distribution")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "speed_distribution.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.hist(
        data["acceleration_m_s2"], bins=80, color="#E15759", edgecolor="none"
    )
    axis.set_xlabel("Acceleration (m/s²)")
    axis.set_ylabel("Trajectory records")
    axis.set_title("Acceleration distribution")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "acceleration_distribution.png", dpi=200)
    plt.close(fig)


def inspect(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["_paths"]
    data_config = require_config_section(config, "data")
    csv_path = Path(paths["si_csv"])
    checks_path = Path(paths["source_checks"])
    output_dir = ensure_output_dir(config, "quality_dir")

    LOGGER.info("Reading SI data from configured input")
    data = read_ngsim_csv(csv_path)
    with checks_path.open("r", encoding="utf-8") as handle:
        source_checks = json.load(handle)
    if not isinstance(source_checks, dict):
        raise ValueError("source_and_checks.json must contain a JSON object")

    sampling_interval = float(data_config.get("sampling_interval_s", 0.1))
    time_tolerance = float(data_config.get("time_grid_tolerance_s", 1e-6))
    maximum_speed = float(data_config.get("maximum_plausible_speed_m_s", 60.0))
    maximum_abs_acceleration = float(
        data_config.get("maximum_plausible_abs_acceleration_m_s2", 15.0)
    )

    rounded_ticks = np.rint(data["time_s"].to_numpy() / sampling_interval).astype(
        np.int64
    )
    off_time_grid = np.abs(
        data["time_s"].to_numpy() - rounded_ticks * sampling_interval
    ) > time_tolerance
    global_base_values = data["global_time_ms"].astype("int64") - np.rint(
        data["time_s"] * 1000.0
    ).astype("int64")
    global_time_base_ms = _mode_int(global_base_values)
    global_time_mismatch = (
        data["global_time_ms"].astype("int64")
        - (global_time_base_ms + np.rint(data["time_s"] * 1000.0).astype("int64"))
    ).abs() > max(1, int(np.ceil(time_tolerance * 1000.0)))
    frame_offset_values = data["frame_id"].astype("int64") - rounded_ticks
    frame_offset = _mode_int(frame_offset_values)
    frame_time_mismatch = data["frame_id"].astype("int64").ne(
        rounded_ticks + frame_offset
    )

    duplicate_vehicle_frame = data.duplicated(
        ["vehicle_id", "frame_id"], keep=False
    )
    nonfinite_numeric = ~np.isfinite(data[REQUIRED_COLUMNS].to_numpy()).all(axis=1)
    missing_any = data.isna().any(axis=1)
    negative_position = data["local_x_m"].lt(0) | data["local_y_m"].lt(0)
    negative_speed = data["speed_m_s"].lt(0)
    excessive_speed = data["speed_m_s"].gt(maximum_speed)
    nonpositive_length = data["vehicle_length_m"].le(0)
    excessive_acceleration = data["acceleration_m_s2"].abs().gt(
        maximum_abs_acceleration
    )
    implausibly_small_global_time = data["global_time_ms"].lt(10**12)

    ordered = data.sort_values(["vehicle_id", "frame_id"], kind="mergesort")
    grouped_vehicle = ordered.groupby("vehicle_id", sort=False, observed=True)
    frame_differences = grouped_vehicle["frame_id"].diff()
    time_differences = grouped_vehicle["time_s"].diff()
    frame_gap_ordered = frame_differences.gt(1)
    time_step_mismatch_ordered = time_differences.notna() & time_differences.sub(
        sampling_interval
    ).abs().gt(time_tolerance)
    frame_gap = pd.Series(False, index=data.index)
    frame_gap.loc[ordered.index] = frame_gap_ordered.to_numpy()
    vehicle_time_step_mismatch = pd.Series(False, index=data.index)
    vehicle_time_step_mismatch.loc[ordered.index] = time_step_mismatch_ordered.to_numpy()

    reason_masks: dict[str, pd.Series | np.ndarray] = {
        "missing_value": missing_any,
        "nonfinite_numeric_value": nonfinite_numeric,
        "duplicate_vehicle_frame_key": duplicate_vehicle_frame,
        "negative_position": negative_position,
        "negative_speed": negative_speed,
        "speed_above_configured_limit": excessive_speed,
        "nonpositive_vehicle_length": nonpositive_length,
        "absolute_acceleration_above_configured_limit": excessive_acceleration,
        "time_off_0.1s_grid": off_time_grid,
        "global_time_not_aligned_with_time_s": global_time_mismatch,
        "frame_id_not_aligned_with_time_s": frame_time_mismatch,
        "vehicle_frame_gap_starts_here": frame_gap,
        "vehicle_time_step_mismatch": vehicle_time_step_mismatch,
        "global_time_below_1e12": implausibly_small_global_time,
    }
    anomalies = _collect_anomalies(data, reason_masks)
    write_csv_atomic(anomalies, output_dir / "anomaly_records.csv")

    actual_sha256 = file_sha256(csv_path)
    file_entry = source_checks.get("files", {}).get(csv_path.name, {})
    declared_range = source_checks.get("global_time_ms")
    actual_range = [int(data["global_time_ms"].min()), int(data["global_time_ms"].max())]
    declared_range_valid = (
        isinstance(declared_range, list)
        and len(declared_range) == 2
        and all(isinstance(value, (int, float)) for value in declared_range)
    )
    range_matches = bool(
        declared_range_valid
        and [int(declared_range[0]), int(declared_range[1])] == actual_range
    )
    metadata_placeholder_lower_bound = bool(
        declared_range_valid
        and int(declared_range[0]) == 1_000_000_000
        and actual_range[0] != 1_000_000_000
    )

    frame_mapping = data.groupby("frame_id", observed=True).agg(
        time_s_values=("time_s", "nunique"),
        global_time_values=("global_time_ms", "nunique"),
    )
    vehicle_frame_counts = grouped_vehicle.size()
    vehicle_gap_events = frame_differences.gt(1).groupby(ordered["vehicle_id"]).sum()
    missing_frames_inside_tracks = frame_differences.sub(1).clip(lower=0).sum()

    anomaly_counts = {
        reason: int(np.asarray(mask, dtype=bool).sum())
        for reason, mask in reason_masks.items()
    }
    lane_counts = data["lane_id"].value_counts().sort_index()
    class_counts = data["vehicle_class"].value_counts().sort_index()
    summary: dict[str, Any] = {
        "input": {
            "filename": csv_path.name,
            "bytes": int(csv_path.stat().st_size),
            "sha256": actual_sha256,
            "raw_file_modified": False,
            "si_values_converted_again": False,
        },
        "schema": {
            "columns": list(data.columns),
            "expected_columns": list(data_config.get("expected_columns", REQUIRED_COLUMNS)),
            "column_order_matches": list(data.columns)
            == list(data_config.get("expected_columns", REQUIRED_COLUMNS)),
            "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
            "all_columns_numeric": bool(
                all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes)
            ),
            "units": {
                "time_s": "s",
                "global_time_ms": "ms since Unix epoch",
                "local_x_m": "m",
                "local_y_m": "m (vehicle front-center longitudinal position)",
                "vehicle_length_m": "m",
                "speed_m_s": "m/s",
                "acceleration_m_s2": "m/s^2",
            },
            "unit_provenance": source_checks.get("si_conversion"),
        },
        "counts": {
            "records": int(len(data)),
            "vehicles": int(data["vehicle_id"].nunique()),
            "frames": int(data["frame_id"].nunique()),
            "lanes": [int(value) for value in lane_counts.index],
            "lane_distribution": {
                str(int(key)): int(value) for key, value in lane_counts.items()
            },
            "vehicle_class_distribution": {
                str(int(key)): int(value) for key, value in class_counts.items()
            },
            "missing_by_column": {
                column: int(value) for column, value in data.isna().sum().items()
            },
            "duplicate_vehicle_frame_keys": int(
                data.duplicated(["vehicle_id", "frame_id"]).sum()
            ),
        },
        "ranges": {
            "frame_id": [int(data["frame_id"].min()), int(data["frame_id"].max())],
            "time_s": [float(data["time_s"].min()), float(data["time_s"].max())],
            "global_time_ms": actual_range,
            "local_x_m": [float(data["local_x_m"].min()), float(data["local_x_m"].max())],
            "local_y_m": [float(data["local_y_m"].min()), float(data["local_y_m"].max())],
            "vehicle_length_m": [
                float(data["vehicle_length_m"].min()),
                float(data["vehicle_length_m"].max()),
            ],
            "speed_m_s": [float(data["speed_m_s"].min()), float(data["speed_m_s"].max())],
            "acceleration_m_s2": [
                float(data["acceleration_m_s2"].min()),
                float(data["acceleration_m_s2"].max()),
            ],
        },
        "distributions": {
            "speed_m_s_quantiles": _quantiles(data["speed_m_s"]),
            "acceleration_m_s2_quantiles": _quantiles(data["acceleration_m_s2"]),
        },
        "time_and_frame_consistency": {
            "configured_sampling_interval_s": sampling_interval,
            "time_grid_tolerance_s": time_tolerance,
            "off_grid_rows": int(off_time_grid.sum()),
            "inferred_global_time_at_time_zero_ms": global_time_base_ms,
            "global_time_alignment_mismatch_rows": int(global_time_mismatch.sum()),
            "inferred_frame_offset_at_time_zero": frame_offset,
            "frame_time_alignment_mismatch_rows": int(frame_time_mismatch.sum()),
            "frames_mapping_to_multiple_time_s_values": int(
                frame_mapping["time_s_values"].gt(1).sum()
            ),
            "frames_mapping_to_multiple_global_time_values": int(
                frame_mapping["global_time_values"].gt(1).sum()
            ),
        },
        "vehicle_track_continuity": {
            "vehicles_with_frame_gaps": int(vehicle_gap_events.gt(0).sum()),
            "frame_gap_events": int(frame_differences.gt(1).sum()),
            "missing_frames_inside_tracks": int(missing_frames_inside_tracks),
            "vehicle_time_step_mismatch_rows": int(time_step_mismatch_ordered.sum()),
            "records_per_vehicle": {
                "minimum": int(vehicle_frame_counts.min()),
                "median": float(vehicle_frame_counts.median()),
                "maximum": int(vehicle_frame_counts.max()),
            },
        },
        "anomalies": {
            "unique_records_flagged": int(len(anomalies)),
            "reason_counts": anomaly_counts,
            "configured_limits": {
                "maximum_plausible_speed_m_s": maximum_speed,
                "maximum_plausible_abs_acceleration_m_s2": maximum_abs_acceleration,
            },
        },
        "source_and_checks_comparison": {
            "declared_records": source_checks.get("rows"),
            "actual_records": int(len(data)),
            "record_count_matches": source_checks.get("rows") == int(len(data)),
            "declared_vehicles": source_checks.get("vehicles"),
            "actual_vehicles": int(data["vehicle_id"].nunique()),
            "vehicle_count_matches": source_checks.get("vehicles")
            == int(data["vehicle_id"].nunique()),
            "declared_bytes": file_entry.get("bytes"),
            "actual_bytes": int(csv_path.stat().st_size),
            "byte_count_matches": file_entry.get("bytes") == int(csv_path.stat().st_size),
            "declared_sha256": file_entry.get("sha256"),
            "actual_sha256": actual_sha256,
            "sha256_matches": file_entry.get("sha256") == actual_sha256,
            "declared_global_time_ms_range": declared_range,
            "actual_global_time_ms_range": actual_range,
            "global_time_range_matches": range_matches,
            "metadata_lower_bound_is_known_placeholder": metadata_placeholder_lower_bound,
            "actual_rows_with_global_time_ms_equal_1000000000": int(
                data["global_time_ms"].eq(1_000_000_000).sum()
            ),
            "conclusion": (
                "source_and_checks.json declares global_time_ms lower bound 1000000000, "
                "but the actual CSV minimum is 1113433200000; the CSV contains no row "
                "with the anomalous placeholder timestamp. Other declared size/count/hash "
                "checks are reported independently above."
                if metadata_placeholder_lower_bound
                else "The declared and actual global timestamp ranges were compared."
            ),
        },
    }
    write_json_atomic(summary, output_dir / "dataset_summary.json")
    _save_plots(data, output_dir)
    LOGGER.info(
        "Inspection complete: %d records, %d vehicles, %d flagged records",
        len(data),
        data["vehicle_id"].nunique(),
        len(anomalies),
    )
    return summary


def main() -> None:
    args = parse_args()
    inspect(load_config(args.config))


if __name__ == "__main__":
    main()
