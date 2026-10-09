#!/usr/bin/env python3
"""Run the complete NGSIM I-80 fundamental-diagram workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.adaptive_aggregation import adaptive_states
from src.config import ensure_output_tree, load_config
from src.data_loader import load_trajectory_csv
from src.evaluation import evaluate_methods, parameter_sensitivity
from src.parallelogram_aggregation import (
    estimate_wave_speed,
    parallelogram_parameter_sweep,
    parallelogram_states,
)
from src.preprocessing import preprocess_lane
from src.rectangular_aggregation import (
    generate_common_anchors,
    rectangular_parameter_sweep,
    rectangular_states,
)
from src.report_generator import generate_method_figures, generate_report
from src.trajectory_processing import build_segments, downsample_points


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Construct empirical fundamental diagrams from NGSIM trajectories")
    parser.add_argument("--input", required=True, help="Input NGSIM CSV")
    parser.add_argument("--config", default=str(project / "config.yaml"))
    parser.add_argument("--lane", type=int, help="Override project.lane_id")
    parser.add_argument("--output-root", default=str(project / "results"))
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sampling_sensitivity(
    cleaned: pd.DataFrame,
    anchors: pd.DataFrame,
    wave_speed: float,
    base_states: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    records = []
    for interval in config["evaluation"]["sampling_intervals_s"]:
        interval = float(interval)
        print(f"[sensitivity] sampling interval {interval:.1f} s", flush=True)
        if abs(interval - 0.1) < 1e-9:
            method_states = base_states
        else:
            sampled = downsample_points(cleaned, interval)
            segments = build_segments(sampled)
            method_states = {
                "Rectangular": rectangular_states(segments, anchors, config),
                "Parallelogram": parallelogram_states(segments, anchors, wave_speed, config),
            }
            adaptive, _ = adaptive_states(segments, anchors, wave_speed, config, progress=False)
            method_states["Adaptive"] = adaptive
        metrics, _ = evaluate_methods(method_states, config)
        for row in metrics.itertuples(index=False):
            records.append(
                {
                    "sampling_interval_s": interval,
                    "method": row.method,
                    "valid_samples": int(row.valid_samples),
                    "capacity_veh_h": row.capacity_veh_h,
                    "critical_density_veh_km": row.critical_density_veh_km,
                    "test_rmse_veh_h": row.test_rmse_veh_h,
                    "max_q_kv_relative_error": row.max_q_kv_relative_error,
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    lane_id = int(args.lane if args.lane is not None else config["project"]["lane_id"])
    config["project"]["lane_id"] = lane_id
    outputs = ensure_output_tree(args.output_root)

    print("[1/12] Inspecting and loading dataset", flush=True)
    raw, load_summary = load_trajectory_csv(args.input, config)
    print("[2/12] Validating SI coordinate system and common longitudinal origin", flush=True)
    if config["input"]["source_units"] == "si" and not {"s_m", "time_s"}.issubset(raw.columns):
        raise ValueError("SI input requires s_m and time_s after field mapping")

    print(f"[3/12] Preprocessing lane {lane_id} and reconstructing continuous runs", flush=True)
    cleaned, preprocessing_summary = preprocess_lane(raw, lane_id, config)
    del raw
    cleaned_path = outputs["processed"] / f"lane_{lane_id}_cleaned_trajectories.csv"
    cleaned.to_csv(cleaned_path, index=False)
    save_json(
        outputs["processed"] / "preprocessing_summary.json",
        {"load": load_summary, "preprocessing": preprocessing_summary},
    )

    print("[4/12] Building time-space segments and estimating wave speed", flush=True)
    segments = build_segments(cleaned)
    wave_speed, wave_diagnostics = estimate_wave_speed(cleaned, config)
    wave_diagnostics.to_csv(outputs["metrics"] / "wave_speed_diagnostics.csv", index=False)
    save_json(
        outputs["metrics"] / "wave_speed_selection.json",
        {
            "selected_wave_speed_m_s": wave_speed,
            "selected_wave_speed_km_h": wave_speed * 3.6,
            "method": "lagged congested speed-field correlation",
            "paper_note": "The wave speed is exogenous in He & Wu (2025); this workflow estimates a candidate and permits manual override.",
        },
    )
    anchors = generate_common_anchors(
        tuple(preprocessing_summary["time_range_s"]),
        tuple(preprocessing_summary["space_range_m"]),
        wave_speed,
        config,
    )
    anchors.to_csv(outputs["processed"] / "common_window_anchors.csv", index=False)

    print("[5/12] Time-space plotting data prepared", flush=True)
    print("[6/12] Rectangular Edie aggregation", flush=True)
    rectangular = rectangular_states(segments, anchors, config)
    print("[7/12] Wave-aligned parallelogram Edie aggregation", flush=True)
    parallelogram = parallelogram_states(segments, anchors, wave_speed, config)
    print("[8/12] Exploratory adaptive aggregation", flush=True)
    adaptive, adaptive_candidates = adaptive_states(segments, anchors, wave_speed, config)
    states = {
        "Rectangular": rectangular,
        "Parallelogram": parallelogram,
        "Adaptive": adaptive,
    }
    for method, table in states.items():
        table.to_csv(outputs["states"] / f"{method.lower()}_states.csv", index=False)
    adaptive_candidates.to_csv(outputs["states"] / "adaptive_candidate_windows.csv", index=False)
    adaptive_class_summary = (
        adaptive.groupby(["state_class", "orientation"], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
        .sort_values(["state_class", "orientation"], kind="stable")
    )
    adaptive_class_summary.to_csv(
        outputs["metrics"] / "adaptive_state_class_summary.csv", index=False
    )

    print("[9/12] Parameter and sampling sensitivity", flush=True)
    rect_sweep = rectangular_parameter_sweep(segments, anchors, config)
    para_sweep = parallelogram_parameter_sweep(segments, anchors, wave_speed, config)
    rect_sweep.to_csv(outputs["states"] / "rectangular_parameter_sweep.csv", index=False)
    para_sweep.to_csv(outputs["states"] / "parallelogram_parameter_sweep.csv", index=False)
    sweep_metrics = parameter_sensitivity(pd.concat([rect_sweep, para_sweep], ignore_index=True))
    sweep_metrics.to_csv(outputs["metrics"] / "parameter_sensitivity.csv", index=False)
    sampling_metrics = _sampling_sensitivity(cleaned, anchors, wave_speed, states, config)
    sampling_metrics.to_csv(outputs["metrics"] / "sampling_sensitivity.csv", index=False)

    print("[10/12] Time-block evaluation and physical checks", flush=True)
    metrics, fit_details = evaluate_methods(states, config)
    metrics.to_csv(outputs["metrics"] / "evaluation_metrics.csv", index=False)
    save_json(outputs["metrics"] / "evaluation_details.json", fit_details)
    max_closure_error = float(metrics["max_q_kv_relative_error"].max())
    if not np.isfinite(max_closure_error) or max_closure_error > 1e-10:
        raise RuntimeError(f"Edie q=k*v closure failed: {max_closure_error}")

    print("[11/12] Generating one-page A3 report", flush=True)
    pdf_path, png_path, manifest_path = generate_report(
        cleaned, states, metrics, fit_details, wave_speed, config, outputs["figures"]
    )
    diagnostic_figures = generate_method_figures(states, fit_details, config, outputs["figures"])
    print("[12/12] Verifying output inventory", flush=True)
    summary = {
        "lane_id": lane_id,
        "cleaned_rows": int(len(cleaned)),
        "trajectory_segments": int(segments.size),
        "common_anchors": int(len(anchors)),
        "wave_speed_m_s": wave_speed,
        "max_q_kv_relative_error": max_closure_error,
        "report_pdf": str(pdf_path),
        "report_png": str(png_path),
        "report_manifest": str(manifest_path),
        "method_diagnostic_figures": [str(path) for path in diagnostic_figures],
    }
    save_json(outputs["root"] / "run_summary.json", summary)
    print(json.dumps(json_ready(summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

