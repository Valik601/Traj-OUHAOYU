# OWNER: viz-dev
"""Smoke tests for ``uavbench.visualization`` on a synthetic results tree.

The fixture builds a tiny but schema-faithful results tree (CONTRACTS.md §11.3 / §12): a handful of
scene ids, both scenarios, all models, ``metrics.json`` / ``agents.csv`` / ``center_events.csv`` /
``violation_events.csv`` / ``boundary_contacts.csv`` / ``min_distance_curve.csv`` /
``arrival_curve.csv`` / ``trajectories.npz`` / ``provenance.json`` / ``config.json`` per run, plus
the aggregate ``comparison.csv`` / ``summary_statistics.csv`` / ``paired_effects.csv``.

Assertions cover: every renderer produces non-empty PNG/GIF files inside ``out_dir``, matplotlib
figures are closed, the median-case selection rule is deterministic, figure 3 recomputes the
per-interval continuous minimum separation (not endpoint distances), and missing data degrades
gracefully with warnings instead of exceptions. Runtime target < 60 s.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uavbench.visualization import (  # noqa: E402
    DISPLAY_SEED,
    interval_min_separation,
    render_all,
    render_animation,
    render_scene_seed,
)

# ---------------------------------------------------------------------------
# synthetic data constants
# ---------------------------------------------------------------------------
N_AGENTS = 6
K_STEPS = 40
DT = 0.1
OBSERVATION_TIME = K_STEPS * DT
SCENE_SEEDS = (1000, 1001, 1002)
SCENE_RADIUS = 20.0
CONFIG_HASH = "cfg0001"
MODEL_ORDER = ("direct", "vo", "rvo", "orca", "orca_release", "direct_release")
MODEL_INDEX = {m: i for i, m in enumerate(MODEL_ORDER)}
SCENARIO_MODELS = {
    "A": ("direct", "vo", "rvo", "orca"),
    "B": ("direct", "vo", "rvo", "orca", "orca_release", "direct_release"),
}
#: orca_release scenario-B safe_completion_rate per scene: median = 0.5 -> median case seed 1001
ORCA_RELEASE_RATES = {1000: 1.0 / 6.0, 1001: 3.0 / 6.0, 1002: 5.0 / 6.0}
RELEASE_DELTA = 0.75

AGENTS_COLUMNS = [
    "agent_id", "release_rank", "release_planned", "release_actual", "wait_duration",
    "center_entry_time", "center_crossing", "arrival_time", "post_release_travel_time",
    "phase_wait_duration", "phase_to_center_duration", "phase_to_goal_duration",
    "path_length", "final_goal_distance", "final_stage",
    "violated", "safe_completed", "stalled", "low_progress_mobile",
]
COMPARISON_COLUMNS = [
    "experiment_id", "model", "scenario", "scene_id", "seed", "scene_hash", "config_hash",
    "status", "valid", "arrival_rate", "safe_completion_rate", "center_crossing_rate",
    "min_separation", "violating_pair_count", "violation_event_count",
    "all_arrival_time", "t90", "average_path_length", "completed",
    "team_safe_success", "planning_time_mean", "planning_time_p95",
    "sample_exhausted_count", "solver_fallback_count",
]


# ---------------------------------------------------------------------------
# fixture: synthetic results tree
# ---------------------------------------------------------------------------
def _directions(count: int) -> np.ndarray:
    i = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / count
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def _build_scene(path: Path, scene_id: str, seed: int, dirs: np.ndarray) -> None:
    starts = SCENE_RADIUS * dirs
    goals = -starts
    meta = {
        "mode": "synthetic_fixture",
        "sphere_radius": SCENE_RADIUS,
        "n_agents": int(dirs.shape[0]),
        "seed": int(seed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        scene_id=np.asarray(scene_id),
        seed=np.asarray(int(seed), dtype=np.int64),
        starts=np.ascontiguousarray(starts, dtype=np.float64),
        goals=np.ascontiguousarray(goals, dtype=np.float64),
        agent_ids=np.arange(dirs.shape[0], dtype=np.int64),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )


def _bootstrap_mean_ci(values: np.ndarray, resamples: int = 2000, seed: int = 20261030):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    if arr.size == 1:
        return (mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, arr.size, size=(resamples, arr.size))].mean(axis=1)
    return (mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(arr.size))


def _bootstrap_diff_ci(diffs: np.ndarray, resamples: int = 2000, seed: int = 20261030):
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    if arr.size == 1:
        return (mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, arr.size, size=(resamples, arr.size))].mean(axis=1)
    return (mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(arr.size))


def _make_profile(scenario: str, model: str, seed: int) -> dict:
    """Deterministic per-run behaviour (arrivals, violations, centre visits, release)."""
    rng = np.random.default_rng(int(seed) * 131 + MODEL_INDEX[model] * 17 + (0 if scenario == "A" else 7))
    is_release = scenario == "B" and model.endswith("_release")
    if scenario == "B" and model == "orca_release":
        n_safe = int(round(ORCA_RELEASE_RATES[int(seed)] * N_AGENTS))
        n_arrived = min(N_AGENTS, n_safe + 2)
    else:
        n_arrived = int(rng.integers(2, N_AGENTS + 1))
        n_safe = int(rng.integers(0, n_arrived + 1))
    # a best-case run so the team-safe-success panel is not all zeros
    if (scenario, model, int(seed)) in {("B", "orca", 1000), ("A", "direct", 1000)}:
        n_arrived = N_AGENTS
        n_safe = N_AGENTS
    ids = np.arange(N_AGENTS, dtype=np.int64)
    if n_safe == N_AGENTS:
        safe_ids = set(int(i) for i in ids)
    elif n_safe == 0:
        safe_ids = set()
    else:
        safe_ids = {int(i) for i in rng.choice(ids, n_safe, replace=False)}
    non_safe = [int(i) for i in ids if int(i) not in safe_ids]
    n_extra = max(0, n_arrived - n_safe)
    extra = (
        {int(i) for i in rng.choice(np.asarray(non_safe), n_extra, replace=False)}
        if n_extra and non_safe
        else set()
    )
    arrived_ids = safe_ids | extra
    n_violated = 0 if (scenario, model, int(seed)) in {("B", "orca", 1000), ("A", "direct", 1000)} else int(
        rng.integers(0, 4)
    )
    n_violated = min(n_violated, len(non_safe))
    violated_ids = (
        {int(i) for i in rng.choice(np.asarray(non_safe), n_violated, replace=False)}
        if n_violated and non_safe
        else set()
    )
    rank = np.arange(N_AGENTS, dtype=np.int64)
    planned = rank.astype(np.float64) * RELEASE_DELTA if is_release else np.full(N_AGENTS, np.nan)
    actual = (
        np.asarray([math.ceil(round(float(p), 9) / DT) * DT for p in planned], dtype=np.float64)
        if is_release
        else np.full(N_AGENTS, np.nan)
    )
    return {
        "rng": rng,
        "is_release": is_release,
        "arrived_ids": arrived_ids,
        "safe_ids": safe_ids,
        "violated_ids": violated_ids,
        "planned": planned,
        "actual": actual,
        "min_sep": (
            float(rng.uniform(0.25, 0.95))
            if violated_ids
            else float(rng.uniform(1.02, 2.4))
        ),
        "path_length": float(rng.uniform(38.0, 55.0)),
        "n_center": (N_AGENTS - int(rng.integers(0, 2))) if scenario == "B" else 0,
    }


def _make_trajectory(scenario: str, profile: dict, starts: np.ndarray, goals: np.ndarray) -> dict:
    rng = profile["rng"]
    n = starts.shape[0]
    t = np.arange(K_STEPS + 1, dtype=np.float64) * DT
    positions = np.empty((K_STEPS + 1, n, 3), dtype=np.float64)
    stages = np.zeros((K_STEPS + 1, n), dtype=np.int8)
    velocities = np.zeros((K_STEPS, n, 3), dtype=np.float64)
    arrival_steps = np.full(n, -1, dtype=np.int64)
    center_steps = np.full(n, -1, dtype=np.int64)
    release_steps = np.zeros(n, dtype=np.int64)
    if profile["is_release"]:
        release_steps = np.rint(profile["actual"] / DT).astype(np.int64)
    for agent in range(n):
        start_step = int(release_steps[agent]) if profile["is_release"] else 0
        arrived = agent in profile["arrived_ids"]
        if arrived:
            end_step = min(K_STEPS, start_step + 16 + int(agent % 5))
            arrival_steps[agent] = end_step
        else:
            end_step = K_STEPS
        # centre crossing only for scenario B and not the last agent
        if scenario == "B" and agent < max(1, profile["n_center"]):
            center_steps[agent] = min(
                start_step + 5 + int(agent % 3),
                end_step - 1 if end_step > start_step + 6 else start_step + 1,
            )
        segment = end_step - start_step
        for step in range(K_STEPS + 1):
            if step <= start_step:
                positions[step, agent] = starts[agent]
            else:
                frac = min(1.0, (step - start_step) / float(max(segment, 1)))
                positions[step, agent] = starts[agent] + frac * (goals[agent] - starts[agent])
        center_step = int(center_steps[agent])
        for step in range(K_STEPS + 1):
            if scenario == "B":
                if step < start_step:
                    stage = 0
                elif center_step >= 0 and step < center_step:
                    stage = 1
                elif arrived and step >= end_step:
                    stage = 3
                else:
                    stage = 2
            else:
                stage = 3 if (arrived and step >= end_step) else 2
            stages[step, agent] = stage
    velocities[:] = np.diff(positions, axis=0) / DT
    frozen = np.isin(stages, (0, 3))
    active = ~frozen[:-1]
    center_entry_times = np.full(n, np.nan)
    for agent in range(n):
        if center_steps[agent] >= 0:
            center_entry_times[agent] = t[center_steps[agent]] + 0.5 * DT
    arrival_times = np.full(n, np.nan)
    for agent in range(n):
        if arrival_steps[agent] >= 0:
            arrival_times[agent] = t[arrival_steps[agent]]
    return {
        "t": t,
        "positions": positions,
        "velocities": velocities,
        "stages": stages,
        "frozen": frozen,
        "active": active,
        "center_entry_times": center_entry_times,
        "center_entry_steps": center_steps,
        "arrival_times": arrival_times,
        "release_planned": profile["planned"].copy(),
        "release_actual": profile["actual"].copy(),
        "planning_step_seconds": rng.uniform(0.0005, 0.002, size=K_STEPS).astype(np.float64),
        "n_steps_executed": np.asarray(K_STEPS, dtype=np.int64),
        "status_numerical_error": np.asarray(False),
        "status_valid": np.asarray(True),
    }


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


def _agents_frame(profile: dict, traj: dict, starts: np.ndarray, goals: np.ndarray) -> pd.DataFrame:
    n = starts.shape[0]
    rows = []
    for agent in range(n):
        arrived = agent in profile["arrived_ids"]
        violated = agent in profile["violated_ids"]
        safe = agent in profile["safe_ids"] and arrived
        center_time = float(traj["center_entry_times"][agent])
        arrival_time = float(traj["arrival_times"][agent])
        release_actual = float(profile["actual"][agent]) if profile["is_release"] else float("nan")
        release_planned = float(profile["planned"][agent]) if profile["is_release"] else float("nan")
        wait = release_actual if profile["is_release"] and np.isfinite(release_actual) else 0.0
        post_travel = (
            arrival_time - release_actual
            if profile["is_release"] and np.isfinite(release_actual) and np.isfinite(arrival_time)
            else float("nan")
        )
        phase_to_center = (
            center_time - release_actual
            if np.isfinite(center_time) and np.isfinite(release_actual)
            else float("nan")
        )
        phase_to_goal = (
            arrival_time - center_time
            if np.isfinite(arrival_time) and np.isfinite(center_time)
            else float("nan")
        )
        steps = np.linalg.norm(np.diff(traj["positions"][:, agent, :], axis=0), axis=1)
        rows.append(
            {
                "agent_id": agent,
                "release_rank": int(agent) if profile["is_release"] else -1,
                "release_planned": release_planned,
                "release_actual": release_actual,
                "wait_duration": float(wait),
                "center_entry_time": center_time,
                "center_crossing": int(np.isfinite(center_time)),
                "arrival_time": arrival_time,
                "post_release_travel_time": float(post_travel),
                "phase_wait_duration": float(wait),
                "phase_to_center_duration": float(phase_to_center),
                "phase_to_goal_duration": float(phase_to_goal),
                "path_length": float(np.sum(steps)),
                "final_goal_distance": float(np.linalg.norm(traj["positions"][-1, agent] - goals[agent])),
                "final_stage": int(traj["stages"][-1, agent]),
                "violated": int(violated),
                "safe_completed": int(safe),
                "stalled": 0,
                "low_progress_mobile": 0,
            }
        )
    return pd.DataFrame(rows, columns=AGENTS_COLUMNS)


def _arrival_curve_frame(traj: dict, agents: pd.DataFrame) -> pd.DataFrame:
    t = np.asarray(traj["t"], dtype=np.float64)
    arrivals = np.sort(_finite_or_empty(agents["arrival_time"].to_numpy()))
    safe_times = np.sort(
        agents.loc[agents["safe_completed"].astype(bool), "arrival_time"].to_numpy(dtype=np.float64)
    )
    safe_times = safe_times[np.isfinite(safe_times)]
    center_times = np.sort(_finite_or_empty(traj["center_entry_times"]))
    rows = []
    for step, time_value in enumerate(t):
        rows.append(
            {
                "step": step,
                "t": float(time_value),
                "cumulative_arrivals": int(np.searchsorted(arrivals, time_value, side="right")),
                "cumulative_center_visits": int(np.searchsorted(center_times, time_value, side="right")),
                "cumulative_safe_completions_known_later": int(
                    np.searchsorted(safe_times, time_value, side="right")
                ),
            }
        )
    return pd.DataFrame(rows)


def _finite_or_empty(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _min_distance_curve_frame(traj: dict) -> pd.DataFrame:
    """Fixture deliberately writes a placeholder column; the renderer must recompute from the NPZ."""
    t = np.asarray(traj["t"], dtype=np.float64)[:-1]
    rows = [
        {"step": step, "t": float(time_value), "min_distance": 123.0,
         "violating_pair_count": 0, "boundary_pair_count": 0}
        for step, time_value in enumerate(t)
    ]
    return pd.DataFrame(rows)


def _violation_events_frame(profile: dict) -> pd.DataFrame:
    rows = []
    for pair_id, agent in enumerate(sorted(profile["violated_ids"])):
        rows.append(
            {
                "pair_id": pair_id,
                "agent_i": int(agent),
                "agent_j": int((agent + 1) % N_AGENTS),
                "start_time": 0.5,
                "end_time": 0.7,
                "duration": 0.2,
                "min_distance": 0.42,
                "min_distance_time": 0.6,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["pair_id", "agent_i", "agent_j", "start_time", "end_time", "duration",
                 "min_distance", "min_distance_time"],
    )


def _metrics_payload(
    scenario: str, model: str, seed: int, profile: dict, agents: pd.DataFrame, traj: dict
) -> dict:
    n = N_AGENTS
    n_arrived = len(profile["arrived_ids"])
    n_safe = len(profile["safe_ids"])
    n_center = int(np.isfinite(traj["center_entry_times"]).sum())
    all_arrived = n_arrived == n
    all_arrival_times = _finite_or_empty(agents["arrival_time"].to_numpy())
    all_arrival_time = float(np.max(all_arrival_times)) if all_arrived else None
    violated = bool(profile["violated_ids"])
    wait = agents["wait_duration"].to_numpy(dtype=np.float64)
    post = agents["post_release_travel_time"].to_numpy(dtype=np.float64)
    center_rate = (n_center / n) if scenario == "B" else None
    return {
        "experiment_id": "formal",
        "model": model,
        "scenario": scenario,
        "scene_id": str(seed),
        "seed": int(seed),
        "scene_hash": f"scenehash_{seed}",
        "config_hash": CONFIG_HASH,
        "status": {
            "completed": bool(all_arrived),
            "timeout": bool(not all_arrived),
            "numerical_error": False,
            "valid": True,
        },
        "n_agents": n,
        "arrival_count": n_arrived,
        "arrival_rate": n_arrived / n,
        "completed_all": bool(all_arrived),
        "safe_completion_count": n_safe,
        "safe_completion_rate": n_safe / n,
        "center_crossing_count": n_center if scenario == "B" else None,
        "center_crossing_rate": center_rate,
        "team_safe_success": bool(all_arrived and not violated),
        "min_separation": float(profile["min_sep"]),
        "min_separation_time": 0.55,
        "violating_pair_count": int(len(profile["violated_ids"])),
        "violation_event_count": int(len(profile["violated_ids"])),
        "violation_pair_seconds": 0.2 * len(profile["violated_ids"]),
        "first_violation_time": 0.5 if violated else None,
        "all_arrival_time": all_arrival_time,
        "t90": None,
        "mean_completion_time_limited": float(
            np.mean(np.where(np.isfinite(agents["arrival_time"].to_numpy(dtype=np.float64)),
                             agents["arrival_time"].to_numpy(dtype=np.float64), OBSERVATION_TIME))
        ),
        "average_path_length": float(profile["path_length"]),
        "mean_path_efficiency_excess": float(profile["path_length"] / 39.7 - 1.0),
        "mean_path_over_40_minus_1": float(profile["path_length"] / 40.0 - 1.0),
        "completed_count_for_path_metrics": n_arrived,
        "waiting": {
            "mean_wait": float(np.mean(wait)),
            "total_wait": float(np.sum(wait)),
            "mean_post_release_travel": float(np.nanmean(post)) if np.isfinite(post).any() else None,
        },
        "planning": {
            "mean_ms": 0.42,
            "p95_ms": 0.61,
            "max_ms": 0.95,
            "total_s": 0.03,
            "calls": 1200,
            "per_active_call_ms": 0.42,
        },
        "planner_diagnostics": {
            "sample_exhausted_count": 0,
            "solver_fallback_count": 0,
            "solver_infeasible_count": 0,
            "constraint_violation_max": 0.0,
            "initial_overlap_pairs_total": 0,
            "numerical_error_count": 0,
            "active_planning_calls": 1200,
        },
        "stall": {"stalled_agent_count": 0, "low_progress_mobile_count": 0},
        "uncompleted": {
            "not_released": 0,
            "not_center": int((~np.isfinite(traj["center_entry_times"])).sum()) if scenario == "B" else 0,
            "center_done_not_arrived": int(max(0, n - n_arrived)),
        },
    }


def _write_run(
    results_dir: Path,
    scenario: str,
    model: str,
    seed: int,
    dirs: np.ndarray,
) -> dict:
    profile = _make_profile(scenario, model, seed)
    starts = SCENE_RADIUS * dirs
    goals = -starts
    traj = _make_trajectory(scenario, profile, starts, goals)
    agents = _agents_frame(profile, traj, starts, goals)
    run_dir = results_dir / "formal" / scenario / model / f"scene_{seed}" / CONFIG_HASH
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "trajectories.npz", **traj)
    agents.to_csv(run_dir / "agents.csv", index=False)
    _arrival_curve_frame(traj, agents).to_csv(run_dir / "arrival_curve.csv", index=False)
    _min_distance_curve_frame(traj).to_csv(run_dir / "min_distance_curve.csv", index=False)
    _violation_events_frame(profile).to_csv(run_dir / "violation_events.csv", index=False)
    pd.DataFrame(
        columns=["pair_id", "agent_i", "agent_j", "min_distance", "min_distance_time",
                 "min_time_inside_band"]
    ).to_csv(run_dir / "boundary_contacts.csv", index=False)
    center_rows = [
        {
            "agent_id": agent,
            "center_entry_time": float(traj["center_entry_times"][agent]),
            "center_entry_step": int(traj["center_entry_steps"][agent]),
        }
        for agent in range(N_AGENTS)
        if np.isfinite(traj["center_entry_times"][agent])
    ]
    pd.DataFrame(center_rows, columns=["agent_id", "center_entry_time", "center_entry_step"]).to_csv(
        run_dir / "center_events.csv", index=False
    )
    metrics = _metrics_payload(scenario, model, seed, profile, agents, traj)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True, allow_nan=False)
    provenance = {
        "experiment_id": "formal",
        "scenario": scenario,
        "model": model,
        "scene_id": str(seed),
        "seed": int(seed),
        "scene_file": f"scenes/scene_{seed}.npz",
        "scene_hash": f"scenehash_{seed}",
        "config_hash": CONFIG_HASH,
    }
    with (run_dir / "provenance.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
    config = {
        "schema_version": 1,
        "experiment": {"name": "synthetic_fixture", "observation_time": OBSERVATION_TIME, "dt": DT},
        "simulation": {
            "n_agents": N_AGENTS,
            "radius": 0.5,
            "d_safe": 1.0,
            "d_plan": 1.0,
            "init_sphere_radius": SCENE_RADIUS,
            "v_max": 2.0,
            "goal_tolerance": 0.3,
            "center_tolerance": 0.1,
        },
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
    row = {
        "experiment_id": "formal",
        "model": model,
        "scenario": scenario,
        "scene_id": str(seed),
        "seed": int(seed),
        "scene_hash": f"scenehash_{seed}",
        "config_hash": CONFIG_HASH,
        "status": "completed" if metrics["status"]["completed"] else "timeout",
        "valid": True,
        "arrival_rate": metrics["arrival_rate"],
        "safe_completion_rate": metrics["safe_completion_rate"],
        "center_crossing_rate": metrics["center_crossing_rate"],
        "min_separation": metrics["min_separation"],
        "violating_pair_count": metrics["violating_pair_count"],
        "violation_event_count": metrics["violation_event_count"],
        "all_arrival_time": metrics["all_arrival_time"],
        "t90": None,
        "average_path_length": metrics["average_path_length"],
        "completed": metrics["status"]["completed"],
        "team_safe_success": metrics["team_safe_success"],
        "planning_time_mean": metrics["planning"]["mean_ms"],
        "planning_time_p95": metrics["planning"]["p95_ms"],
        "sample_exhausted_count": 0,
        "solver_fallback_count": 0,
    }
    return row


def _write_aggregates(results_dir: Path, rows: list[dict]) -> None:
    exp_dir = results_dir / "formal"
    exp_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    comparison.to_csv(exp_dir / "comparison.csv", index=False)

    summary_rows = []
    for (scenario, model), group in comparison.groupby(["scenario", "model"], sort=True):
        valid = group[group["valid"]]
        rates = valid["safe_completion_rate"].to_numpy(dtype=np.float64)
        mean, lo, hi, n = _bootstrap_mean_ci(rates)
        summary_rows.append(
            {
                "experiment_id": "formal",
                "scenario": scenario,
                "model": model,
                "n_runs": int(len(group)),
                "n_valid": int(len(valid)),
                "n_completed": int(valid["completed"].sum()),
                "completed_rate": float(valid["completed"].mean()),
                "team_safe_success_rate": float(valid["team_safe_success"].mean()),
                "safe_completion_rate_n": n,
                "safe_completion_rate_mean": mean,
                "safe_completion_rate_sd": float(np.std(rates, ddof=1)) if rates.size > 1 else None,
                "safe_completion_rate_median": float(np.median(rates)) if rates.size else None,
                "safe_completion_rate_ci_low": lo,
                "safe_completion_rate_ci_high": hi,
            }
        )
    pd.DataFrame(summary_rows).to_csv(exp_dir / "summary_statistics.csv", index=False)

    pair_rows = []
    for scenario in ("A", "B"):
        model_pairs = [
            ("orca_release", "orca"),
            ("orca", "vo"),
            ("orca", "rvo"),
            ("orca", "direct"),
        ]
        for model_a, model_b in model_pairs:
            a = comparison[
                (comparison["scenario"] == scenario) & (comparison["model"] == model_a) & comparison["valid"]
            ]
            b = comparison[
                (comparison["scenario"] == scenario) & (comparison["model"] == model_b) & comparison["valid"]
            ]
            if a.empty or b.empty:
                continue
            merged = a.set_index("scene_id")[["safe_completion_rate"]].join(
                b.set_index("scene_id")[["safe_completion_rate"]], lsuffix="_a", rsuffix="_b", how="inner"
            )
            diffs = (
                merged["safe_completion_rate_a"].to_numpy(dtype=np.float64)
                - merged["safe_completion_rate_b"].to_numpy(dtype=np.float64)
            )
            mean, lo, hi, n = _bootstrap_diff_ci(diffs)
            pair_rows.append(
                {
                    "experiment_id": "formal",
                    "scenario": scenario,
                    "metric": "safe_completion_rate",
                    "pair": f"{model_a} - {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": n,
                    "mean_diff": mean,
                    "sd_diff": float(np.std(diffs, ddof=1)) if n > 1 else None,
                    "median_diff": float(np.median(diffs)) if n else None,
                    "ci_low": lo,
                    "ci_high": hi,
                    "fraction_positive": float(np.mean(diffs > 0)) if n else None,
                    "fraction_zero": float(np.mean(diffs == 0)) if n else None,
                }
            )
    pd.DataFrame(pair_rows).to_csv(exp_dir / "paired_effects.csv", index=False)


@pytest.fixture(scope="session")
def fake_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("viz_synthetic")
    dirs = _directions(N_AGENTS)
    for seed in SCENE_SEEDS:
        _build_scene(root / "scenes" / f"scene_{seed}.npz", str(seed), int(seed), dirs)
    results_dir = root / "results"
    rows: list[dict] = []
    for scenario, models in SCENARIO_MODELS.items():
        for model in models:
            for seed in SCENE_SEEDS:
                rows.append(_write_run(results_dir, scenario, model, seed, dirs))
    _write_aggregates(results_dir, rows)
    return root


@pytest.fixture(scope="session")
def rendered(fake_tree: Path) -> dict:
    out_dir = fake_tree / "figures"
    fignums_before = plt.get_fignums()
    files_before = {
        str(path.relative_to(fake_tree))
        for path in fake_tree.rglob("*")
        if path.is_file() and out_dir not in path.parents
    }
    started = time.perf_counter()
    outputs = render_all(fake_tree / "results", out_dir)
    elapsed = time.perf_counter() - started
    fignums_after = plt.get_fignums()
    files_after = {
        str(path.relative_to(fake_tree))
        for path in fake_tree.rglob("*")
        if path.is_file() and out_dir not in path.parents
    }
    return {
        "out_dir": out_dir,
        "outputs": [Path(p) for p in outputs],
        "elapsed": elapsed,
        "fignums_before": fignums_before,
        "fignums_after": fignums_after,
        "files_before": files_before,
        "files_after": files_after,
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_interval_min_separation_uses_continuous_closest_approach() -> None:
    """handoff §8 的擦碰例：两端距离 1.01 m，区间内连续最小距离恰为 0.99 m。"""
    dt = 0.1
    positions = np.array(
        [
            [[-0.1, 0.495, 0.0], [0.1, -0.495, 0.0]],
            [[0.1, 0.495, 0.0], [-0.1, -0.495, 0.0]],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    inter = interval_min_separation(positions, velocities, dt)
    assert inter.shape == (1,)
    assert inter[0] == pytest.approx(0.99, abs=1e-12)
    endpoint_min = min(
        float(np.linalg.norm(positions[0][0] - positions[0][1])),
        float(np.linalg.norm(positions[1][0] - positions[1][1])),
    )
    assert endpoint_min > 1.0  # endpoint distances would wrongly hide the inside-interval violation
    assert inter[0] < endpoint_min


def test_interval_min_separation_zero_relative_velocity() -> None:
    """相对速度为零的区间退化为静态距离（不允许除零）。"""
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    velocities = np.zeros((1, 2, 3), dtype=np.float64)
    values = interval_min_separation(positions, velocities, 0.1)
    assert values[0] == pytest.approx(0.3, abs=1e-12)


def test_render_all_produces_all_nine_figure_types(rendered: dict) -> None:
    outputs = rendered["outputs"]
    names = [path.name for path in outputs]
    for prefix in (
        "fig01_", "fig02_", "fig03_", "fig04_", "fig05_",
        "fig06_", "fig07_", "fig08_", "fig09_",
    ):
        matches = [name for name in names if name.startswith(prefix)]
        assert matches, f"missing figure type {prefix}: {names}"
    for path in outputs:
        assert path.exists(), f"missing output file: {path}"
        assert path.stat().st_size > 0, f"empty output file: {path}"
    assert len(outputs) >= 15
    assert rendered["elapsed"] < 55.0, f"render_all too slow for smoke test: {rendered['elapsed']:.1f}s"


def test_render_all_writes_only_inside_out_dir(rendered: dict) -> None:
    out_dir = rendered["out_dir"].resolve()
    for path in rendered["outputs"]:
        resolved = Path(path).resolve()
        assert resolved == out_dir or out_dir in resolved.parents, f"output outside out_dir: {resolved}"


def test_render_all_does_not_touch_input_tree(rendered: dict) -> None:
    assert rendered["files_before"] == rendered["files_after"], (
        "render_all modified files outside out_dir: "
        f"added={sorted(rendered['files_after'] - rendered['files_before'])} "
        f"removed={sorted(rendered['files_before'] - rendered['files_after'])}"
    )


def test_render_all_closes_matplotlib_figures(rendered: dict) -> None:
    assert rendered["fignums_after"] == rendered["fignums_before"], (
        f"matplotlib figures leaked: before={rendered['fignums_before']} after={rendered['fignums_after']}"
    )
    assert plt.get_fignums() == []


def test_render_all_creates_required_animations(rendered: dict) -> None:
    out_dir = rendered["out_dir"]
    for filename in (
        "anim_orca_B_seed1000.gif",
        "anim_orca_release_B_seed1000.gif",
    ):
        path = out_dir / filename
        assert path.exists(), f"missing required animation {filename}"
        assert path.stat().st_size > 0
        assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_median_case_seed_rule(rendered: dict) -> None:
    selection = json.loads((rendered["out_dir"] / "selection.json").read_text(encoding="utf-8"))
    assert selection["display_seed_used"] == DISPLAY_SEED
    median = selection["median_case"]
    assert median["selected_seed"] == 1001, median
    assert median["median_rate"] == pytest.approx(0.5, abs=1e-12)
    assert median["candidates"] == {
        "1000": pytest.approx(1.0 / 6.0),
        "1001": pytest.approx(3.0 / 6.0),
        "1002": pytest.approx(5.0 / 6.0),
    }
    assert "ties broken by smallest seed" in median["rule"]
    names = {path.name for path in rendered["outputs"]}
    assert "fig02_trajectories_median_seed1001.png" in names
    assert "fig03_min_separation_median_seed1001.png" in names
    assert "fig09_animation_snapshots_median_seed1001.png" in names
    assert "anim_orca_B_seed1001.gif" in names
    assert "anim_orca_release_B_seed1001.gif" in names


def test_render_scene_seed_outputs_and_closes_figures(fake_tree: Path) -> None:
    out_dir = fake_tree / "scene_seed_render"
    before = plt.get_fignums()
    outputs = render_scene_seed("B", 1000, fake_tree / "results", out_dir, models=["vo", "orca"])
    after = plt.get_fignums()
    names = {path.name for path in outputs}
    assert before == after, f"figure leak: {before} -> {after}"
    assert plt.get_fignums() == []
    assert "fig02_trajectories_seed1000.png" in names
    assert "fig03_min_separation_seed1000.png" in names
    assert "fig07_center_events_seed1000.png" in names
    assert "anim_orca_B_seed1000.gif" in names
    for path in outputs:
        assert path.exists() and path.stat().st_size > 0
        assert out_dir.resolve() in Path(path).resolve().parents or Path(path).resolve().parent == out_dir.resolve()


def test_render_animation_standalone(fake_tree: Path) -> None:
    run_dir = fake_tree / "results" / "formal" / "B" / "orca" / "scene_1000" / CONFIG_HASH
    out_path = fake_tree / "anim_standalone" / "anim_orca_B_seed1000.gif"
    before = plt.get_fignums()
    path = render_animation(run_dir, out_path, fps=10)
    after = plt.get_fignums()
    assert before == after
    assert plt.get_fignums() == []
    assert path == out_path
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_render_animation_missing_trajectories_raises(fake_tree: Path) -> None:
    empty_run = fake_tree / "empty_run"
    empty_run.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError):
        render_animation(empty_run, fake_tree / "anim_empty.gif", fps=5)


def test_render_all_missing_data_graceful(fake_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """只给一个缺 trajectories.npz 的 run + 无聚合 CSV：必须 warning 后继续，不抛异常。"""
    partial_root = tmp_path / "partial"
    src_run = fake_tree / "results" / "formal" / "B" / "orca" / "scene_1000" / CONFIG_HASH
    dst_run = partial_root / "results" / "formal" / "B" / "orca" / "scene_1000" / CONFIG_HASH
    dst_run.mkdir(parents=True)
    for name in ("metrics.json", "provenance.json", "config.json", "agents.csv", "center_events.csv"):
        shutil.copyfile(src_run / name, dst_run / name)
    (partial_root / "scenes").mkdir()
    shutil.copyfile(
        fake_tree / "scenes" / "scene_1000.npz",
        partial_root / "scenes" / "scene_1000.npz",
    )
    out_dir = tmp_path / "partial_figures"
    before = plt.get_fignums()
    outputs = render_all(partial_root / "results", out_dir)
    after = plt.get_fignums()
    captured = capsys.readouterr()
    assert before == after
    assert plt.get_fignums() == []
    assert "WARNING" in captured.err
    assert outputs, "render_all should still produce aggregate figures from the remaining run"
    for path in outputs:
        assert path.exists() and path.stat().st_size > 0
        assert out_dir.resolve() in Path(path).resolve().parents


def test_render_all_empty_results_graceful(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    before = plt.get_fignums()
    outputs = render_all(tmp_path / "does_not_exist", tmp_path / "figs_empty")
    after = plt.get_fignums()
    captured = capsys.readouterr()
    assert outputs == []
    assert before == after
    assert "WARNING" in captured.err


def test_render_scene_seed_unknown_seed_graceful(fake_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out_dir = tmp_path / "figs_unknown"
    before = plt.get_fignums()
    outputs = render_scene_seed("B", 424242, fake_tree / "results", out_dir)
    after = plt.get_fignums()
    captured = capsys.readouterr()
    assert outputs == []
    assert before == after
    assert "WARNING" in captured.err
