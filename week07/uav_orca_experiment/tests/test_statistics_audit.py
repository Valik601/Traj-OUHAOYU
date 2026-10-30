# OWNER: lead
"""统计与审计单元测试（场景级配对 bootstrap 口径）。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from uavbench.audit import compare_metrics
from uavbench.config import load_config
from uavbench.statistics import (
    bootstrap_mean_ci,
    paired_effects,
    sensitivity_effects,
    summary_statistics,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_bootstrap_mean_ci_basic():
    cfg = load_config()
    values = np.full(7, 0.5)
    mean, lo, hi, n = bootstrap_mean_ci(values, resamples=500, seed=7)
    assert n == 7
    assert mean == pytest.approx(0.5)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    mean, lo, hi, n = bootstrap_mean_ci([], resamples=100, seed=1)
    assert n == 0 and np.isnan(mean)
    # 单样本区间退化为点
    mean, lo, hi, n = bootstrap_mean_ci([0.25], resamples=100, seed=1)
    assert mean == lo == hi == 0.25 and n == 1


def test_summary_statistics_conditions():
    cfg = load_config()
    rows = [
        dict(experiment_id="formal", scenario="B", model="orca", scene_id="1", valid=True,
             completed=True, team_safe_success=True, safe_completion_rate=1.0,
             arrival_rate=1.0, center_crossing_rate=1.0, min_separation=1.0,
             violating_pair_count=0, violation_event_count=0, all_arrival_time=60.0,
             t90=59.0, average_path_length=40.0, planning_time_mean=1.0,
             planning_time_p95=2.0, sample_exhausted_count=0, solver_fallback_count=0),
        dict(experiment_id="formal", scenario="B", model="orca", scene_id="2", valid=True,
             completed=False, team_safe_success=False, safe_completion_rate=0.4,
             arrival_rate=0.5, center_crossing_rate=0.9, min_separation=0.5,
             violating_pair_count=120, violation_event_count=130, all_arrival_time=None,
             t90=None, average_path_length=35.0, planning_time_mean=1.5,
             planning_time_p95=None, sample_exhausted_count=2, solver_fallback_count=1),
        dict(experiment_id="formal", scenario="B", model="orca", scene_id="3", valid=False,
             completed=False, team_safe_success=False, safe_completion_rate=None,
             arrival_rate=None, center_crossing_rate=None, min_separation=None,
             violating_pair_count=None, violation_event_count=None, all_arrival_time=None,
             t90=None, average_path_length=None, planning_time_mean=None,
             planning_time_p95=None, sample_exhausted_count=None, solver_fallback_count=None),
    ]
    out = summary_statistics(_df(rows), cfg)
    row = out.iloc[0]
    assert row["n_runs"] == 3
    assert row["n_valid"] == 2
    assert row["n_completed"] == 1
    assert row["safe_completion_rate_n"] == 2
    assert row["safe_completion_rate_mean"] == pytest.approx(0.7)
    assert row["all_arrival_time_n"] == 1
    assert row["all_arrival_time_mean"] == pytest.approx(60.0)
    assert row["t90_n"] == 1
    assert row["team_safe_success_rate"] == pytest.approx(0.5)


def test_paired_effects_scene_level():
    cfg = load_config()
    rows = []
    diffs = [0.0, 0.5, 1.0, -0.5, 0.25]
    for i, diff in enumerate(diffs):
        for model, value in (("orca_release", max(0.0, 0.5 + diff)), ("orca", max(0.0, 0.5))):
            rows.append(
                dict(experiment_id="formal", scenario="B", model=model, scene_id=str(i),
                     valid=True, completed=True, team_safe_success=True,
                     safe_completion_rate=value, arrival_rate=1.0,
                     center_crossing_rate=1.0, min_separation=1.0,
                     violating_pair_count=0, violation_event_count=0,
                     all_arrival_time=60.0, t90=59.0, average_path_length=40.0,
                     planning_time_mean=1.0, planning_time_p95=2.0,
                     sample_exhausted_count=0, solver_fallback_count=0)
            )
    df = _df(rows)
    pairs = [("orca_release - orca", "orca_release", "orca", "B", "safe_completion_rate")]
    out = paired_effects(df, pairs, cfg)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_pairs"] == 5
    assert row["mean_diff"] == pytest.approx(np.mean([max(0, 0.5 + d) - 0.5 for d in diffs]))
    assert row["ci_low"] <= row["mean_diff"] <= row["ci_high"]
    assert row["ci_low"] >= -0.6
    assert row["fraction_positive"] == pytest.approx(np.mean([d > 0 for d in diffs]))


def test_sensitivity_effects_pairs_same_scene():
    cfg = load_config()
    base_rows, sens_rows = [], []
    for i in range(4):
        for model in ("vo",):
            base_rows.append(
                dict(experiment_id="formal", scenario="B", model=model, scene_id=str(i),
                     valid=True, completed=True, team_safe_success=True,
                     safe_completion_rate=0.2, arrival_rate=1.0, min_separation=0.8,
                     violating_pair_count=10, violation_event_count=10,
                     all_arrival_time=100.0, t90=99.0, average_path_length=40.0,
                     planning_time_mean=1.0, planning_time_p95=2.0,
                     sample_exhausted_count=0, solver_fallback_count=0)
            )
            sens_rows.append(
                dict(experiment_id="sensitivity_candidate_dirs_64", scenario="B", model=model,
                     scene_id=str(i), valid=True, completed=True, team_safe_success=True,
                     safe_completion_rate=0.35, arrival_rate=1.0, min_separation=0.7,
                     violating_pair_count=12, violation_event_count=12,
                     all_arrival_time=101.0, t90=100.0, average_path_length=41.0,
                     planning_time_mean=1.0, planning_time_p95=2.0,
                     sample_exhausted_count=1, solver_fallback_count=0)
            )
    out = sensitivity_effects(_df(sens_rows), _df(base_rows), cfg)
    sel = out[(out["experiment_id"] == "sensitivity_candidate_dirs_64") & (out["metric"] == "safe_completion_rate")]
    assert len(sel) == 1
    assert sel.iloc[0]["n_pairs"] == 4
    assert sel.iloc[0]["mean_diff"] == pytest.approx(0.15, abs=1e-12)


def test_compare_metrics_detects_differences():
    base = {
        "arrival_rate": 1.0,
        "violating_pair_count": 0,
        "status": {"completed": True, "valid": True},
        "planning": {"mean_ms": 1.5},
        "test": None,
        "list": [1, 2],
    }
    assert compare_metrics(base, deepcopy(base)) == []
    changed = deepcopy(base)
    changed["arrival_rate"] = 0.5
    assert any("arrival_rate" in p for p in compare_metrics(base, changed))
    changed = deepcopy(base)
    changed["status"]["completed"] = False
    assert any("status.completed" in p for p in compare_metrics(base, changed))
    changed = deepcopy(base)
    changed["planning"]["mean_ms"] = 1.5 + 1e-13
    assert compare_metrics(base, changed) == []  # 容差内不报错
