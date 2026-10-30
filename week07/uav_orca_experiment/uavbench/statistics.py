"""跨场景统计、汇总表与配对效应（handoff §7.3、§9.3）。

独立重复单位是场景（不是 100 架无人机或 4950 个机对）。所有区间为场景级配对 bootstrap
百分位区间（默认 10000 次，固定统计种子）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config, load_config

COMPARISON_COLUMNS = [
    "experiment_id",
    "model",
    "scenario",
    "scene_id",
    "seed",
    "scene_hash",
    "config_hash",
    "status",
    "valid",
    "arrival_rate",
    "safe_completion_rate",
    "center_crossing_rate",
    "min_separation",
    "violating_pair_count",
    "violation_event_count",
    "all_arrival_time",
    "t90",
    "average_path_length",
    "completed",
    "team_safe_success",
    "planning_time_mean",
    "planning_time_p95",
    "sample_exhausted_count",
    "solver_fallback_count",
]

#: 跨场景汇总的数值指标（其余为计数/布尔）
SUMMARY_METRICS = [
    "arrival_rate",
    "safe_completion_rate",
    "center_crossing_rate",
    "min_separation",
    "violating_pair_count",
    "violation_event_count",
    "all_arrival_time",
    "t90",
    "average_path_length",
    "planning_time_mean",
    "planning_time_p95",
    "sample_exhausted_count",
    "solver_fallback_count",
]

#: 配对比较定义：(label, model_a, model_b, scenario, metric)
FORMAL_PAIRS: list[tuple[str, str, str, str, str]] = [
    ("orca_release - orca", "orca_release", "orca", "B", "safe_completion_rate"),
    ("orca - vo", "orca", "vo", "B", "safe_completion_rate"),
    ("orca - rvo", "orca", "rvo", "B", "safe_completion_rate"),
    ("orca_release - direct_release", "orca_release", "direct_release", "B", "safe_completion_rate"),
    ("orca - direct_release", "orca", "direct_release", "B", "safe_completion_rate"),
    ("direct_release - direct", "direct_release", "direct", "B", "safe_completion_rate"),
    ("orca - direct", "orca", "direct", "B", "safe_completion_rate"),
    ("orca - vo", "orca", "vo", "A", "safe_completion_rate"),
    ("orca - rvo", "orca", "rvo", "A", "safe_completion_rate"),
    ("vo - direct", "vo", "direct", "A", "safe_completion_rate"),
    ("orca - direct", "orca", "direct", "A", "safe_completion_rate"),
    ("orca_release - orca", "orca_release", "orca", "B", "min_separation"),
    ("orca - vo", "orca", "vo", "B", "min_separation"),
]


# ---------------------------------------------------------------------------
# 发现与读取结果
# ---------------------------------------------------------------------------
def discover_run_dirs(results_dir: Path, experiment_id: str | None = None) -> list[Path]:
    """按 ``results/<exp>/<scenario>/<model>/scene_<id>/<config_hash>`` 结构发现运行目录。"""
    results_dir = Path(results_dir)
    found: list[Path] = []
    if experiment_id is not None:
        roots = [results_dir / experiment_id]
    else:
        roots = [p for p in results_dir.iterdir() if p.is_dir()] if results_dir.exists() else []
    for root in roots:
        if not root.exists():
            continue
        for metrics in sorted(root.rglob("metrics.json")):
            found.append(metrics.parent)
    return found


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_run_row(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    prov_path = run_dir / "provenance.json"
    if not (metrics_path.exists() and prov_path.exists()):
        return None
    metrics = _load_json(metrics_path)
    prov = _load_json(prov_path)
    status = metrics.get("status") or {}
    if status.get("numerical_error"):
        status_str = "numerical_error"
    elif status.get("completed"):
        status_str = "completed"
    else:
        status_str = "timeout"
    planning = metrics.get("planning") or {}
    diag = metrics.get("planner_diagnostics") or {}
    row = {
        "experiment_id": prov.get("experiment_id") or metrics.get("experiment_id"),
        "model": prov.get("model") or metrics.get("model"),
        "scenario": prov.get("scenario") or metrics.get("scenario"),
        "scene_id": prov.get("scene_id") or metrics.get("scene_id"),
        "seed": prov.get("seed", metrics.get("seed")),
        "scene_hash": metrics.get("scene_hash"),
        "config_hash": metrics.get("config_hash") or prov.get("config_hash"),
        "status": status_str,
        "valid": bool(status.get("valid", True)),
        "arrival_rate": metrics.get("arrival_rate"),
        "safe_completion_rate": metrics.get("safe_completion_rate"),
        "center_crossing_rate": metrics.get("center_crossing_rate"),
        "min_separation": metrics.get("min_separation"),
        "violating_pair_count": metrics.get("violating_pair_count"),
        "violation_event_count": metrics.get("violation_event_count"),
        "all_arrival_time": metrics.get("all_arrival_time"),
        "t90": metrics.get("t90"),
        "average_path_length": metrics.get("average_path_length"),
        "completed": bool(status.get("completed")),
        "team_safe_success": bool(metrics.get("team_safe_success")),
        "planning_time_mean": planning.get("mean_ms"),
        "planning_time_p95": planning.get("p95_ms"),
        "sample_exhausted_count": diag.get("sample_exhausted_count"),
        "solver_fallback_count": diag.get("solver_fallback_count"),
        "run_dir": str(run_dir),
    }
    if not row["experiment_id"]:
        # 从目录结构兜底推断
        row["experiment_id"] = run_dir.parents[4].name
    return row


def load_runs(
    results_dir: Path, experiment_ids: Iterable[str] | None = None
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows: list[dict[str, Any]] = []
    if experiment_ids is None:
        run_dirs = discover_run_dirs(results_dir)
    else:
        run_dirs = []
        for exp in experiment_ids:
            run_dirs.extend(discover_run_dirs(results_dir, str(exp)))
    for d in run_dirs:
        row = read_run_row(d)
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS + ["run_dir"])
    order = {c: i for i, c in enumerate(COMPARISON_COLUMNS)}
    cols = [c for c in COMPARISON_COLUMNS if c in df.columns] + [
        c for c in df.columns if c not in order
    ]
    df = df[cols]
    return df.sort_values(["experiment_id", "scenario", "model", "scene_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    values: Iterable[float], *, resamples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    if n == 1:
        return (mean, mean, mean, 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(resamples), n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (mean, lo, hi, n)


def _bootstrap_cfg(cfg: Config | None) -> tuple[int, int]:
    if cfg is None:
        cfg = load_config()
    return int(cfg.metrics.bootstrap_resamples), int(cfg.metrics.bootstrap_seed)


# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------
def summary_statistics(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    resamples, seed = _bootstrap_cfg(cfg)
    rows = []
    for (exp, scen, model), group in df.groupby(["experiment_id", "scenario", "model"], sort=True):
        valid = group[group["valid"]]
        row: dict[str, Any] = {
            "experiment_id": exp,
            "scenario": scen,
            "model": model,
            "n_runs": int(len(group)),
            "n_valid": int(len(valid)),
            "n_completed": int(valid["completed"].sum()),
            "completed_rate": float(valid["completed"].mean()) if len(valid) else None,
            "team_safe_success_rate": float(valid["team_safe_success"].mean()) if len(valid) else None,
        }
        for metric in SUMMARY_METRICS:
            if metric not in valid.columns:
                continue
            vals = pd.to_numeric(valid[metric], errors="coerce")
            n_defined = int(vals.notna().sum())
            mean, lo, hi, n = bootstrap_mean_ci(vals.dropna(), resamples=resamples, seed=seed)
            row[f"{metric}_n"] = n_defined
            row[f"{metric}_mean"] = mean if n else None
            row[f"{metric}_sd"] = float(vals.std(ddof=1)) if n_defined > 1 else None
            row[f"{metric}_median"] = float(vals.median()) if n_defined else None
            row[f"{metric}_ci_low"] = lo if n else None
            row[f"{metric}_ci_high"] = hi if n else None
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, str, str, str]] | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """场景级配对差（只在两模型都 valid 的同一场景上配对）。"""
    resamples, seed = _bootstrap_cfg(cfg)
    pairs = pairs if pairs is not None else FORMAL_PAIRS
    rows: list[dict[str, Any]] = []
    for label, model_a, model_b, scenario, metric in pairs:
        a = df[(df["scenario"] == scenario) & (df["model"] == model_a) & (df["valid"])]
        b = df[(df["scenario"] == scenario) & (df["model"] == model_b) & (df["valid"])]
        if a.empty or b.empty:
            continue
        merged = a.set_index("scene_id")[[metric]].join(
            b.set_index("scene_id")[[metric]], lsuffix="_a", rsuffix="_b", how="inner"
        )
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        if col_a not in merged.columns or col_b not in merged.columns:
            continue
        va = pd.to_numeric(merged[col_a], errors="coerce")
        vb = pd.to_numeric(merged[col_b], errors="coerce")
        both = va.notna() & vb.notna()
        diff = (va[both] - vb[both]).to_numpy(dtype=np.float64)
        if diff.size == 0:
            continue
        rng = np.random.default_rng(seed)
        boots = diff[rng.integers(0, diff.size, size=(resamples, diff.size))].mean(axis=1)
        rows.append(
            {
                "experiment_id": a["experiment_id"].iloc[0],
                "scenario": scenario,
                "metric": metric,
                "pair": label,
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": int(diff.size),
                "mean_diff": float(np.mean(diff)),
                "sd_diff": float(np.std(diff, ddof=1)) if diff.size > 1 else None,
                "median_diff": float(np.median(diff)),
                "ci_low": float(np.percentile(boots, 2.5)),
                "ci_high": float(np.percentile(boots, 97.5)),
                "fraction_positive": float(np.mean(diff > 0)),
                "fraction_zero": float(np.mean(diff == 0)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 敏感性 block 汇总
# ---------------------------------------------------------------------------
def sensitivity_experiment_ids(results_dir: Path) -> list[str]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return sorted(p.name for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("sensitivity_"))


def sensitivity_effects(
    sensitivity_df: pd.DataFrame, formal_df: pd.DataFrame, cfg: Config | None = None
) -> pd.DataFrame:
    """敏感性变体 vs 正式基线的场景级配对差（同一 model/scene）。"""
    resamples, seed = _bootstrap_cfg(cfg)
    rows: list[dict[str, Any]] = []
    formal_valid = formal_df[formal_df["valid"] & (formal_df["experiment_id"] == "formal")]
    if formal_valid.empty or sensitivity_df.empty:
        return pd.DataFrame(rows)
    metrics = ["safe_completion_rate", "arrival_rate", "min_separation", "violation_event_count"]
    for exp, group in sensitivity_df.groupby("experiment_id"):
        for (scen, model), sub in group.groupby(["scenario", "model"]):
            base = formal_valid[(formal_valid["scenario"] == scen) & (formal_valid["model"] == model)]
            if base.empty:
                continue
            merged = sub.set_index("scene_id").join(
                base.set_index("scene_id"), lsuffix="_var", rsuffix="_base", how="inner"
            )
            if merged.empty:
                continue
            for metric in metrics:
                ca, cb = f"{metric}_var", f"{metric}_base"
                if ca not in merged.columns or cb not in merged.columns:
                    continue
                va = pd.to_numeric(merged[ca], errors="coerce")
                vb = pd.to_numeric(merged[cb], errors="coerce")
                both = va.notna() & vb.notna()
                diff = (va[both] - vb[both]).to_numpy(dtype=np.float64)
                if diff.size == 0:
                    continue
                rng = np.random.default_rng(seed)
                boots = diff[rng.integers(0, diff.size, size=(resamples, diff.size))].mean(axis=1)
                rows.append(
                    {
                        "experiment_id": exp,
                        "scenario": scen,
                        "metric": metric,
                        "pair": f"{exp} - baseline",
                        "model_a": model,
                        "model_b": model,
                        "n_pairs": int(diff.size),
                        "mean_diff": float(np.mean(diff)),
                        "sd_diff": float(np.std(diff, ddof=1)) if diff.size > 1 else None,
                        "median_diff": float(np.median(diff)),
                        "ci_low": float(np.percentile(boots, 2.5)),
                        "ci_high": float(np.percentile(boots, 97.5)),
                        "fraction_positive": float(np.mean(diff > 0)),
                        "fraction_zero": float(np.mean(diff == 0)),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 入口：summarize
# ---------------------------------------------------------------------------
def summarize_results(
    results_dir: Path,
    *,
    experiments: Iterable[str] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    cfg = load_config(config_path) if config_path is not None else load_config()
    df = load_runs(results_dir, experiments)
    outputs: list[str] = []
    if df.empty:
        return {"n_runs": 0, "outputs": outputs, "experiments": []}
    comparison = df[COMPARISON_COLUMNS].copy()
    exps = sorted(df["experiment_id"].dropna().unique().tolist())
    pair_frames: list[pd.DataFrame] = []

    formal = df[df["experiment_id"] == "formal"]
    sensitivity = df[df["experiment_id"].astype(str).str.startswith("sensitivity_")]
    if not formal.empty:
        pair_frames.append(paired_effects(formal, FORMAL_PAIRS, cfg))
    if not sensitivity.empty:
        pair_frames.append(sensitivity_effects(sensitivity, formal if not formal.empty else df, cfg))

    for exp in exps:
        sub = df[df["experiment_id"] == exp]
        if sub.empty:
            continue
        exp_dir = results_dir / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        comp = sub[COMPARISON_COLUMNS]
        comp.to_csv(exp_dir / "comparison.csv", index=False)
        summary = summary_statistics(sub, cfg)
        summary.to_csv(exp_dir / "summary_statistics.csv", index=False)
        blocks = [f for f in pair_frames if not f.empty]
        if blocks:
            comb = pd.concat(blocks, ignore_index=True)
            comb = comb[comb["experiment_id"] == exp]
            if not comb.empty:
                comb.to_csv(exp_dir / "paired_effects.csv", index=False)
        outputs.extend(
            [
                str(exp_dir / "comparison.csv"),
                str(exp_dir / "summary_statistics.csv"),
            ]
        )
    # 全局汇总（全部正式 + 敏感性）
    comparison.to_csv(results_dir / "comparison_all.csv", index=False)
    all_pairs = pd.concat([f for f in pair_frames if not f.empty], ignore_index=True) if pair_frames else pd.DataFrame()
    if not all_pairs.empty:
        all_pairs.to_csv(results_dir / "paired_effects_all.csv", index=False)
        outputs.append(str(results_dir / "paired_effects_all.csv"))
    # 报告用数字快照
    numbers = _numbers_snapshot(df, all_pairs, cfg)
    with (results_dir / "numbers.json").open("w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, sort_keys=True, allow_nan=False)
    outputs.extend(
        [str(results_dir / "comparison_all.csv"), str(results_dir / "numbers.json")]
    )
    return {
        "n_runs": int(len(df)),
        "experiments": exps,
        "outputs": outputs,
    }


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _numbers_snapshot(df: pd.DataFrame, pairs: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    valid = df[df["valid"]]
    snap: dict[str, Any] = {
        "n_runs_total": int(len(df)),
        "n_runs_valid": int(len(valid)),
        "n_runs_invalid": int((~df["valid"]).sum()),
        "experiments": {},
        "paired_effects": [],
    }
    for exp, sub in df.groupby("experiment_id"):
        entry: dict[str, Any] = {}
        for (scen, model), g in sub.groupby(["scenario", "model"]):
            gv = g[g["valid"]]
            entry[f"{scen}:{model}"] = {
                "n_runs": int(len(g)),
                "n_valid": int(len(gv)),
                "n_completed": int(gv["completed"].sum()),
                "arrival_rate_mean": _json_safe(gv["arrival_rate"].mean()),
                "safe_completion_rate_mean": _json_safe(gv["safe_completion_rate"].mean()),
                "safe_completion_rate_sd": _json_safe(gv["safe_completion_rate"].std(ddof=1)),
                "min_separation_min": _json_safe(gv["min_separation"].min()),
                "violating_pair_count_mean": _json_safe(gv["violating_pair_count"].mean()),
                "all_arrival_time_mean": _json_safe(gv["all_arrival_time"].mean()),
                "all_arrival_time_n": int(gv["all_arrival_time"].notna().sum()),
                "average_path_length_mean": _json_safe(gv["average_path_length"].mean()),
                "team_safe_success_rate": _json_safe(gv["team_safe_success"].mean()),
                "planning_time_mean_ms": _json_safe(gv["planning_time_mean"].mean()),
                "sample_exhausted_total": _json_safe(pd.to_numeric(gv["sample_exhausted_count"], errors="coerce").sum()),
                "solver_fallback_total": _json_safe(pd.to_numeric(gv["solver_fallback_count"], errors="coerce").sum()),
            }
        snap["experiments"][exp] = entry
    if not pairs.empty:
        snap["paired_effects"] = [
            {
                "experiment_id": str(r["experiment_id"]),
                "scenario": str(r["scenario"]),
                "metric": str(r["metric"]),
                "pair": str(r["pair"]),
                "n_pairs": int(r["n_pairs"]),
                "mean_diff": _json_safe(r["mean_diff"]),
                "ci_low": _json_safe(r["ci_low"]),
                "ci_high": _json_safe(r["ci_high"]),
                "fraction_positive": _json_safe(r["fraction_positive"]),
            }
            for _, r in pairs.iterrows()
        ]
    return _json_safe(snap)
