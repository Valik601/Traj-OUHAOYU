"""实验编排：生成情景矩阵、批量仿真、落盘轨迹 / 指标 / 解析预测。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    analytic_predictions,
    draw_initial_conditions,
    load_config,
    required_clearance,
    write_json,
)
from .simulate import RunResult, RunSpec, simulate

MAIN_MODELS = ("sfm", "wiedemann")
MAIN_RULES = ("R0", "R1")


def build_specs(cfg: dict[str, Any], initial_conditions: list[dict[str, Any]]) -> list[RunSpec]:
    """构造实验矩阵。"""
    widths = [float(w) for w in cfg["road"]["widths_m"]]
    sweep_widths = [float(w) for w in cfg["road"]["sweep_widths_m"]]
    specs: list[RunSpec] = []

    # 1) 主实验：两种宽度 × 两种规则 × 两种模型
    for width in widths:
        for rule in MAIN_RULES:
            for model in MAIN_MODELS:
                specs.append(
                    RunSpec(
                        run_id=f"L{width:.1f}_{rule}_{model}",
                        width_m=width,
                        rule=rule,
                        model=model,
                        initial_conditions=initial_conditions,
                        save_trace=True,
                    )
                )

    # 2) 单因素消融（只在窄路 L=3.0）
    for rule in ("A1", "A2"):
        for model in MAIN_MODELS:
            specs.append(
                RunSpec(
                    run_id=f"L3.0_{rule}_{model}",
                    width_m=3.0,
                    rule=rule,
                    model=model,
                    initial_conditions=initial_conditions,
                    save_trace=True,
                )
            )

    # 3) 宽度扫描（补足中间宽度）
    for width in sweep_widths:
        if any(abs(width - w) < 1e-9 for w in widths):
            continue
        for rule in MAIN_RULES:
            for model in MAIN_MODELS:
                specs.append(
                    RunSpec(
                        run_id=f"L{width:.1f}_{rule}_{model}",
                        width_m=width,
                        rule=rule,
                        model=model,
                        initial_conditions=initial_conditions,
                        save_trace=False,
                    )
                )
    return specs


def run_all(
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """运行完整实验并写出所有产物，返回汇总字典。"""
    cfg = load_config(config_path)
    project_root = Path(__file__).resolve().parents[1]
    out = Path(out_dir) if out_dir is not None else project_root / "outputs"
    (out / "trajectories").mkdir(parents=True, exist_ok=True)

    initial_conditions = draw_initial_conditions(cfg)
    specs = build_specs(cfg, initial_conditions)

    results: list[RunResult] = []
    ped_rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        res, trace, ped_metrics = simulate(spec, cfg)
        if trace is not None:
            trace_path = out / "trajectories" / f"{spec.run_id}.csv"
            trace.to_csv(trace_path, index=False, float_format="%.5f")
            res.trace_path = str(trace_path.relative_to(out.parent))
        results.append(res)
        ped_rows.extend(ped_metrics)
        if verbose:
            delay = "n/a" if res.car_delay_s is None else f"{res.car_delay_s:+.1f}s"
            print(
                f"[{i:2d}/{len(specs)}] {spec.run_id:22s} {res.outcome:8s} "
                f"car_delay={delay:>8s} min_clear={res.min_pass_clearance_m:+.3f} "
                f"stops={res.car_stop_count} body_d={res.min_body_distance_m:+.3f}"
            )

    metrics_df = pd.DataFrame([r.as_dict() for r in results])
    ped_df = pd.DataFrame(ped_rows)
    metrics_df.to_csv(out / "metrics.csv", index=False, float_format="%.6f")
    ped_df.to_csv(out / "ped_metrics.csv", index=False, float_format="%.6f")

    predictions = analytic_predictions(cfg, [float(w) for w in cfg["road"]["widths_m"]])
    write_json(predictions, out / "predictions.json")
    write_json(initial_conditions, out / "initial_conditions.json")

    # 一致性检查：解析预测（只含几何 + 规则阈值）vs 仿真结果
    consistency = _check_predictions(cfg, metrics_df, predictions)
    write_json(consistency, out / "prediction_consistency.json")

    summary = _build_summary(cfg, metrics_df, ped_df, consistency, initial_conditions)
    write_json(summary, out / "summary.json")
    return summary


def _check_predictions(
    cfg: dict[str, Any], metrics: pd.DataFrame, predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """检查“解析可通行判据”是否与动力学仿真的通过/僵持结果一致。"""
    rows = []
    ok = True
    for pred in predictions:
        width = float(pred["width_m"])
        for rule in MAIN_RULES:
            for model in MAIN_MODELS:
                sub = metrics[
                    (metrics["width_m"].round(6) == round(width, 6))
                    & (metrics["rule"] == rule)
                    & (metrics["model"] == model)
                ]
                if sub.empty:
                    continue
                observed_pass = sub.iloc[0]["outcome"] == "PASS"
                predicted_pass = bool(pred[f"{rule}_feasible"])
                agree = observed_pass == predicted_pass
                ok = ok and agree
                rows.append(
                    {
                        "width_m": width,
                        "rule": rule,
                        "model": model,
                        "predicted_pass": predicted_pass,
                        "observed_pass": bool(observed_pass),
                        "agree": bool(agree),
                    }
                )
    return {
        "all_agree": bool(ok),
        "n_checked": len(rows),
        "rows": rows,
        "note": (
            "解析判据 c_avail = L/2 - Wc/2 - Wp，只比较“几何 + 规则阈值”"
            "所决定的可通行性，不含动力学延误。"
        ),
    }


def _build_summary(
    cfg: dict[str, Any],
    metrics: pd.DataFrame,
    ped: pd.DataFrame,
    consistency: dict[str, Any],
    initial_conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    """整理报告要用的关键数字（全部来自本次真实运行）。"""
    def pick(run_id: str, col: str):
        sub = metrics[metrics["run_id"] == run_id]
        return None if sub.empty else sub.iloc[0][col]

    key_runs = [
        "L4.0_R0_sfm", "L4.0_R0_wiedemann",
        "L3.0_R0_sfm", "L3.0_R0_wiedemann",
        "L3.5_R0_sfm", "L3.5_R0_wiedemann",
        "L2.8_R0_sfm", "L2.8_R0_wiedemann",
        "L3.0_R1_sfm", "L3.0_R1_wiedemann",
        "L2.8_R1_sfm", "L2.8_R1_wiedemann",
        "L3.0_A1_sfm", "L3.0_A1_wiedemann",
        "L3.0_A2_sfm", "L3.0_A2_wiedemann",
        "L2.6_R1_wiedemann", "L2.6_R0_wiedemann",
        "L2.9_R1_wiedemann", "L3.1_R1_wiedemann", "L3.2_R1_wiedemann",
        "L2.9_R0_wiedemann", "L3.1_R0_wiedemann", "L3.2_R0_wiedemann",
    ]
    rows = []
    for rid in key_runs:
        sub = metrics[metrics["run_id"] == rid]
        if sub.empty:
            continue
        r = sub.iloc[0].to_dict()
        rows.append({k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()})
    return {
        "meta": {
            "name": cfg["meta"]["name"],
            "version": cfg["meta"]["version"],
            "seed": cfg["meta"]["seed"],
            "dt_s": cfg["meta"]["dt_s"],
            "t_max_s": cfg["meta"]["t_max_s"],
            "n_runs": int(len(metrics)),
        },
        "initial_conditions": initial_conditions,
        "prediction_consistency": consistency,
        "widths": sorted(metrics["width_m"].unique().tolist()),
        "outcomes": metrics[["run_id", "width_m", "rule", "model", "outcome"]].to_dict(
            orient="records"
        ),
        "key_runs": rows,
        "notes": [
            "所有数字均由本仓库 run_experiments.py 真实运行产生。",
            "宽度是唯一被改变的道路参数；行人初始位置/速度、车辆起点/速度、"
            "模型与规则参数在所有宽度间完全一致。",
            "初始条件由固定种子一次抽取，写入 initial_conditions.json，仿真中冻结。",
        ],
    }

