"""实验 2：速度相关个人空间 / 间隙接受模型（慢车时行人愿意贴近穿行）。

与实验 1 共享同一个 Agent 框架、积分器与指标定义，只新增规则族 S_*：
    - 共享接受关系 c_acc(v) = c0 + k·v
    - 行人：c ≥ c_acc(v_car) → 继续贴着车走（PASS）；
            c <  c_acc(v_car) → 让行，纵向速度降到 0（STOP），横向移向路缘
    - 车辆：v_clear = speed_factor·(c − c0)/k，并提前 car_approach_m 减速到该速度
    - SFM：车辆对行人的斥力按车速缩放（慢车风险低）

本模块负责实验矩阵、解析预测、通过事件提取、边界汇总与产物落盘。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    clearance_available,
    draw_initial_conditions,
    load_config,
    required_clearance,
    rule_clearance_curve,
    write_json,
)
from .simulate import RunResult, RunSpec, simulate

WIDTHS = [2.80, 2.82, 2.84, 2.86, 2.88, 2.90, 2.95, 3.00, 3.10, 3.20, 3.50, 4.00]
RULES = ["R1", "S_cons", "S_mod", "S_lib"]
MODELS = ["sfm", "wiedemann"]
T_MAX_S = 420.0
ABLATION_RUNS = [("S_mod_nf", 3.00, "sfm"), ("S_mod_nf", 2.90, "sfm")]
TRACE_RUNS = {
    ("R1", 3.00, "wiedemann"),
    ("S_cons", 3.00, "wiedemann"),
    ("S_mod", 3.00, "wiedemann"),
    ("S_lib", 3.00, "wiedemann"),
    ("S_mod", 3.00, "sfm"),
    ("S_mod", 2.95, "wiedemann"),
    ("S_mod", 2.90, "wiedemann"),
    ("S_mod", 2.84, "wiedemann"),
    ("S_lib", 2.84, "wiedemann"),
    ("S_lib", 2.82, "wiedemann"),
    ("S_lib", 2.80, "wiedemann"),
}


def run_id(rule: str, width: float, model: str) -> str:
    return f"L{width:.2f}_{rule}_{model}"


# ---------------------------------------------------------------------------
# 解析预测
# ---------------------------------------------------------------------------
def predicted_boundary(cfg: dict[str, Any], rule_name: str) -> dict[str, Any]:
    """解析边界。

    - 速度接受规则（S_*）：行人只要能在路缘得到 c ≥ c0 就愿意贴着车走，因此
      遭遇可以解开的条件是 c_avail ≥ c0（车可以先停、等人走过去，再继续）；
    - 旧 R1：行人用固定 0.30 m 阈值，不会从车侧走过去，必须由车辆以
      v_clear = (c − 0.05)/0.05 > 0 通过，因此条件是 c_avail > 0.05。
    """
    rule = cfg["rules"][rule_name]
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    curve = rule_clearance_curve(rule)
    if rule.get("ped_acceptance") == "speed":
        c_req = float(rule["car_c_min_m"])
        inclusive = True
        note = "c_avail ≥ c0（行人可从停住的车侧走过）"
    else:
        c_req = required_clearance(rule, 0.0)
        inclusive = False
        note = "c_avail > c_min（车辆必须低速通过）"
    l_crit = 2.0 * (c_req + car_w / 2.0 + ped_w)
    return {
        "rule": rule_name,
        "c0_m": None if curve is None else curve[0],
        "k_s_per_m": None if curve is None else curve[1],
        "c_req_m": c_req,
        "inclusive": inclusive,
        "l_crit_m": l_crit,
        "note": note,
    }


def predict_outcome(cfg: dict[str, Any], rule_name: str, width_m: float) -> bool:
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    c_avail = clearance_available(width_m, car_w, ped_w)
    b = predicted_boundary(cfg, rule_name)
    if b["inclusive"]:
        return c_avail >= b["c_req_m"] - 1e-6
    return c_avail > b["c_req_m"] + 1e-6


def allowed_speed(cfg: dict[str, Any], rule_name: str, width_m: float) -> float:
    rule = cfg["rules"][rule_name]
    c_avail = clearance_available(
        width_m, float(cfg["car"]["width_m"]), float(cfg["pedestrians"]["width_m"])
    )
    curve = rule_clearance_curve(rule)
    if curve is None:
        return float("nan")
    c0, k = curve
    return max(0.0, float(rule.get("car_speed_factor", 1.0)) * (c_avail - c0) / k)


# ---------------------------------------------------------------------------
# 实验矩阵
# ---------------------------------------------------------------------------
def build_specs(cfg: dict[str, Any], ics: list[dict[str, Any]]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for width in WIDTHS:
        for rule in RULES:
            for model in MODELS:
                rid = run_id(rule, width, model)
                specs.append(
                    RunSpec(
                        run_id=rid,
                        width_m=width,
                        rule=rule,
                        model=model,
                        initial_conditions=ics,
                        save_trace=(rule, width, model) in TRACE_RUNS,
                        t_max_s=T_MAX_S,
                    )
                )
    for rule, width, model in ABLATION_RUNS:
        specs.append(
            RunSpec(
                run_id=run_id(rule, width, model),
                width_m=width,
                rule=rule,
                model=model,
                initial_conditions=ics,
                save_trace=True,
                t_max_s=T_MAX_S,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# 通过事件：每个行人被车辆驶过的瞬间（车速、余量、行人速度）
# ---------------------------------------------------------------------------
def extract_pass_events(
    run: RunResult, trace: pd.DataFrame, ped_names: list[str], car_w: float, ped_w: float
) -> list[dict[str, Any]]:
    car = trace[trace["agent"] == "CAR"].sort_values("t")
    events: list[dict[str, Any]] = []
    for name in ped_names:
        ped = trace[trace["agent"] == name].sort_values("t")
        if ped.empty or car.empty:
            continue
        n = min(len(car), len(ped))
        d = (car["x"].to_numpy()[:n] - ped["x"].to_numpy()[:n])
        # 车辆由 +x 驶来：取车与行人纵向最接近的时刻（d 由正变负）
        idx = int(np.argmin(np.abs(d)))
        row = {
            "run_id": run.run_id,
            "width_m": run.width_m,
            "rule": run.rule,
            "model": run.model,
            "ped": name,
            "t_pass_s": float(car["t"].to_numpy()[idx]),
            "car_speed_mps": float(car["speed"].to_numpy()[idx]),
            "ped_speed_mps": float(ped["speed"].to_numpy()[idx]),
            "clearance_m": float(
                abs(float(ped["y"].to_numpy()[idx]) - float(car["y"].to_numpy()[idx]))
                - 0.5 * (car_w + ped_w)
            ),
            "gap_m": float(ped["gap_to_car"].to_numpy()[idx]),
            "ped_mode": str(ped["mode"].to_numpy()[idx]),
            "ped_min_speed_mps": float(ped["speed"].min()),
            "ped_stop_time_s": float((ped["speed"] < 0.1).sum() * 0.1),
        }
        events.append(row)
    return events


# ---------------------------------------------------------------------------
# 运行实验 2
# ---------------------------------------------------------------------------
def run_experiment(
    config_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    project_root = Path(__file__).resolve().parents[1]
    out = Path(out_dir) if out_dir is not None else project_root / "outputs_acceptance"
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    ics = draw_initial_conditions(cfg)
    specs = build_specs(cfg, ics)
    results: list[RunResult] = []
    ped_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []

    for i, spec in enumerate(specs, start=1):
        res, trace, ped_metrics = simulate(spec, cfg)
        results.append(res)
        ped_rows.extend(ped_metrics)
        if trace is not None:
            trace_path = out / "trajectories" / f"{spec.run_id}.csv"
            trace.to_csv(trace_path, index=False, float_format="%.5f")
            res.trace_path = str(trace_path.relative_to(out.parent))
            pass_rows.extend(
                extract_pass_events(
                    res,
                    trace,
                    [f"P{k}" for k in range(1, 4)],
                    float(cfg["car"]["width_m"]),
                    float(cfg["pedestrians"]["width_m"]),
                )
            )
        if verbose:
            delay = "n/a" if res.car_delay_s is None else f"{res.car_delay_s:+.1f}s"
            print(
                f"[{i:3d}/{len(specs)}] {spec.run_id:26s} {res.outcome:8s} "
                f"delay={delay:>8s} vmin={res.car_min_speed_mps:5.2f} "
                f"clear={res.min_pass_clearance_m:+.3f} "
                f"ped_stop={res.ped_stop_time_total_s:6.1f} "
                f"body={res.min_body_distance_m:+.3f}"
            )

    metrics = pd.DataFrame([r.as_dict() for r in results])
    ped_df = pd.DataFrame(ped_rows)
    pass_df = pd.DataFrame(pass_rows)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.6f")
    ped_df.to_csv(out / "ped_metrics.csv", index=False, float_format="%.6f")
    if not pass_df.empty:
        pass_df.to_csv(out / "pass_events.csv", index=False, float_format="%.6f")

    # 预测与一致性
    predictions = []
    consistency_rows = []
    for rule in RULES:
        b = predicted_boundary(cfg, rule)
        for width in WIDTHS:
            c_avail = clearance_available(
                width,
                float(cfg["car"]["width_m"]),
                float(cfg["pedestrians"]["width_m"]),
            )
            predictions.append(
                {
                    "rule": rule,
                    "width_m": width,
                    "c_avail_m": round(c_avail, 6),
                    "predicted_pass": bool(predict_outcome(cfg, rule, width)),
                    "c0_m": b["c0_m"],
                    "k_s_per_m": b["k_s_per_m"],
                    "l_crit_m": round(b["l_crit_m"], 6),
                    "allowed_speed_mps": round(allowed_speed(cfg, rule, width), 6),
                }
            )
    for _, r in metrics.iterrows():
        if r["rule"] not in RULES:
            continue
        pred = predict_outcome(cfg, r["rule"], float(r["width_m"]))
        observed = r["outcome"] == "PASS"
        consistency_rows.append(
            {
                "run_id": r["run_id"],
                "width_m": float(r["width_m"]),
                "rule": r["rule"],
                "model": r["model"],
                "predicted_pass": bool(pred),
                "observed_pass": bool(observed),
                "agree": bool(pred == observed),
            }
        )
    consistency = {
        "n_checked": len(consistency_rows),
        "n_agree": int(sum(x["agree"] for x in consistency_rows)),
        "all_agree": all(x["agree"] for x in consistency_rows),
        "rows": consistency_rows,
    }
    write_json(predictions, out / "predictions.json")
    write_json(consistency, out / "prediction_consistency.json")

    summary = _summarize(cfg, metrics, ped_df, pass_df, consistency)
    write_json(summary, out / "summary.json")
    _write_tables(cfg, summary, metrics, out / "tables")
    return summary


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _summarize(
    cfg: dict[str, Any],
    metrics: pd.DataFrame,
    ped: pd.DataFrame,
    pass_events: pd.DataFrame,
    consistency: dict[str, Any],
) -> dict[str, Any]:
    def row(rid: str, col: str):
        sub = metrics[metrics["run_id"] == rid]
        return None if sub.empty else sub.iloc[0][col]

    boundaries = []
    for rule in RULES:
        b = predicted_boundary(cfg, rule)
        for model in MODELS:
            sub = metrics[(metrics["rule"] == rule) & (metrics["model"] == model)]
            passing = sorted(sub[sub["outcome"] == "PASS"]["width_m"].tolist())
            observed = passing[0] if passing else None
            boundaries.append(
                {
                    "rule": rule,
                    "model": model,
                    "predicted_l_crit_m": round(b["l_crit_m"], 4),
                    "observed_min_pass_width_m": observed,
                    "c0_m": b["c0_m"],
                    "k_s_per_m": b["k_s_per_m"],
                }
            )

    key_runs = []
    for rid in [
        run_id("R1", 3.00, "wiedemann"),
        run_id("S_cons", 3.00, "wiedemann"),
        run_id("S_mod", 3.00, "wiedemann"),
        run_id("S_lib", 3.00, "wiedemann"),
        run_id("S_mod", 3.00, "sfm"),
        run_id("R1", 3.00, "sfm"),
        run_id("S_mod", 2.90, "wiedemann"),
        run_id("S_mod", 2.86, "wiedemann"),
        run_id("S_mod", 2.84, "wiedemann"),
        run_id("S_lib", 2.84, "wiedemann"),
        run_id("S_lib", 2.82, "wiedemann"),
        run_id("S_mod", 2.82, "wiedemann"),
        run_id("S_lib", 2.80, "wiedemann"),
        run_id("S_mod", 3.20, "wiedemann"),
        run_id("S_mod_nf", 3.00, "sfm"),
        run_id("S_mod", 3.20, "sfm"),
        run_id("S_mod", 4.00, "wiedemann"),
        run_id("S_mod", 3.50, "wiedemann"),
    ]:
        sub = metrics[metrics["run_id"] == rid]
        if sub.empty:
            continue
        d = sub.iloc[0].to_dict()
        d = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}
        key_runs.append(d)

    peak = pass_events.dropna(subset=["clearance_m"]) if not pass_events.empty else None
    return {
        "meta": {
            "name": "week06_acceptance",
            "n_runs": int(len(metrics)),
            "t_max_s": T_MAX_S,
            "widths_m": WIDTHS,
            "rules": RULES,
            "models": MODELS,
        },
        "boundaries": boundaries,
        "prediction_consistency": consistency,
        "key_runs": key_runs,
        "pass_events_n": 0 if peak is None else int(len(peak)),
        "notes": [
            "接受关系 c_acc(v) = c0 + k·v 由行人与车辆共享；c0 是静止时愿意接受的最小余量。",
            "S_* 规则允许行人让行（纵向速度降到 0），也允许其在慢车旁继续走。",
            "边界判据：S_* 为 c_avail ≥ c0（行人可从停住的车侧走过）；R1 为 c_avail > 0.05（必须车辆通过）。",
        ],
    }


def _write_tables(
    cfg: dict[str, Any], summary: dict[str, Any], metrics: pd.DataFrame, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 模型/参数表
    lines = [
        "# 表 A1  速度相关接受模型（实验 2）",
        "",
        "共享接受关系：`c_acc(v) = c0 + k·v`；车辆允许速度 `v_clear = 0.9·(c − c0)/k`。",
        "",
        "| 规则 | 行人模型 | c0 [m] | k [s/m] | 解析临界宽度 [m] | 3.0 m 时允许速度 [m/s] |",
        "|---|---|---|---|---|---|",
    ]
    for rule_name in RULES:
        rule = cfg["rules"][rule_name]
        curve = rule_clearance_curve(rule)
        b = predicted_boundary(cfg, rule_name)
        c0 = "—" if curve is None else f"{curve[0]:.2f}"
        k = "—" if curve is None else f"{curve[1]:.3f}"
        speed = allowed_speed(cfg, rule_name, 3.00)
        lines.append(
            f"| {rule_name} | {'速度接受' if rule.get('ped_acceptance') == 'speed' else '固定 0.30 m'} | "
            f"{c0} | {k} | {b['l_crit_m']:.2f} | {speed:.2f} |"
        )
    lines += [
        "",
        "> 旧 R1 等价于“车辆速度相关余量 + 行人固定 0.30 m 阈值”的保守组合；"
        "S_cons 与 R1 车辆参数相同、只把行人换成速度接受模型，用于隔离行人模型的影响。",
        "",
    ]
    (out_dir / "A1_acceptance_model.md").write_text("\n".join(lines), encoding="utf-8")

    # 结果表（L=3.0 与边界附近）
    lines = [
        "# 表 A2  实验 2 主要结果",
        "",
        "| 宽度 [m] | 规则 | 模型 | 结果 | 车辆延误 [s] | 最低车速 [m/s] | 通过余量 [m] | 行人停车总时长 [s] | 行人平均延误 [s] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in metrics.sort_values(["width_m", "rule", "model"]).iterrows():
        if r["width_m"] > 3.10:
            continue
        delay = "—" if pd.isna(r["car_delay_s"]) else f"{r['car_delay_s']:.1f}"
        ped_delay = "—" if pd.isna(r["ped_delay_mean_s"]) else f"{r['ped_delay_mean_s']:.2f}"
        lines.append(
            f"| {r['width_m']:.2f} | {r['rule']} | {r['model']} | {r['outcome']} | {delay} | "
            f"{r['car_min_speed_mps']:.2f} | {r['min_pass_clearance_m']:.3f} | "
            f"{r['ped_stop_time_total_s']:.1f} | {ped_delay} |"
        )
    (out_dir / "A2_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 边界表
    lines = [
        "# 表 A3  可通行边界：解析 vs 仿真",
        "",
        "| 规则 | 模型 | 解析临界宽度 [m] | 仿真最小通过宽度 [m] | c0 [m] |",
        "|---|---|---|---|---|",
    ]
    for b in summary["boundaries"]:
        obs = "—" if b["observed_min_pass_width_m"] is None else f"{b['observed_min_pass_width_m']:.2f}"
        c0 = "—" if b["c0_m"] is None else f"{b['c0_m']:.2f}"
        lines.append(
            f"| {b['rule']} | {b['model']} | {b['predicted_l_crit_m']:.2f} | {obs} | {c0} |"
        )
    lines.append("")
    lines.append(f"预测与仿真结果一致性：{summary['prediction_consistency']['n_agree']}"
                 f"/{summary['prediction_consistency']['n_checked']}")
    (out_dir / "A3_boundaries.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
