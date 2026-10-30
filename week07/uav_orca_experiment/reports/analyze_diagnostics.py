"""补充诊断分析（供报告使用，从已保存轨迹/事件重算，不改动结果）。

用法：python reports/analyze_diagnostics.py
输出：reports/diagnostics.md, reports/diagnostics.json

内容包括：
1. 场景几何诊断：起点—他机终点交叉邻接、最小间距（20 个正式场景）。
2. Direct + Release 的冻结体冲突：侵犯事件中涉及“等待/已到达冻结体”的比例。
3. ORCA 在 B 的中心瓶颈行为：未过中心/停滞/位移统计。
4. 分时放行的“相对理想到心时刻偏差” center_entry - (release_actual + R/v_max)。
5. 每模型每场景最小间距与到达/安全完成分布摘要。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SCENES = ROOT / "scenes"
OUT_MD = ROOT / "reports" / "diagnostics.md"
OUT_JSON = ROOT / "reports" / "diagnostics.json"

R = 20.0
V_MAX = 2.0
DT = 0.1


def _run_dirs(exp: str, scenario: str, model: str) -> list[Path]:
    base = RESULTS / exp / scenario / model
    if not base.exists():
        return []
    return sorted({p.parent for p in base.rglob("metrics.json")})


def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def scene_geometry() -> dict:
    manifest = json.loads((SCENES / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for sid, rec in sorted(manifest["scenes"].items()):
        if sid == "base_fibonacci":
            continue
        rows.append(
            {
                "scene_id": sid,
                "min_start_separation": rec.get("min_start_separation"),
                "min_goal_separation": rec.get("min_goal_separation"),
                "min_start_other_goal": rec.get("min_start_other_goal"),
                "cross_pairs_below_safety": rec.get("cross_pairs_below_safety"),
                "resample_count": rec.get("resample_count"),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "n_scenes": len(df),
        "min_start_other_goal_min": float(df["min_start_other_goal"].min()),
        "min_start_other_goal_mean": float(df["min_start_other_goal"].mean()),
        "scenes_with_cross_below_safety": int((df["cross_pairs_below_safety"] > 0).sum()),
        "cross_pairs_total": int(df["cross_pairs_below_safety"].sum()),
        "rows": df.to_dict(orient="records"),
    }


def frozen_conflict(exp: str = "formal") -> dict:
    """B 场景 Direct + Release：侵犯事件是否由冻结体（等待/已到达）参与。"""
    records = []
    for run_dir in _run_dirs(exp, "B", "direct_release"):
        with np.load(run_dir / "trajectories.npz", allow_pickle=False) as npz:
            frozen = np.array(npz["frozen"])
            stages = np.array(npz["stages"])
            t = np.array(npz["t"])
        events = pd.read_csv(run_dir / "violation_events.csv")
        n_events = len(events)
        n_frozen = 0
        n_arrived = 0
        n_waiting = 0
        pair_frozen = set()
        for _, ev in events.iterrows():
            start = float(ev["start_time"])
            k = int(np.clip(np.searchsorted(t, start, side="right") - 1, 0, len(t) - 1))
            i, j = int(ev["agent_i"]), int(ev["agent_j"])
            involved = []
            for idx in (i, j):
                if frozen[k, idx]:
                    involved.append(int(stages[k, idx]))
                    pair_frozen.add(tuple(sorted((i, j))))
            if involved:
                n_frozen += 1
                if 3 in involved:
                    n_arrived += 1
                if 0 in involved:
                    n_waiting += 1
        records.append(
            {
                "scene_id": int(run_dir.parent.name.replace("scene_", "")),
                "events": n_events,
                "events_with_frozen": n_frozen,
                "events_with_arrived": n_arrived,
                "events_with_waiting": n_waiting,
                "frozen_pair_count": len(pair_frozen),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return {"n_scenes": 0}
    return {
        "n_scenes": len(df),
        "events_total": int(df["events"].sum()),
        "events_with_frozen": int(df["events_with_frozen"].sum()),
        "events_with_arrived": int(df["events_with_arrived"].sum()),
        "events_with_waiting": int(df["events_with_waiting"].sum()),
        "fraction_events_with_frozen": float(df["events_with_frozen"].sum() / max(df["events"].sum(), 1)),
        "rows": df.to_dict(orient="records"),
    }


def orca_bottleneck(exp: str = "formal", model: str = "orca") -> dict:
    rows = []
    for run_dir in _run_dirs(exp, "B", model):
        metrics = _load_json(run_dir / "metrics.json")
        agents = pd.read_csv(run_dir / "agents.csv")
        area = run_dir.parent.name.replace("scene_", "")
        rows.append(
            {
                "scene_id": int(area),
                "arrival_count": metrics["arrival_count"],
                "center_crossing_count": metrics["center_crossing_count"],
                "safe_completion_count": metrics["safe_completion_count"],
                "stalled_agent_count": metrics["stall"]["stalled_agent_count"],
                "min_separation": metrics["min_separation"],
                "path_mean": float(agents["path_length"].mean()),
                "path_min": float(agents["path_length"].min()),
                "path_max": float(agents["path_length"].max()),
                "final_goal_distance_mean": float(agents["final_goal_distance"].mean()),
                "center_entry_finite": int(agents["center_entry_time"].notna().sum()),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_scenes": 0}
    return {
        "model": model,
        "n_scenes": len(df),
        "arrival_mean": float(df["arrival_count"].mean()),
        "center_mean": float(df["center_crossing_count"].mean()),
        "safe_mean": float(df["safe_completion_count"].mean()),
        "stalled_mean": float(df["stalled_agent_count"].mean()),
        "min_separation_min": float(df["min_separation"].min()),
        "path_mean_of_scenes": float(df["path_mean"].mean()),
        "rows": df.to_dict(orient="records"),
    }


def release_deviation(exp: str = "formal") -> dict:
    """相对理想到心时刻偏差 = center_entry - (release_actual + R/v_max)。"""
    out = {}
    for model in ("orca_release", "direct_release"):
        vals = []
        for run_dir in _run_dirs(exp, "B", model):
            agents = pd.read_csv(run_dir / "agents.csv")
            sub = agents.dropna(subset=["center_entry_time", "release_actual"])
            if len(sub):
                vals.append((sub["center_entry_time"] - sub["release_actual"] - R / V_MAX).to_numpy())
        if not vals:
            continue
        arr = np.concatenate(vals)
        out[model] = {
            "n_agents": int(arr.size),
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "fraction_negative": float(np.mean(arr < 0)),
        }
    return out


def all_arrivals_by_model(exp: str = "formal") -> dict:
    comp = pd.read_csv(RESULTS / exp / "comparison.csv")
    out = {}
    for scenario in ("A", "B"):
        for model, g in comp[comp["scenario"] == scenario].groupby("model"):
            out[f"{scenario}:{model}"] = {
                "n": int(len(g)),
                "arrival_rate_mean": float(g["arrival_rate"].mean()),
                "safe_completion_rate_mean": float(g["safe_completion_rate"].mean()),
                "min_separation_mean": float(g["min_separation"].mean()),
                "min_separation_min": float(g["min_separation"].min()),
                "violating_pair_count_mean": float(g["violating_pair_count"].mean()),
                "violating_pair_count_max": int(g["violating_pair_count"].max()),
                "all_arrival_time_n": int(g["all_arrival_time"].notna().sum()),
                "all_arrival_time_mean": (
                    float(g["all_arrival_time"].mean()) if g["all_arrival_time"].notna().any() else None
                ),
                "completed_scenes": int(g["completed"].sum()),
                "timeout_scenes": int(((g["status"] == "timeout") & g["valid"]).sum()),
            }
    return out


def main() -> None:
    diag = {
        "scene_geometry": scene_geometry(),
        "frozen_conflict_direct_release": frozen_conflict(),
        "orca_b_bottleneck": orca_bottleneck(),
        "release_center_deviation": release_deviation(),
        "arrivals_by_model": all_arrivals_by_model(),
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    geo = diag["scene_geometry"]
    fc = diag["frozen_conflict_direct_release"]
    orca = diag["orca_b_bottleneck"]
    dev = diag["release_center_deviation"]
    lines = [
        "# 补充诊断（自动生成，见 reports/analyze_diagnostics.py）",
        "",
        "## 1. 正式场景几何",
        f"- 场景数：{geo['n_scenes']}；起点—他机终点最小距离 "
        f"min={geo['min_start_other_goal_min']:.3f} m，mean={geo['min_start_other_goal_mean']:.3f} m",
        f"- 存在 <1 m 起点—他机终点交叉的场景数：{geo['scenes_with_cross_below_safety']}/{geo['n_scenes']}，"
        f"交叉邻接对数合计 {geo['cross_pairs_total']}",
        "",
        "## 2. Direct + Release 的冻结体冲突（B）",
        f"- 侵犯事件总数 {fc['events_total']}，其中涉及冻结体的事件 {fc['events_with_frozen']} "
        f"（{100 * fc['fraction_events_with_frozen']:.1f}%）；涉及已到达者 {fc['events_with_arrived']}，"
        f"涉及未释放者 {fc['events_with_waiting']}",
        "",
        "## 3. ORCA 在 B 的中心瓶颈行为",
        f"- 场景数 {orca['n_scenes']}：平均到达 {orca['arrival_mean']:.2f}/100，"
        f"平均中心访问 {orca['center_mean']:.2f}/100，平均安全完成 {orca['safe_mean']:.2f}/100，"
        f"平均停滞机数 {orca['stalled_mean']:.1f}",
        f"- 最小间距场景最小值 {orca['min_separation_min']:.4f} m；平均路程（场景均值）"
        f"{orca['path_mean_of_scenes']:.2f} m",
        "",
        "## 4. 相对理想到心时刻偏差 center_entry − (release_actual + R/v_max)",
    ]
    for model, stats in dev.items():
        lines.append(
            f"- {model}: n={stats['n_agents']}，mean={stats['mean']:.3f} s，"
            f"p05/p50/p95={stats['p05']:.2f}/{stats['p50']:.2f}/{stats['p95']:.2f} s，"
            f"负值比例 {100 * stats['fraction_negative']:.1f}%"
        )
    lines += ["", "## 5. 各模型汇总（来自 comparison.csv）"]
    lines.append(
        "| 场景:模型 | n | 到达率 | 安全完成率 | 最小间距均值 | 最小间距最小 | 侵犯机对均值 | 侵犯机对最大 | 全部到达样本 | 全部到达均值(s) | 超时场景 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, v in diag["arrivals_by_model"].items():
        all_arr = "—" if v["all_arrival_time_mean"] is None else f"{v['all_arrival_time_mean']:.2f}"
        lines.append(
            f"| {key} | {v['n']} | {v['arrival_rate_mean']:.3f} | {v['safe_completion_rate_mean']:.3f} | "
            f"{v['min_separation_mean']:.3f} | {v['min_separation_min']:.3f} | "
            f"{v['violating_pair_count_mean']:.1f} | {v['violating_pair_count_max']} | "
            f"{v['all_arrival_time_n']} | {all_arr} | "
            f"{v['timeout_scenes']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} and {OUT_JSON}")


if __name__ == "__main__":
    main()
