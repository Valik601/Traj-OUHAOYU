"""从 results/ 汇总 CSV 生成报告用 Markdown 表格（可重跑，非一次性手抄）。

用法：python reports/make_report_tables.py
输出：reports/tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "tables.md"

MODEL_ORDER_A = ["direct", "vo", "rvo", "orca"]
MODEL_ORDER_B = ["direct", "vo", "rvo", "orca", "orca_release", "direct_release"]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(f):
        return "—"
    return f"{f:.{digits}f}"


def _load(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def table_runs() -> str:
    comp = _load("comparison_all.csv")
    if comp is None:
        return "_缺少 comparison_all.csv_\n"
    rows = []
    for exp, sub in comp.groupby("experiment_id", sort=True):
        rows.append(
            {
                "experiment_id": exp,
                "runs": len(sub),
                "valid": int(sub["valid"].sum()),
                "completed": int(sub["completed"].sum()),
                "timeout": int(((sub["status"] == "timeout") & sub["valid"]).sum()),
                "numerical_error": int((sub["status"] == "numerical_error").sum()),
                "safe_completed_scenes": int((sub["safe_completion_rate"] > 0).sum()),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_markdown(index=False) + "\n"


def _summary_table(exp: str, scenario: str, models: list[str]) -> str:
    p = RESULTS / exp / "summary_statistics.csv"
    if not p.exists():
        return f"_缺少 {p}_\n"
    df = pd.read_csv(p)
    df = df[(df["scenario"] == scenario) & (df["model"].isin(models))]
    if df.empty:
        return "_无数据_\n"
    comp = pd.read_csv(RESULTS / exp / "comparison.csv")
    comp = comp[comp["scenario"] == scenario]
    worst = comp.groupby("model")["min_separation"].min()
    df = df.set_index("model").loc[[m for m in models if m in df["model"].values]]
    out = pd.DataFrame(
        {
            "n_valid": df["n_valid"],
            "完成率": df["completed_rate"].map(_fmt),
            "到达率": df["arrival_rate_mean"].map(_fmt),
            "安全完成率": df["safe_completion_rate_mean"].map(_fmt),
            "安全完成率SD": df["safe_completion_rate_sd"].map(_fmt),
            "全队安全成功": df["team_safe_success_rate"].map(_fmt),
            "最小间距均值(m)": df["min_separation_mean"].map(_fmt),
            "最小间距最小值(m)": worst.reindex(df.index).map(_fmt),
            "侵犯机对均值": df["violating_pair_count_mean"].map(_fmt),
            "全部到达时间(s)": df["all_arrival_time_mean"].map(_fmt),
            "全部到达样本": df["all_arrival_time_n"],
            "T90(s)": df["t90_mean"].map(_fmt),
            "平均路程(m)": df["average_path_length_mean"].map(_fmt),
            "规划耗时均值(ms)": df["planning_time_mean_mean"].map(_fmt),
            "规划耗时p95(ms)": df["planning_time_p95_mean"].map(_fmt),
            "耗尽/回退": (
                df["sample_exhausted_count_mean"].map(_fmt)
                + " / "
                + df["solver_fallback_count_mean"].map(_fmt)
            ),
        }
    )
    return out.to_markdown() + "\n"


def table_paired(exp: str = "formal") -> str:
    p = RESULTS / exp / "paired_effects.csv"
    if not p.exists():
        return "_缺少 paired_effects.csv_\n"
    df = pd.read_csv(p)
    if df.empty:
        return "_无配对效应数据_\n"
    df = df.assign(
        mean_diff=df["mean_diff"].map(_fmt),
        ci_low=df["ci_low"].map(_fmt),
        ci_high=df["ci_high"].map(_fmt),
        fraction_positive=df["fraction_positive"].map(lambda v: _fmt(v, 2)),
    )
    cols = ["scenario", "metric", "pair", "n_pairs", "mean_diff", "ci_low", "ci_high", "fraction_positive"]
    return df[cols].to_markdown(index=False) + "\n"


def table_sensitivity() -> str:
    p = RESULTS / "paired_effects_all.csv"
    if not p.exists():
        return "_缺少 paired_effects_all.csv_\n"
    df = pd.read_csv(p)
    if df.empty:
        return "_无敏感性配对数据_\n"
    df = df[df["experiment_id"].astype(str).str.startswith("sensitivity_")]
    if df.empty:
        return "_无敏感性配对数据_\n"
    df = df.assign(
        mean_diff=df["mean_diff"].map(_fmt),
        ci_low=df["ci_low"].map(_fmt),
        ci_high=df["ci_high"].map(_fmt),
        fraction_positive=df["fraction_positive"].map(lambda v: _fmt(v, 2)),
    )
    cols = ["experiment_id", "metric", "model_a", "n_pairs", "mean_diff", "ci_low", "ci_high", "fraction_positive"]
    return df[cols].to_markdown(index=False) + "\n"


def table_runtime() -> str:
    comp = _load("comparison_all.csv")
    if comp is None:
        return "_缺少 comparison_all.csv_\n"
    runtimes = []
    for exp_dir in sorted(RESULTS.iterdir()):
        if not exp_dir.is_dir():
            continue
        man = exp_dir / "run_manifest.csv"
        if not man.exists():
            continue
        df = pd.read_csv(man)
        df = df[df["status"] != "resumed"]
        for (scen, model), g in df.groupby(["scenario", "model"]):
            runtimes.append(
                {
                    "experiment_id": exp_dir.name,
                    "scenario": scen,
                    "model": model,
                    "mean_wall_s": round(float(g["runtime_s"].mean()), 1),
                    "max_wall_s": round(float(g["runtime_s"].max()), 1),
                }
            )
    if not runtimes:
        return "_无运行时间数据_\n"
    out = pd.DataFrame(runtimes)
    return out.to_markdown(index=False) + "\n"


def table_stress() -> str:
    p = RESULTS / "stress" / "comparison.csv"
    if not p.exists():
        return "_缺少 stress/comparison.csv_\n"
    df = pd.read_csv(p)
    cols = [
        "scenario",
        "model",
        "arrival_rate",
        "safe_completion_rate",
        "min_separation",
        "violating_pair_count",
        "violation_event_count",
        "all_arrival_time",
        "average_path_length",
        "status",
    ]
    return df[cols].to_markdown(index=False) + "\n"


def main() -> None:
    parts = [
        "# 报告用表格（由 reports/make_report_tables.py 从 results/ 生成，禁止手工改数）",
        "",
        "## 表 1 运行清单",
        table_runs(),
        "## 表 2 Scenario A（对跖点导航，辅助实验）",
        _summary_table("formal", "A", MODEL_ORDER_A),
        "## 表 3 Scenario B（必须先访问中心，主实验）",
        _summary_table("formal", "B", MODEL_ORDER_B),
        "## 表 4 正式配对效应（场景级配对 bootstrap 95% 区间）",
        table_paired("formal"),
        "## 表 5 敏感性变体 vs 正式基线",
        table_sensitivity(),
        "## 表 6 运行成本（每次运行墙钟耗时，秒）",
        table_runtime(),
        "## 表 7 确定性压力测试（未扰动 Fibonacci，基础场景）",
        table_stress(),
    ]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
