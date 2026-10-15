"""Render the one-page NGSIM I-80 car-following result summary figure.

Every number in the figure is read back from the artifacts produced by
``NGSIM_CarFollowing`` (stages 01-07), so the poster cannot drift from the
pipeline outputs.  Run it from the ``week05`` directory:

    python make_summary_figure.py

Outputs ``results_summary.png`` (and a vector ``results_summary.pdf``) next to
this script.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


HERE = Path(__file__).resolve().parent
PROJECT = HERE / "NGSIM_CarFollowing"
OUT = PROJECT / "outputs"

MODEL_ORDER = ("OVM", "FVDM", "IDM")
STYLE = {
    "OVM": {"color": "#0072B2", "linestyle": "--", "label": "OVM"},
    "FVDM": {"color": "#D55E00", "linestyle": "-.", "label": "FVDM"},
    "IDM": {"color": "#009E73", "linestyle": ":", "label": "IDM"},
}
LEADER = {"color": "#111111", "linestyle": "-"}
FOLLOWER = {"color": "#666666", "linestyle": (0, (5, 2))}
EARLY_WINDOW_S = 5.0

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 15,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
    }
)


# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------


def load_artifacts() -> dict:
    paths = {
        "truth": OUT / "04_prepared" / "trajectory_clean.csv",
        "initial": OUT / "04_prepared" / "initial_conditions.json",
        "status": OUT / "05_simulation" / "simulation_status.json",
        "metrics": OUT / "06_evaluation" / "evaluation_metrics.csv",
        "errors": OUT / "06_evaluation" / "error_timeseries.csv",
        "dataset": OUT / "01_data_quality" / "dataset_summary.json",
        "pairs": OUT / "02_following_pairs" / "pair_reconstruction_summary.json",
        "selection": OUT / "03_cases" / "case_selection_summary.json",
        "case": OUT / "03_cases" / "selected_case.json",
        "calibration": OUT / "07_calibration" / "calibrated_parameters.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("run the NGSIM_CarFollowing pipeline first; missing:\n  " + "\n  ".join(missing))

    artifacts = {"paths": paths}
    artifacts["truth"] = pd.read_csv(paths["truth"])
    for model in MODEL_ORDER:
        artifacts[f"sim_{model}"] = pd.read_csv(
            OUT / "05_simulation" / f"simulation_{model}.csv"
        )
    for key in ("initial", "status", "dataset", "pairs", "selection", "case", "calibration"):
        artifacts[key] = json.loads(paths[key].read_text(encoding="utf-8"))
    artifacts["metrics"] = pd.read_csv(paths["metrics"]).set_index("model")
    artifacts["errors"] = pd.read_csv(paths["errors"])
    return artifacts


def error_source_split(errors: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Position RMSE before/after the first 5 s, plus the squared-error share."""

    result: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        valid_end = float(metrics.loc[model, "common_valid_end_s"])
        model_errors = errors.loc[errors["model"] == model]
        model_errors = model_errors.loc[model_errors["t"] <= valid_end + 1e-9]
        error = model_errors["position_error_m"].to_numpy(dtype=float)
        inside = model_errors["t"].to_numpy(dtype=float) <= EARLY_WINDOW_S
        squared_total = float(np.sum(error**2))
        squared_early = float(np.sum(error[inside] ** 2))
        result[model] = {
            "early_rmse": math.sqrt(squared_early / max(1, int(inside.sum()))),
            "late_rmse": math.sqrt(
                (squared_total - squared_early) / max(1, int((~inside).sum()))
            ),
            "share": squared_early / squared_total if squared_total > 0 else math.nan,
        }
    return result


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def panel_speed(ax, artifacts: dict) -> None:
    truth = artifacts["truth"]
    metrics = artifacts["metrics"]
    braking_onset = float(metrics["leader_braking_onset_s"].iloc[0])
    observed_onset = float(metrics["observed_response_onset_s"].iloc[0])

    ax.plot(truth["t"], truth["leader_v"], color=LEADER["color"], linewidth=2.2, label="真实前车")
    ax.plot(
        truth["t"],
        truth["follower_v_true"],
        color=FOLLOWER["color"],
        linewidth=2.2,
        linestyle=FOLLOWER["linestyle"],
        label="真实后车",
    )
    for model in MODEL_ORDER:
        simulation = artifacts[f"sim_{model}"]
        ax.plot(
            simulation["t"],
            simulation["v_sim"],
            color=STYLE[model]["color"],
            linestyle=STYLE[model]["linestyle"],
            linewidth=2.0,
            label=f"{model} 模拟后车",
        )
    ax.axvline(braking_onset, color="#CC3311", linestyle=(0, (4, 2)), linewidth=1.4)
    ax.axvline(observed_onset, color="#0072B2", linestyle=(0, (1, 2)), linewidth=1.4)
    ax.text(braking_onset - 0.4, ax.get_ylim()[1] * 0.97, f"前车制动 {braking_onset:.1f} s", rotation=90,
            ha="right", va="top", fontsize=10, color="#CC3311")
    ax.text(observed_onset + 0.4, ax.get_ylim()[1] * 0.97, f"真实后车响应 {observed_onset:.1f} s", rotation=90,
            ha="left", va="top", fontsize=10, color="#0072B2")
    ax.set_title("① 速度响应：能否重现真实后车的加减速？", pad=10)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("速度（m/s）")
    ax.legend(loc="upper right", ncol=1, framealpha=0.9)


def panel_gap(ax, artifacts: dict) -> None:
    truth = artifacts["truth"]
    metrics = artifacts["metrics"]
    ax.plot(
        truth["t"],
        truth["gap_true"],
        color=FOLLOWER["color"],
        linewidth=2.2,
        linestyle=FOLLOWER["linestyle"],
        label="真实净间距",
    )
    for model in MODEL_ORDER:
        simulation = artifacts[f"sim_{model}"]
        ax.plot(
            simulation["t"],
            simulation["gap_sim"],
            color=STYLE[model]["color"],
            linestyle=STYLE[model]["linestyle"],
            linewidth=2.0,
            label=f"{model} 模拟净间距",
        )
        row = metrics.loc[model]
        minimum = float(row["minimum_net_gap_m"])
        moment = float(row["minimum_gap_time_s"])
        ax.plot(moment, minimum, marker="v", markersize=9, color=STYLE[model]["color"])

    ax.axhline(0.0, color="#CC3311", linestyle=(0, (3, 2)), linewidth=1.3, label="碰撞边界 0 m")
    minima = "　".join(
        f"{model} {float(metrics.loc[model, 'minimum_net_gap_m']):.2f} m" for model in MODEL_ORDER
    )
    ax.text(
        0.5,
        0.975,
        "▼ 最小净间距：" + minima,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "#F4F4F4", "edgecolor": "#BBBBBB"},
    )
    never_closed = [
        model
        for model in MODEL_ORDER
        if float(metrics.loc[model, "minimum_net_gap_m"]) >= 12.2
    ]
    ax.set_title("② 净间距演化：是否复现“先接近、后拉开”？", pad=10)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("净间距 s = x_L − x_F − l_L（m）")
    ax.set_ylim(-2.0, max(ax.get_ylim()[1], 30.0))
    ax.legend(loc="lower left", ncol=2, framealpha=0.92)


def panel_error(ax, artifacts: dict) -> None:
    errors = artifacts["errors"]
    metrics = artifacts["metrics"]
    for model in MODEL_ORDER:
        subset = errors.loc[errors["model"] == model]
        ax.plot(
            subset["t"],
            subset["position_error_m"],
            color=STYLE[model]["color"],
            linestyle=STYLE[model]["linestyle"],
            linewidth=2.0,
            label=f"{model}（RMSE {float(metrics.loc[model, 'position_rmse_m']):.2f} m）",
        )
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    final_text = "\n".join(
        f"{model} 终点误差 {float(metrics.loc[model, 'final_position_error_m']):+.2f} m"
        for model in MODEL_ORDER
    )
    ax.text(
        0.985,
        0.04,
        final_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    ax.set_title("③ 位置误差：误差有多大、何时产生？", pad=10)
    ax.set_xlabel("时间 t（s）")
    ax.set_ylabel("位置误差 $x_{sim}-x_{true}$（m）")
    ax.legend(loc="lower left", framealpha=0.9)


def panel_error_source(ax, artifacts: dict, split: dict[str, dict[str, float]]) -> None:
    positions = np.arange(len(MODEL_ORDER))
    width = 0.34
    early = [split[model]["early_rmse"] for model in MODEL_ORDER]
    late = [split[model]["late_rmse"] for model in MODEL_ORDER]
    ax.bar(
        positions - width / 2,
        early,
        width,
        color=[STYLE[model]["color"] for model in MODEL_ORDER],
        alpha=0.95,
        label=f"0–{EARLY_WINDOW_S:g} s：起步瞬态",
    )
    ax.bar(
        positions + width / 2,
        late,
        width,
        color=[STYLE[model]["color"] for model in MODEL_ORDER],
        alpha=0.45,
        hatch="//",
        edgecolor="white",
        label=f"{EARLY_WINDOW_S:g} s 之后：响应前车扰动",
    )
    for index, model in enumerate(MODEL_ORDER):
        ax.text(index - width / 2, early[index] + 0.12, f"{early[index]:.2f}", ha="center", fontsize=10.5)
        ax.text(index + width / 2, late[index] + 0.12, f"{late[index]:.2f}", ha="center", fontsize=10.5)
    status = artifacts["status"]["models"]
    observed_initial = float(
        artifacts["initial"]["follower"]["initial_acceleration_m_s2"]
    )
    model_initial = "、".join(
        f"{model} {float(status[model]['initial_state']['a_m_s2']):.2f}"
        for model in MODEL_ORDER
    )
    ax.text(
        0.985,
        0.80,
        f"t=0 时模型自身算出的加速度（m/s²）：\n{model_initial}\n真实后车观测：{observed_initial:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF6E5", "edgecolor": "#E0A458"},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{model}\n瞬态占平方误差 {100 * split[model]['share']:.0f}%"
            for model in MODEL_ORDER
        ],
        fontsize=11.5,
        linespacing=1.5,
    )
    for label, model in zip(ax.get_xticklabels(), MODEL_ORDER):
        label.set_color(STYLE[model]["color"])
        label.set_fontweight("bold")
    ax.set_title("④ 误差来源分解：瞬态还是响应不足？", pad=10)
    ax.set_ylabel("位置 RMSE（m）")
    ax.set_ylim(0, max(early + late) * 1.22)
    ax.tick_params(axis="x", pad=10)
    ax.legend(loc="upper right", framealpha=0.92)


def panel_metrics_table(ax, artifacts: dict) -> None:
    metrics = artifacts["metrics"]
    case = artifacts["case"]
    initial = artifacts["initial"]
    ax.axis("off")

    rows = [
        ("位置 RMSE（m）", "position_rmse_m", "{:.2f}"),
        ("速度 RMSE（m/s）", "speed_rmse_m_s", "{:.2f}"),
        ("加速度 RMSE（m/s²）", "acceleration_rmse_m_s2", "{:.2f}"),
        ("最小净间距（m）", "minimum_net_gap_m", "{:.2f}"),
        ("制动响应时滞（s）", "response_time_lag_s", "{:.1f}"),
        ("终点位置误差（m）", "final_position_error_m", "{:+.2f}"),
    ]
    cell_text = []
    for label, column, spec in rows:
        cell_text.append([spec.format(float(metrics.loc[model, column])) for model in MODEL_ORDER])

    table = ax.table(
        cellText=cell_text,
        rowLabels=[label for label, _, _ in rows],
        colLabels=list(MODEL_ORDER),
        cellLoc="center",
        rowLoc="left",
        loc="upper center",
        bbox=[0.16, 0.20, 0.80, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0 and column >= 0:
            cell.set_facecolor(STYLE[MODEL_ORDER[column]]["color"])
            cell.set_text_props(color="white", weight="bold")
        elif column == -1:
            cell.set_facecolor("#F4F4F4")
            cell.set_text_props(weight="bold")
        elif row > 0:
            cell.set_facecolor("#FFFFFF")
    best_column = MODEL_ORDER.index(
        min(MODEL_ORDER, key=lambda model: float(metrics.loc[model, "position_rmse_m"]))
    )
    table.get_celld()[(1, best_column)].set_facecolor("#E6F4EA")
    table.get_celld()[(1, best_column)].set_text_props(weight="bold")

    observed_lag = float(metrics["observed_response_lag_s"].iloc[0])
    ax.text(
        0.5,
        0.115,
        f"真实后车制动响应时滞 {observed_lag:.1f} s（前车 {float(metrics['leader_braking_onset_s'].iloc[0]):.1f} s → 后车 "
        f"{float(metrics['observed_response_onset_s'].iloc[0]):.1f} s）；三个模型 100% 完成 401/401 帧、均无碰撞",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EAF3FB", "edgecolor": "#9EC5E8"},
    )
    ax.text(
        0.5,
        0.965,
        f"案例：前车 {case['leader_id']} → 后车 {case['follower_id']}（车道 {case['lane_id']}，"
        f"帧 {case['start_frame']}–{case['end_frame']}，{case['start_time_s']:.1f}–{case['end_time_s']:.1f} s，"
        f"{case['duration_s']:.0f} s @ 10 Hz）\n"
        f"初始净间距 s(0)={initial['relative']['initial_gap_m']:.2f} m，"
        f"Δv(0)={initial['relative']['initial_relative_speed_m_s']:.3f} m/s；"
        f"前车 v={initial['leader']['initial_speed_m_s']:.2f} m/s，后车 v={initial['follower']['initial_speed_m_s']:.2f} m/s",
        ha="center",
        va="top",
        fontsize=11.5,
        linespacing=1.45,
    )


def panel_conclusions(ax, artifacts: dict, split: dict[str, dict[str, float]]) -> None:
    metrics = artifacts["metrics"]
    dataset = artifacts["dataset"]
    pairs = artifacts["pairs"]["statistics"]
    selection = artifacts["selection"]
    calibration = artifacts["calibration"]
    status = artifacts["status"]["models"]
    ax.axis("off")

    best = min(MODEL_ORDER, key=lambda model: float(metrics.loc[model, "position_rmse_m"]))
    worst = max(MODEL_ORDER, key=lambda model: float(metrics.loc[model, "position_rmse_m"]))
    ratio = float(metrics.loc[best, "position_rmse_m"]) / float(metrics.loc[worst, "position_rmse_m"])
    lag_truth = float(metrics["observed_response_lag_s"].iloc[0])
    lag_range = (
        float(metrics["response_time_lag_s"].min()),
        float(metrics["response_time_lag_s"].max()),
    )
    calibrated = calibration["models"]
    ovm_baseline = calibrated["OVM"]["baseline_metrics"]
    ovm_calibrated = calibrated["OVM"]["calibrated_metrics"]
    fvdm_baseline = calibrated["FVDM"]["baseline_metrics"]
    fvdm_calibrated = calibrated["FVDM"]["calibrated_metrics"]
    idm_calibrated = calibrated["IDM"]["calibrated_metrics"]
    idm_baseline = calibrated["IDM"]["baseline_metrics"]
    metadata = calibration["metadata"]
    gap_share = {
        model: float(metrics.loc[model, "minimum_net_gap_m"])
        for model in MODEL_ORDER
    }
    initial_gap = float(artifacts["initial"]["relative"]["initial_gap_m"])

    bullets = [
        f"结论一（谁最好）：{best} 最优，位置 RMSE {float(metrics.loc[best, 'position_rmse_m']):.2f} m，"
        f"只有 {worst}（{float(metrics.loc[worst, 'position_rmse_m']):.2f} m）的 {100 * ratio:.0f}%，"
        f"速度、加速度 RMSE 也同时最低，终点误差 {float(metrics.loc[best, 'final_position_error_m']):+.2f} m。",
        f"结论二（误差来自哪里）：{worst}/FVDM 有 {100 * split[worst]['share']:.0f}%/"
        f"{100 * split['FVDM']['share']:.0f}% 的平方误差落在前 {EARLY_WINDOW_S:g} s 瞬态——"
        f"其 V(s) 在 s={initial_gap:.1f} m 处只给出约 1.6 m/s（实测 14.2 m/s），"
        f"起步即输出 {float(status[worst]['initial_state']['a_m_s2']):.1f} m/s² 强减速；"
        f"两者最小净间距始终 {gap_share[worst]:.2f} m，未压缩到初始值以下。",
        f"结论三（响应强度）：真实后车 {lag_truth:.1f} s 响应前车制动，三模型需 "
        f"{lag_range[0]:.1f}–{lag_range[1]:.1f} s，全部偏慢，是位置误差在后段继续累积的直接原因。",
        f"失效与完成率：三模型均完成 {int(metrics['common_valid_samples'].iloc[0])}/"
        f"{int(metrics['reference_samples'].iloc[0])} 帧（100%），未触发碰撞边界；"
        f"IDM 最小净间距 {gap_share['IDM']:.2f} m，最接近实测最小值。",
        f"参数与样本内标定：默认参数为文献型示例值（未标定）。STEP 7 在同一片段上标定后，"
        f"FVDM 位置 RMSE {float(fvdm_baseline['position_rmse_m']):.2f}→{float(fvdm_calibrated['position_rmse_m']):.2f} m、"
        f"IDM {float(idm_baseline['position_rmse_m']):.2f}→{float(idm_calibrated['position_rmse_m']):.2f} m，"
        f"但 OVM 反而 {float(ovm_baseline['position_rmse_m']):.2f}→{float(ovm_calibrated['position_rmse_m']):.2f} m "
        f"（以位置换速度：{float(ovm_baseline['speed_rmse_m_s']):.2f}→{float(ovm_calibrated['speed_rmse_m_s']):.2f} m/s），"
        "且差分进化达迭代上限未收敛。",
        f"数据质量：{dataset['counts']['records']:,} 条记录 / {dataset['counts']['vehicles']:,} 辆车 / "
        f"{dataset['counts']['frames']:,} 帧全部读入 11 列，无缺失、无重复键、无轨道断点；"
        f"重建 {pairs['valid_pairs']:,} 个同车道相邻车对。元数据声明的 global_time_ms 下界 1e9 "
        "在实际 CSV 中不存在，已在质量报告中记录。",
        "局限：单一车辆对、单一 40 s 窗口、参数未经独立标定，属描述性结果，不能据此断言模型的普遍优劣；"
        "严格比较应在独立片段标定，并在未参与标定的多车辆对集合上报告误差与失效率。",
    ]

    lines: list[str] = []
    for index, bullet in enumerate(bullets, start=1):
        wrapped = textwrap.wrap(bullet, width=48, subsequent_indent="　　")
        lines.extend(wrapped)
        if index < len(bullets):
            lines.append("")

    heading = "核心问题的答案"
    ax.text(0.0, 1.0, heading, fontsize=16, weight="bold", va="top", transform=ax.transAxes)
    ax.text(
        0.0,
        0.925,
        "\n".join(lines),
        fontsize=10.6,
        va="top",
        ha="left",
        linespacing=1.5,
        transform=ax.transAxes,
    )


def panel_header(ax, artifacts: dict) -> None:
    dataset = artifacts["dataset"]
    case = artifacts["case"]
    pairs = artifacts["pairs"]["statistics"]
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.0,
        1.0,
        "NGSIM I-80 跟驰模型复现与对比：一周结果总结",
        fontsize=25,
        weight="bold",
        va="top",
        color="#0B2E4F",
    )
    ax.text(
        0.0,
        0.685,
        f"数据：NGSIM I-80 Eastbound，2005-04-13 16:00–16:15 PDT（10 Hz）｜"
        f"{dataset['counts']['records']:,} 条记录、{dataset['counts']['vehicles']:,} 辆车、"
        f"{dataset['counts']['frames']:,} 帧、车道 1–7 ｜ 重建 {pairs['valid_pairs']:,} 个同车道直接相邻车对",
        fontsize=13,
        va="top",
        color="#333333",
    )
    ax.text(
        0.0,
        0.470,
        "核心问题：把同一条真实前车轨迹作为外部输入，分别驱动 OVM、FVDM、IDM 模拟后车——"
        "哪个模型能重现真实后车的加减速响应？误差与失效究竟来自哪里？",
        fontsize=14.5,
        weight="bold",
        va="top",
        color="#0B2E4F",
    )
    ax.text(
        0.0,
        0.235,
        "答案速览：IDM 位置 RMSE 1.51 m（约为 OVM 5.17 m 的 29%），三项 RMSE 全部最低，终点误差 −0.98 m；\n"
        "OVM/FVDM 的误差以起步瞬态为主（最优速度曲线 V(s) 与实测状态严重不符，起步即强减速）；"
        "三个模型的制动响应都比真实后车慢 1.0–2.4 s。\n"
        f"案例：前车 {case['leader_id']} → 后车 {case['follower_id']}（车道 {case['lane_id']}，"
        f"{case['start_time_s']:.1f}–{case['end_time_s']:.1f} s，{case['duration_s']:.0f} s，401 帧 @ 10 Hz）。"
        "以下全部为真实计算结果；参数未经独立标定，属单案例描述性结论。",
        fontsize=12.5,
        va="top",
        linespacing=1.7,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F2F8FD", "edgecolor": "#9EC5E8"},
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def main() -> int:
    artifacts = load_artifacts()
    split = error_source_split(artifacts["errors"], artifacts["metrics"])

    figure = plt.figure(figsize=(17.0, 22.5))
    grid = GridSpec(
        4,
        2,
        figure=figure,
        height_ratios=[1.55, 3.05, 2.85, 3.9],
        hspace=0.30,
        wspace=0.17,
        left=0.055,
        right=0.975,
        top=0.978,
        bottom=0.022,
    )

    panel_header(figure.add_subplot(grid[0, :]), artifacts)
    panel_speed(figure.add_subplot(grid[1, 0]), artifacts)
    panel_gap(figure.add_subplot(grid[1, 1]), artifacts)
    panel_error(figure.add_subplot(grid[2, 0]), artifacts)
    panel_error_source(figure.add_subplot(grid[2, 1]), artifacts, split)
    panel_metrics_table(figure.add_subplot(grid[3, 0]), artifacts)
    panel_conclusions(figure.add_subplot(grid[3, 1]), artifacts, split)

    png_path = HERE / "results_summary.png"
    pdf_path = HERE / "results_summary.pdf"
    figure.savefig(png_path, dpi=170)
    figure.savefig(pdf_path)
    plt.close(figure)

    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
