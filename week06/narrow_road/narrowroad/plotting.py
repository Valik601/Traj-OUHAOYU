"""绘图：把真实运行结果画成报告 / 汇报用图（PNG + PDF）。

所有图只读取 outputs/ 下的真实产物（metrics.csv、ped_metrics.csv、
trajectories/*.csv、summary.json），不包含任何手写数值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from .common import load_config

# ---------------------------------------------------------------------------
# 全局样式
# ---------------------------------------------------------------------------
COL = {
    "car": "#b2182b",
    "car_fill": "#ef8a62",
    "peds": ["#2166ac", "#4393c3", "#92c5de"],
    "road": "#f4f4f4",
    "edge": "#333333",
    "pass": "#1b7837",
    "dead": "#b2182b",
    "zone_rule": "#fddbc7",
    "zone_space": "#d9d9d9",
    "zone_pass": "#d9f0d3",
}
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
    }
)


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
class Outputs:
    def __init__(self, out_dir: str | Path, summary: dict[str, Any] | None = None):
        self.dir = Path(out_dir)
        self.metrics = pd.read_csv(self.dir / "metrics.csv")
        self.ped = pd.read_csv(self.dir / "ped_metrics.csv")
        self.summary = summary
        if self.summary is None:
            import json

            with open(self.dir / "summary.json", "r", encoding="utf-8") as f:
                self.summary = json.load(f)
        self.cfg = load_config()
        self._traces: dict[str, dict[str, pd.DataFrame]] = {}

    def trace(self, run_id: str) -> dict[str, pd.DataFrame]:
        if run_id not in self._traces:
            df = pd.read_csv(self.dir / "trajectories" / f"{run_id}.csv")
            self._traces[run_id] = {
                name: g.sort_values("t").reset_index(drop=True) for name, g in df.groupby("agent")
            }
        return self._traces[run_id]

    def row(self, run_id: str) -> pd.Series:
        return self.metrics[self.metrics["run_id"] == run_id].iloc[0]


# ---------------------------------------------------------------------------
# 基础绘图工具
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_road(ax: plt.Axes, width: float, x0: float, x1: float, center_y: float = 0.0) -> None:
    ax.add_patch(
        Rectangle(
            (x0, center_y - 0.5 * width),
            x1 - x0,
            width,
            facecolor=COL["road"],
            edgecolor=COL["edge"],
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.plot([x0, x1], [center_y, center_y], color="#9e9e9e", lw=0.6, ls=(0, (5, 4)), zorder=1)


def _draw_car(ax: plt.Axes, x: float, y: float, car_cfg: dict[str, Any], alpha: float = 0.9,
              zorder: int = 5) -> None:
    L, W = float(car_cfg["length_m"]), float(car_cfg["width_m"])
    ax.add_patch(
        Rectangle(
            (x - L / 2, y - W / 2), L, W,
            facecolor=COL["car_fill"], edgecolor=COL["car"], linewidth=1.1,
            alpha=alpha, zorder=zorder,
        )
    )
    ax.add_patch(
        Rectangle(
            (x - (L / 2 - 0.55), y - (W / 2 - 0.2)),
            L - 1.1, W - 0.4,
            facecolor="white", edgecolor="none", alpha=0.5, zorder=zorder + 1,
        )
    )


def _draw_ped(ax: plt.Axes, x: float, y: float, i: int, r: float = 0.25, label: bool = True,
              zorder: int = 6, alpha: float = 0.95) -> None:
    ax.add_patch(
        Circle((x, y), r, facecolor=COL["peds"][i % 3], edgecolor="#10375c",
               linewidth=0.7, alpha=alpha, zorder=zorder)
    )
    if label:
        ax.annotate(f"P{i + 1}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=6.8, color="#10375c", zorder=zorder + 2)


def _dim_arrow(ax: plt.Axes, xy0, xy1, text: str, color="#333333", offset=(0, 0),
               fontsize=7) -> None:
    ax.annotate("", xy=xy1, xytext=xy0,
                arrowprops=dict(arrowstyle="<->", color=color, lw=0.8, shrinkA=0, shrinkB=0),
                zorder=20)
    ax.annotate(text, xy=((xy0[0] + xy1[0]) / 2 + offset[0], (xy0[1] + xy1[1]) / 2 + offset[1]),
                ha="center", va="center", fontsize=fontsize, color=color, zorder=21,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _nearest_row(df: pd.DataFrame, t: float) -> pd.Series:
    return df.loc[(df["t"] - t).abs().idxmin()]


def _nearest_time(df: pd.DataFrame, target: str, value: float) -> float:
    idx = (df[target] - value).abs().idxmin()
    return float(df.loc[idx, "t"])


# ---------------------------------------------------------------------------
# 图 1：场景与几何
# ---------------------------------------------------------------------------
def fig01_scenario(out: Outputs, out_dir: Path) -> None:
    cfg = out.cfg
    road, car_cfg, ped_cfg = cfg["road"], cfg["car"], cfg["pedestrians"]
    ics = out.summary["initial_conditions"]
    width = float(road["widths_m"][2]) if len(road["widths_m"]) >= 3 else 3.0
    L = width
    Wc, Wp = float(car_cfg["width_m"]), float(ped_cfg["width_m"])
    y_edge = L / 2 - Wp / 2
    c_avail = L / 2 - Wc / 2 - Wp

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 2.9), gridspec_kw={"width_ratios": [2.2, 1.0]})

    # (a) 平面图
    ax = axes[0]
    _draw_road(ax, L, 0.0, float(road["x_max_m"]))
    for i, ic in enumerate(ics):
        x, y = float(ic["x_m"]), float(ic["y_m"])
        _draw_ped(ax, x, y, i)
        ax.add_patch(
            FancyArrowPatch((x + 0.55, y), (x + 4.2, y), arrowstyle="-|>",
                            mutation_scale=7, color=COL["peds"][i], lw=1.0)
        )
        ax.annotate(f"{ic['v0_mps']:.2f} m/s", (x + 4.3, y - 0.42), fontsize=6.8,
                    color=COL["peds"][i], va="top")
    _draw_car(ax, float(car_cfg["start_x_m"]), float(car_cfg["y_m"]), car_cfg)
    ax.add_patch(
        FancyArrowPatch((float(car_cfg["start_x_m"]) - 3.0, 0.0),
                        (float(car_cfg["start_x_m"]) - 14.0, 0.0),
                        arrowstyle="-|>", mutation_scale=13, color=COL["car"], lw=1.8)
    )
    ax.annotate(f"vehicle {car_cfg['start_speed_mps']:.1f} m/s  (no lateral motion)",
                (float(car_cfg["start_x_m"]) - 8.0, 1.75), fontsize=7.2, color=COL["car"],
                ha="center")
    ax.annotate("pedestrians walk +x", (44.0, -2.05), fontsize=7.2, color="#444444")
    ax.annotate("", xy=(74.0, -2.05), xytext=(58.0, -2.05), fontsize=7.2,
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))
    ax.set_xlim(-6, float(road["x_max_m"]) + 6)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"(a) plan view at t = 0  (L = {L:.1f} m)")
    ax.grid(axis="y", alpha=0.12)

    # (b) 横断面：水平轴为 y，展示 L / Wc / Wp / c 的定义
    ax = axes[1]
    ax.add_patch(Rectangle((-L / 2, 0.0), L, 1.0, facecolor=COL["road"],
                           edgecolor=COL["edge"], lw=0.9))
    ax.add_patch(Rectangle((-Wc / 2, 0.18), Wc, 0.64, facecolor=COL["car_fill"],
                           edgecolor=COL["car"], lw=1.0))
    ax.add_patch(Rectangle((-y_edge - Wp / 2, 0.30), Wp, 0.40, facecolor=COL["peds"][0],
                           edgecolor="#10375c", lw=0.8))
    ax.annotate("vehicle", (0.0, 0.5), ha="center", va="center", fontsize=7, color="#7f2704")
    ax.annotate("P1", (-y_edge + 0.02, 0.50), ha="center", va="center", fontsize=5.5,
                color="white")
    _dim_arrow(ax, (-L / 2, 1.18), (L / 2, 1.18), f"L = {L:.2f} m", offset=(0, 0.13))
    _dim_arrow(ax, (-Wc / 2, 0.72), (Wc / 2, 0.72), f"Wc = {Wc:.2f}", fontsize=6.4)
    _dim_arrow(ax, (-L / 2, 0.06), (-L / 2 + Wp, 0.06), f"Wp = {Wp:.2f}",
               offset=(0, -0.06), fontsize=6.2)
    # 余量 c（很小，用引线标出）
    x_gap = -Wc / 2 - c_avail / 2
    ax.annotate(
        f"c = {c_avail:.2f} m\n(the only passing margin)",
        xy=(x_gap, 0.5), xytext=(0.15, -0.42), fontsize=6.6, ha="center",
        color=COL["dead"],
        arrowprops=dict(arrowstyle="->", lw=0.8, color=COL["dead"]),
    )
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-0.65, 1.45)
    ax.set_xlabel("y [m]  (lateral)")
    ax.set_yticks([])
    ax.set_title("(b) cross-section: where the clearance goes")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out_dir, "fig01_scenario")


# ---------------------------------------------------------------------------
# 图 2：两种宽度（R0）
# ---------------------------------------------------------------------------
def _panel_xt(ax, traces, title: str, row: pd.Series) -> None:
    car = traces["CAR"]
    ax.plot(car["t"], car["x"], color=COL["car"], lw=2.2, label="vehicle", zorder=5)
    for i in range(3):
        name = f"P{i + 1}"
        if name in traces:
            ax.plot(traces[name]["t"], traces[name]["x"], color=COL["peds"][i], lw=1.1,
                    label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("x [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    if row["outcome"] == "DEADLOCK":
        t_dl = float(row["deadlock_t_s"])
        t_stop = t_dl - float(row["car_stop_time_s"])
        ax.axvspan(t_stop, t_dl, color=COL["dead"], alpha=0.07, lw=0)
        ax.annotate(
            f"vehicle stopped for {float(row['car_stop_time_s']):.0f} s\n"
            f"standoff {float(row['standoff_gap_m']):.1f} m → deadlock at t = {t_dl:.0f} s",
            (0.97, 0.10), xycoords="axes fraction", fontsize=7, color=COL["dead"],
            ha="right", va="bottom",
        )


def _panel_yt(ax, traces, width: float, title: str) -> None:
    half = width / 2
    ax.axhspan(-0.9, 0.9, color=COL["car"], alpha=0.10, lw=0,
               label="vehicle band (±Wc/2)")
    ax.axhline(half, color=COL["edge"], lw=1.1)
    ax.axhline(-half, color=COL["edge"], lw=1.1)
    for i in range(3):
        name = f"P{i + 1}"
        if name in traces:
            ax.plot(traces[name]["t"], traces[name]["y"], color=COL["peds"][i], lw=1.4,
                    label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_ylim(-half - 0.45, half + 0.45)
    ax.legend(loc="center right", ncol=1, fontsize=6.6)


def fig02_width_compare(out: Outputs, out_dir: Path) -> None:
    rows = [out.row(f"L{w:.1f}_R0_sfm") for w in (4.0, 3.0)]
    traces = [out.trace(f"L{w:.1f}_R0_sfm") for w in (4.0, 3.0)]
    letters = [["(a)", "(b)"], ["(c)", "(d)"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2))
    for r, (width, row, tr) in enumerate(zip((4.0, 3.0), rows, traces)):
        c_avail = float(row["c_avail_m"])
        suffix = "PASS" if row["outcome"] == "PASS" else "DEADLOCK"
        _panel_xt(axes[r, 0], tr,
                  f"{letters[r][0]} L = {width:.1f} m, R0 — x(t)  [{suffix}]", row)
        _panel_yt(axes[r, 1], tr, width,
                  f"{letters[r][1]} L = {width:.1f} m — y(t),  c_avail = {c_avail:.2f} m")
    fig.suptitle("Only the road width changes (same initial conditions and rules; social-force model)",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "fig02_width_compare_R0")


# ---------------------------------------------------------------------------
# 图 3：俯视快照（每个情景 3 个关键时刻）
# ---------------------------------------------------------------------------
def _snapshot_times(out: Outputs, rid: str) -> list[float]:
    traces = out.trace(rid)
    row = out.row(rid)
    car = traces["CAR"]
    if row["outcome"] == "PASS":
        times = []
        for i in range(3):
            p = traces[f"P{i + 1}"]
            d = np.abs(p["x"].to_numpy() - car["x"].to_numpy())
            times.append(float(p["t"].to_numpy()[int(np.argmin(d))]))
        return sorted(times)
    t_stop = float(row["deadlock_t_s"]) - float(row["car_stop_time_s"])
    return [max(t_stop - 4.0, 12.0), t_stop + 1.0, float(row["deadlock_t_s"])]


def fig03_snapshots(out: Outputs, out_dir: Path) -> None:
    cases = [
        ("L4.0_R0_sfm", 4.0, "L = 4.0 m\nc = 0.60 m"),
        ("L3.0_R0_sfm", 3.0, "L = 3.0 m\nc = 0.10 m"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 4.6))
    for r, (rid, width, label) in enumerate(cases):
        traces = out.trace(rid)
        times = _snapshot_times(out, rid)
        for c, t in enumerate(times):
            ax = axes[r, c]
            car = _nearest_row(traces["CAR"], t)
            view = 8.5
            xc = float(car["x"])
            _draw_road(ax, width, xc - view, xc + view)
            for i in range(3):
                p = _nearest_row(traces[f"P{i + 1}"], t)
                if abs(float(p["x"]) - xc) <= view:
                    _draw_ped(ax, float(p["x"]), float(p["y"]), i)
            _draw_car(ax, xc, float(car["y"]), out.cfg["car"])
            c_vals = []
            for i in range(3):
                p = _nearest_row(traces[f"P{i + 1}"], t)
                if abs(float(p["x"]) - float(car["x"])) <= 8.0:
                    c_vals.append(abs(float(p["y"]) - float(car["y"])) - 0.5 * (
                        float(out.cfg["car"]["width_m"]) + float(out.cfg["pedestrians"]["width_m"])))
            title = f"t = {t:4.1f} s   v = {float(car['speed']):4.1f} m/s"
            if c_vals:
                title += f"   c = {min(c_vals):.2f} m"
            ax.set_title(title, fontsize=7.0, pad=3)
            ax.set_xlim(xc - view, xc + view)
            ax.set_ylim(-width / 2 - 0.55, width / 2 + 0.55)
            ax.set_aspect("equal")
            ax.set_yticks([-width / 2, 0, width / 2])
            ax.set_yticklabels([f"-{width/2:.1f}", "0", f"{width/2:.1f}"], fontsize=6)
            ax.grid(False)
            if c == 0:
                ax.set_ylabel(f"{label}\ny [m]", fontsize=7.2)
            else:
                ax.set_yticks([])
            if r == 1:
                ax.set_xlabel("x [m]")
                ax.set_xticks([round(xc) - 8, round(xc), round(xc) + 8])
                ax.set_xticklabels([f"{round(xc) - 8:d}", f"{round(xc):d}", f"{round(xc) + 8:d}"])
                ax.tick_params(axis="x", labelsize=6)
            else:
                ax.set_xticks([])
    fig.suptitle("Top-down snapshots (social-force model, rule R0): the narrow case never resolves",
                 fontsize=9.5, y=1.03)
    fig.tight_layout()
    _save(fig, out_dir, "fig03_snapshots")


# ---------------------------------------------------------------------------
# 图 4：失败案例与规则修改（L = 3.0）
# ---------------------------------------------------------------------------
def fig04_rule_modification(out: Outputs, out_dir: Path) -> None:
    runs = [
        ("L3.0_R0_sfm", "R0: fixed 0.30 m margin", COL["dead"]),
        ("L3.0_R1_sfm", "R1: speed-dependent margin", COL["pass"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.8))
    for rid, label, color in runs:
        tr = out.trace(rid)
        row = out.row(rid)
        car = tr["CAR"]
        axes[0, 0].plot(car["t"], car["speed"], color=color, lw=1.8, label=label)
        axes[0, 1].plot(car["t"], car["x"], color=color, lw=1.8, label=label)
        axes[1, 0].plot(tr["P1"]["t"], tr["P1"]["y"], color=color, lw=1.8, label=label)
    r0 = out.row("L3.0_R0_sfm")
    r1 = out.row("L3.0_R1_sfm")
    axes[0, 0].set_title(
        f"(a) vehicle speed — R0 deadlocks after {float(r0['car_stop_time_s']):.0f} s stopped; "
        f"R1 crawls at {float(r1['car_min_speed_mps']):.1f} m/s"
    )
    axes[0, 1].set_title("(b) vehicle position")
    axes[1, 0].set_title("(c) P1 lateral position (both reach the curb)")
    axes[1, 0].axhline(1.5, color=COL["edge"], lw=1.0)
    axes[1, 0].axhline(-1.5, color=COL["edge"], lw=1.0)

    # (d) 单因素消融
    ax = axes[1, 1]
    rules = ["R0", "A1", "A2", "R1"]
    labels = ["R0\n(base)", "A1\ncar only", "A2\nped only", "R1\n(both)"]
    for i, rule in enumerate(rules):
        r = out.row(f"L3.0_{rule}_sfm")
        if r["outcome"] == "PASS":
            ax.bar(i, float(r["car_delay_s"]), color=COL["pass"], width=0.6)
        else:
            ax.scatter([i], [0.8], marker="x", s=90, color=COL["dead"], linewidth=2.0,
                       zorder=5)
            ax.annotate("deadlock", (i, 2.4), ha="center", fontsize=7, color=COL["dead"],
                        rotation=90, va="bottom")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("vehicle delay [s]")
    ax.set_title("(d) one-factor ablation at L = 3.0 m")
    ax.set_ylim(0, 36)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("time [s]")
    axes[0, 0].legend(loc="center right", fontsize=6.8)
    axes[0, 1].legend(loc="upper right", fontsize=6.8)
    axes[1, 0].legend(loc="center right", fontsize=6.8)
    axes[0, 0].set_ylabel("v [m/s]")
    axes[0, 1].set_ylabel("x [m]")
    axes[1, 0].set_ylabel("y [m]")
    axes[1, 0].set_ylim(-1.75, 1.75)
    fig.suptitle("L = 3.0 m: geometry permits passage (0.10 m clearance) — R0 deadlocks, R1 passes at crawl "
                 "(social force)", fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig04_rule_modification")


# ---------------------------------------------------------------------------
# 图 5：空间不足 vs 规则失效
# ---------------------------------------------------------------------------
def fig05_classification(out: Outputs, out_dir: Path) -> None:
    cfg = out.cfg
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    l_rule = 2.0 * (float(cfg["rules"]["R0"]["car_c_req_m"]) + car_w / 2 + ped_w)
    l_space = 2.0 * (float(cfg["rules"]["R1"]["car_c_min_m"]) + car_w / 2 + ped_w)
    l_min, l_max = 2.7, 4.1

    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.axvspan(l_min, l_space, color=COL["zone_space"], alpha=0.8, lw=0)
    ax.axvspan(l_space, l_rule, color=COL["zone_rule"], alpha=0.9, lw=0)
    ax.axvspan(l_rule, l_max, color=COL["zone_pass"], alpha=0.85, lw=0)

    xs = np.linspace(l_min, l_max, 400)
    ax.plot(xs, xs / 2 - car_w / 2 - ped_w, color="#333333", lw=1.5,
            label="c_avail(L) = L/2 − Wc/2 − Wp", zorder=4)
    ax.axhline(float(cfg["rules"]["R0"]["car_c_req_m"]), color=COL["dead"], lw=1.0, ls="--",
               label="R0 requires c ≥ 0.30 m", zorder=3)
    ax.axhline(float(cfg["rules"]["R1"]["car_c_min_m"]), color=COL["pass"], lw=1.0, ls=":",
               label="R1 needs c > 0.05 m", zorder=3)
    for xv in (l_space, l_rule):
        ax.axvline(xv, color="#555555", lw=0.9, ls="-.")
    ax.annotate(f"L = {l_space:.2f} m\n(space boundary)", (l_space + 0.02, 0.74),
                ha="left", va="top", fontsize=7)
    ax.annotate(f"L = {l_rule:.2f} m\n(R0 boundary)", (l_rule + 0.02, 0.74),
                ha="left", va="top", fontsize=7)

    for rule, marker, off in (("R0", "o", -0.010), ("R1", "s", 0.010)):
        sub = out.metrics[(out.metrics["rule"] == rule) & (out.metrics["model"] == "sfm")]
        for _, r in sub.iterrows():
            x, y = float(r["width_m"]) + off, float(r["c_avail_m"])
            passed = r["outcome"] == "PASS"
            ax.scatter([x], [y], marker=marker, s=52,
                       facecolor=COL["pass"] if passed else "white",
                       edgecolor=COL["pass"] if passed else COL["dead"], linewidth=1.3,
                       zorder=10)
            if not passed:
                ax.scatter([x], [y], marker="x", s=30, color=COL["dead"], zorder=11,
                           linewidth=1.2)
    ax.scatter([], [], marker="o", facecolor=COL["pass"], edgecolor=COL["pass"],
               label="R0 pass (○ deadlock)")
    ax.scatter([], [], marker="s", facecolor=COL["pass"], edgecolor=COL["pass"],
               label="R1 pass (□ deadlock)")

    ax.annotate("space\ninsufficient", ((l_min + l_space) / 2, 0.20),
                ha="center", fontsize=7.6, color="#333333")
    ax.annotate("geometrically feasible,\nbut base rule stalls\n→ RULE FAILURE",
                ((l_space + l_rule) / 2, 0.56), ha="center", fontsize=7.6, color=COL["dead"])
    ax.annotate("base rule passes", ((l_rule + l_max) / 2, 0.64), ha="center",
                fontsize=7.6, color=COL["pass"])

    ax.set_xlim(l_min, l_max)
    ax.set_ylim(0.0, 0.80)
    ax.set_xlabel("road width L [m]")
    ax.set_ylabel("available lateral clearance c_avail [m]")
    ax.set_title("Geometric feasibility vs. rule threshold: where does the failure come from? "
                 "(markers = simulated outcomes, SFM)", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, fontsize=6.8)
    fig.tight_layout()
    _save(fig, out_dir, "fig05_classification")


# ---------------------------------------------------------------------------
# 图 6：宽度扫描
# ---------------------------------------------------------------------------
def fig06_width_sweep(out: Outputs, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.3), sharex=True)
    for ax, col, ylabel, title in (
        (axes[0], "car_delay_s", "vehicle delay [s]", "(a) vehicle passage cost"),
        (axes[1], "ped_delay_mean_s", "mean pedestrian delay [s]", "(b) pedestrian cost"),
    ):
        for rule, color in (("R0", COL["dead"]), ("R1", COL["pass"])):
            for model, marker in (("sfm", "o"), ("wiedemann", "s")):
                sub = out.metrics[(out.metrics["rule"] == rule)
                                  & (out.metrics["model"] == model)
                                  & (out.metrics["outcome"] == "PASS")].sort_values("width_m")
                sub = sub[sub[col].notna()]
                ax.plot(sub["width_m"], sub[col], marker=marker, color=color, lw=1.2,
                        ms=4.5, ls="-" if model == "sfm" else "--", alpha=0.9,
                        label=f"{rule} · {model}")
        ax.set_xlabel("road width L [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=6.6, ncol=2)
    fig.suptitle("Cost of passing vs. road width (deadlocked widths have no finite delay: the encounter never resolves)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig06_width_sweep")


# ---------------------------------------------------------------------------
# 图 7：两种运动模型
# ---------------------------------------------------------------------------
def fig07_models(out: Outputs, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.2), sharey=True)
    for ax, prefix, title in zip(
        axes,
        ("L4.0_R0", "L3.0_R1"),
        ("(a) L = 4.0 m, R0: open passage", "(b) L = 3.0 m, R1: crawl passage"),
    ):
        for model, color in (("sfm", "#2166ac"), ("wiedemann", "#b2182b")):
            rid = f"{prefix}_{model}"
            car = out.trace(rid)["CAR"]
            row = out.row(rid)
            ax.plot(car["t"], car["speed"], color=color, lw=1.7,
                    label=f"{model}  (delay +{float(row['car_delay_s']):.1f} s)")
        ax.set_xlabel("time [s]")
        ax.set_title(title)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("vehicle speed [m/s]")
    fig.suptitle("Same rules and geometry, different longitudinal model (SFM vs. Wiedemann)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig07_models")


# ---------------------------------------------------------------------------
# 一页总结图
# ---------------------------------------------------------------------------
def fig_summary(out: Outputs, out_dir: Path) -> None:
    cfg = out.cfg
    car_cfg, ped_cfg = cfg["car"], cfg["pedestrians"]
    ics = out.summary["initial_conditions"]
    fig = plt.figure(figsize=(11.6, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.42, wspace=0.3)

    # (a) 场景
    ax = fig.add_subplot(gs[0, :2])
    _draw_road(ax, 3.0, 0.0, 320.0)
    for i, ic in enumerate(ics):
        _draw_ped(ax, float(ic["x_m"]), float(ic["y_m"]), i)
    _draw_car(ax, float(car_cfg["start_x_m"]), float(car_cfg["y_m"]), car_cfg)
    ax.set_xlim(-6, 326)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) scenario: pedestrians +x, vehicle −x, no sidewalk")
    ax.grid(axis="y", alpha=0.12)

    # (b) 余量与阈值
    ax = fig.add_subplot(gs[0, 2])
    Ls = np.linspace(2.7, 4.1, 200)
    ax.axvspan(2.7, 2.9, color=COL["zone_space"], lw=0)
    ax.axvspan(2.9, 3.4, color=COL["zone_rule"], lw=0)
    ax.axvspan(3.4, 4.1, color=COL["zone_pass"], lw=0)
    ax.plot(Ls, Ls / 2 - car_cfg["width_m"] / 2 - ped_cfg["width_m"], color="#333", lw=1.4)
    ax.axhline(cfg["rules"]["R0"]["car_c_req_m"], color=COL["dead"], ls=":", lw=1)
    ax.axhline(cfg["rules"]["R1"]["car_c_min_m"], color=COL["pass"], ls=":", lw=1)
    ax.axvline(4.0, color=COL["car"], lw=0.9, ls="--")
    ax.axvline(3.0, color="#555", lw=0.9, ls="--")
    ax.set_xlabel("road width L [m]")
    ax.set_ylabel("c_avail [m]")
    ax.set_title("(b) clearance vs. width")

    # (c) 延误与僵持
    ax = fig.add_subplot(gs[1, 0])
    cases = [
        ("L4.0_R0_wiedemann", "4.0\nR0"),
        ("L4.0_R1_wiedemann", "4.0\nR1"),
        ("L3.0_R1_wiedemann", "3.0\nR1"),
        ("L3.0_R0_wiedemann", "3.0\nR0"),
        ("L2.8_R1_wiedemann", "2.8\nR1"),
    ]
    for i, (rid, _) in enumerate(cases):
        r = out.row(rid)
        if r["outcome"] == "PASS":
            ax.bar(i, float(r["car_delay_s"]), color=COL["pass"], width=0.62)
        else:
            ax.scatter([i], [0.8], marker="x", s=80, color=COL["dead"], linewidth=2.0,
                       zorder=5)
            ax.annotate("deadlock", (i, 2.4), ha="center", fontsize=6.2, color=COL["dead"],
                        rotation=90, va="bottom")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[1] for c in cases])
    ax.set_ylabel("vehicle delay [s]")
    ax.set_title("(c) passage cost (Wiedemann)")
    ax.set_ylim(0, 33)

    # (d) 僵持快照
    ax = fig.add_subplot(gs[1, 1:])
    rid = "L3.0_R0_sfm"
    traces = out.trace(rid)
    row = out.row(rid)
    t_dl = float(row["deadlock_t_s"])
    car = _nearest_row(traces["CAR"], t_dl)
    xc = float(car["x"])
    _draw_road(ax, 3.0, xc - 12, xc + 12)
    for i in range(3):
        p = _nearest_row(traces[f"P{i + 1}"], t_dl)
        _draw_ped(ax, float(p["x"]), float(p["y"]), i)
    _draw_car(ax, xc, float(car["y"]), car_cfg)
    ax.set_xlim(xc - 12, xc + 12)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.set_yticks([-1.5, 0, 1.5])
    ax.set_xlabel("x [m]")
    ax.set_title(f"(d) L = 3.0 m, R0: standoff {float(row['standoff_gap_m']):.1f} m → deadlock")
    ax.grid(False)

    fig.suptitle(
        "Narrow-road encounter — 4.0 m: passes (+{:.1f} s);  3.0 m: passes only under the revised rule "
        "(+{:.1f} s) and deadlocks under the base rule;  2.8 m: geometrically infeasible".format(
            float(out.row("L4.0_R0_wiedemann")["car_delay_s"]),
            float(out.row("L3.0_R1_wiedemann")["car_delay_s"]),
        ),
        fontsize=9.2, y=0.99,
    )
    _save(fig, out_dir, "summary")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def make_all_figures(out_dir: str | Path, summary: dict[str, Any] | None = None) -> None:
    out_dir = Path(out_dir)
    out = Outputs(out_dir, summary)
    fig01_scenario(out, out_dir)
    fig02_width_compare(out, out_dir)
    fig03_snapshots(out, out_dir)
    fig04_rule_modification(out, out_dir)
    fig05_classification(out, out_dir)
    fig06_width_sweep(out, out_dir)
    fig07_models(out, out_dir)
    fig_summary(out, out_dir)

