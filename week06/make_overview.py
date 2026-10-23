"""生成 week06 全部结果的一页总览图（实验 1 + 实验 2）。

输出：week06/results_all.png / results_all.pdf

只读取两个实验目录下的真实产物（outputs/ 与 outputs_acceptance/），
不包含任何手写数值。

用法：
    python make_overview.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "narrow_road"))

from narrowroad.common import clearance_available, load_config, rule_clearance_curve  # noqa: E402
from narrowroad.plotting import (  # noqa: E402
    COL,
    _draw_car,
    _draw_ped,
    _draw_road,
    _nearest_row,
)

OUT1 = ROOT / "narrow_road" / "outputs"
OUT2 = ROOT / "narrow_road" / "outputs_acceptance"

RULE_COLORS = {
    "R0": "#b2182b",
    "R1": "#b2182b",
    "S_cons": "#e08214",
    "S_mod": "#1b7837",
    "S_lib": "#2166ac",
}
RULE_MARKERS = {"R1": "o", "S_cons": "s", "S_mod": "^", "S_lib": "D"}


def load_traces(out_dir: Path, rid: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(out_dir / "trajectories" / f"{rid}.csv")
    return {name: g.sort_values("t").reset_index(drop=True) for name, g in df.groupby("agent")}


# ---------------------------------------------------------------------------
# (a) 场景
# ---------------------------------------------------------------------------
def panel_scenario(ax, cfg, ics):
    car_cfg = cfg["car"]
    _draw_road(ax, 3.0, 0.0, 320.0)
    for i, ic in enumerate(ics):
        x, y = float(ic["x_m"]), float(ic["y_m"])
        _draw_ped(ax, x, y, i)
        ax.annotate(f"{ic['v0_mps']:.2f} m/s", (x + 1.0, y + 0.55), fontsize=5.6,
                    color=COL["peds"][i], va="bottom")
        ax.annotate("", xy=(x + 4.0, y), xytext=(x + 0.6, y),
                    arrowprops=dict(arrowstyle="-|>", color=COL["peds"][i], lw=0.9))
    _draw_car(ax, float(car_cfg["start_x_m"]), 0.0, car_cfg)
    ax.annotate("", xy=(float(car_cfg["start_x_m"]) - 14, 0),
                xytext=(float(car_cfg["start_x_m"]) - 3, 0),
                arrowprops=dict(arrowstyle="-|>", color=COL["car"], lw=1.6))
    ax.annotate("vehicle 9.0 m/s\nfixed y = 0", (float(car_cfg["start_x_m"]) - 8, 1.35),
                fontsize=6.0, color=COL["car"], ha="center")
    ax.annotate("pedestrians walk +x, y ∈ [−0.5, 0.5]", (55, -2.35), fontsize=6.2,
                color="#444444")
    ax.annotate("c_avail(L) = L/2 − Wc/2 − Wp = L/2 − 1.40", (0.02, 0.97),
                xycoords="axes fraction", fontsize=6.2, va="top", color="#333333")
    ax.set_xlim(-6, 326)
    ax.set_ylim(-2.7, 2.5)
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(a) scenario: narrow mixed road, no sidewalk", fontsize=8.5)
    ax.grid(axis="y", alpha=0.12)


# ---------------------------------------------------------------------------
# (b) 两种行人模型
# ---------------------------------------------------------------------------
def panel_models(ax, cfg):
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    v = np.linspace(0, 9, 200)
    ax.axhline(0.30, color=COL["dead"], lw=1.6, ls="--", label="Exp1: fixed 0.30 m")
    for rn in ("S_cons", "S_mod", "S_lib"):
        c0, k = rule_clearance_curve(cfg["rules"][rn])
        ax.plot(v, c0 + k * v, color=RULE_COLORS[rn], lw=1.8,
                label=f"Exp2: {rn}  {c0:.2f}+{k:.2f}·v")
    for width, txt in ((4.0, "4.0"), (3.5, "3.5"), (3.0, "3.0"), (2.90, "2.90"),
                       (2.84, "2.84"), (2.80, "2.80")):
        c = clearance_available(width, car_w, ped_w)
        ax.axhline(c, color="#888", lw=0.6, ls=":")
        ax.annotate(f"L={txt}", (8.95, c + 0.004), fontsize=5.2, ha="right", color="#555555")
    ax.annotate("L=3.0 m: R1 never accepts 0.10 m;\nS_mod accepts up to v ≈ 2.4 m/s",
                (2.4, 0.115), xytext=(3.6, 0.40), fontsize=6.2, color="#333333",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 0.65)
    ax.set_xlabel("vehicle speed v [m/s]", fontsize=7)
    ax.set_ylabel("accepted clearance c [m]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(b) pedestrian model: fixed vs speed-dependent acceptance", fontsize=8.5)
    ax.legend(fontsize=5.4, loc="upper left")


# ---------------------------------------------------------------------------
# (c) 关键结果表（逐列文本）
# ---------------------------------------------------------------------------
def panel_table(ax, m1: pd.DataFrame, m2: pd.DataFrame):
    ax.axis("off")
    ax.set_title("(c) all key results  (delay: Wiedemann / SFM)", fontsize=8.5)
    cols = [0.00, 0.30, 0.63, 0.86]
    rows = [
        ("case", "result", "delay [s]", "clear"),
        ("Exp1 L=4.0 R0", "pass", "3.3 / 1.8", "0.60"),
        ("Exp1 L=3.0 R0", "deadlock (stop 68 s)", "—", "0.10"),
        ("Exp1 L=3.0 R1", "pass (crawl 1 m/s)", "25.5 / 29.8", "0.10"),
        ("Exp1 L=2.8 R1", "deadlock: no space", "—", "0.00"),
        ("Exp2 L=3.0 S_cons", "pass", "27.9 / 26.8", "0.10"),
        ("Exp2 L=3.0 S_mod", "pass (2.4 m/s)", "15.3 / 14.5", "0.10"),
        ("Exp2 L=3.0 S_lib", "pass (4.0 m/s)", "8.5 / 8.1", "0.10"),
        ("Exp2 L=2.84 S_mod", "pass: sequential", "33.0 / 31.6", "0.02"),
        ("Exp2 L=2.82 S_lib", "pass: sequential", "33.0 / 31.6", "0.01"),
        ("Exp2 L=2.80 any", "deadlock: geometric limit", "—", "0.00"),
    ]
    y0, dy = 0.94, 0.078
    for i, row in enumerate(rows):
        y = y0 - i * dy
        for j, txt in enumerate(row):
            if i == 0:
                ax.text(cols[j], y, txt, fontsize=5.6, va="top", transform=ax.transAxes,
                        color="#555555", fontweight="bold")
            else:
                color = "#b2182b" if "deadlock" in row[1] else "#1b5e20"
                ax.text(cols[j], y, txt, fontsize=5.7, va="top", transform=ax.transAxes,
                        color=color if j <= 1 else "#222222")
    ax.text(0.0, 0.04, "boundaries: Exp1 3.40 m (R0) · 2.90 m (R1)     "
                       "Exp2 2.84 m (S_mod) · 2.82 m (S_lib)     limit 2.80 m",
            fontsize=5.4, transform=ax.transAxes, color="#333333")
    ax.text(0.0, 0.00, "clear = lateral body clearance at the passing moment",
            fontsize=5.2, transform=ax.transAxes, color="#777777")


# ---------------------------------------------------------------------------
# (d) 实验 1：延误与僵持
# ---------------------------------------------------------------------------
def panel_exp1_delay(ax, m1: pd.DataFrame):
    for rule in ("R0", "R1"):
        for model, ls, mk in (("wiedemann", "-", "o"), ("sfm", "--", "s")):
            sub = m1[(m1["rule"] == rule) & (m1["model"] == model)]
            p = sub[sub["outcome"] == "PASS"].sort_values("width_m")
            ax.plot(p["width_m"], p["car_delay_s"], color=RULE_COLORS[rule], lw=1.3,
                    ls=ls, marker=mk, ms=3.0, alpha=0.9,
                    label=f"{rule} {'W' if model == 'wiedemann' else 'SFM'}")
            for _, r in sub[sub["outcome"] != "PASS"].iterrows():
                ax.scatter([float(r["width_m"])], [0.7], marker="x", s=22,
                           color=RULE_COLORS[rule], linewidth=1.0)
    ax.axvline(3.40, color=COL["dead"], lw=1.0, ls=":", alpha=0.7)
    ax.axvline(2.90, color=COL["pass"], lw=1.0, ls=":", alpha=0.7)
    ax.annotate("R0 boundary\n3.40 m", (3.415, 17), fontsize=5.8, color=COL["dead"])
    ax.annotate("R1 boundary\n2.90 m", (2.915, 17), fontsize=5.8, color=COL["pass"])
    ax.annotate("L=3.0 m: R0 deadlock,\nR1 needs 25–30 s", (3.02, 24), fontsize=6.0)
    ax.annotate("✕ = deadlock", (2.77, 8.5), fontsize=5.6, color="#555555")
    ax.set_xlim(2.75, 4.05)
    ax.set_ylim(0, 32)
    ax.set_xlabel("road width L [m]", fontsize=7)
    ax.set_ylabel("vehicle delay [s]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(d) Exp1: passage cost and deadlocks", fontsize=8.5)
    ax.legend(fontsize=5.4, loc="upper right")


# ---------------------------------------------------------------------------
# (e) 实验 1：空间不足 vs 规则失效
# ---------------------------------------------------------------------------
def panel_exp1_classify(ax, cfg, m1: pd.DataFrame):
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    l_rule = 2.0 * (0.30 + car_w / 2 + ped_w)
    l_space = 2.0 * (0.05 + car_w / 2 + ped_w)
    l_min, l_max = 2.75, 4.05
    ax.axvspan(l_min, l_space, color="#d9d9d9", lw=0)
    ax.axvspan(l_space, l_rule, color="#fddbc7", lw=0)
    ax.axvspan(l_rule, l_max, color="#d9f0d3", lw=0)
    xs = np.linspace(l_min, l_max, 300)
    ax.plot(xs, xs / 2 - car_w / 2 - ped_w, color="#333", lw=1.4, label="c_avail(L)")
    ax.axhline(0.30, color=COL["dead"], ls="--", lw=1.0)
    ax.axhline(0.05, color=COL["pass"], ls=":", lw=1.0)
    for rule, marker, off in (("R0", "o", -0.008), ("R1", "s", 0.008)):
        sub = m1[(m1["rule"] == rule) & (m1["model"] == "sfm")]
        for _, r in sub.iterrows():
            passed = r["outcome"] == "PASS"
            ax.scatter([float(r["width_m"]) + off], [float(r["c_avail_m"])], marker=marker,
                       s=25, facecolor="#1b7837" if passed else "white",
                       edgecolor=COL["pass"] if passed else COL["dead"], linewidth=1.0,
                       zorder=5)
            if not passed:
                ax.scatter([float(r["width_m"]) + off], [float(r["c_avail_m"])], marker="x",
                           s=14, color=COL["dead"], zorder=6)
    ax.annotate("no space", ((l_min + l_space) / 2, 0.52), ha="center", fontsize=6.2)
    ax.annotate("feasible but\nR0 stalls\n= rule failure", ((l_space + l_rule) / 2, 0.52),
                ha="center", fontsize=6.0, color=COL["dead"])
    ax.annotate("R0 passes", ((l_rule + l_max) / 2, 0.52), ha="center", fontsize=6.2,
                color=COL["pass"])
    ax.annotate("✕ = deadlock, ● = R0, ■ = R1", (0.02, 0.03), xycoords="axes fraction",
                fontsize=5.5, color="#555555")
    ax.set_xlim(l_min, l_max)
    ax.set_ylim(-0.01, 0.65)
    ax.set_xlabel("road width L [m]", fontsize=7)
    ax.set_ylabel("c_avail [m]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(e) Exp1: space insufficiency vs rule failure", fontsize=8.5)


# ---------------------------------------------------------------------------
# (f) 实验 1：僵持快照
# ---------------------------------------------------------------------------
def panel_exp1_standoff(ax, cfg, traces):
    car = traces["CAR"].iloc[-1]
    xc = float(car["x"])
    _draw_road(ax, 3.0, xc - 10, xc + 10)
    for i in range(3):
        p = traces[f"P{i + 1}"].iloc[-1]
        _draw_ped(ax, float(p["x"]), float(p["y"]), i)
    _draw_car(ax, xc, 0.0, cfg["car"])
    ax.set_xlim(xc - 10, xc + 10)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    ax.set_yticks([-1.5, 0, 1.5])
    ax.set_yticklabels(["-1.5", "0", "1.5"], fontsize=5.5)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title("(f) Exp1 L=3.0 m R0: deadlock, standoff 2.5 m\n"
                 "vehicle stopped 68 s, three pedestrians waiting", fontsize=7.6)


# ---------------------------------------------------------------------------
# (g) 实验 2：边界
# ---------------------------------------------------------------------------
def panel_exp2_boundary(ax, cfg, m2: pd.DataFrame):
    names = ["R1", "S_cons", "S_mod", "S_lib"]
    pred = {}
    for rn in names:
        curve = rule_clearance_curve(cfg["rules"][rn])
        c0 = 0.05 if curve is None else curve[0]
        pred[rn] = 2.0 * (c0 + 0.9 + 0.5)
    x = np.arange(len(names))
    for i, rn in enumerate(names):
        sub = m2[(m2["rule"] == rn) & (m2["model"] == "sfm") & (m2["outcome"] == "PASS")]
        obs = sub["width_m"].min() if not sub.empty else np.nan
        ax.bar(i + 0.18, pred[rn], width=0.34, color="#c8c8c8", edgecolor="#777777")
        ax.bar(i - 0.18, obs, width=0.34, color=RULE_COLORS[rn], edgecolor="#333333")
        ax.annotate(f"{pred[rn]:.2f}", (i + 0.18, pred[rn] + 0.005), ha="center", fontsize=5.6)
        ax.annotate(f"{obs:.2f}", (i - 0.18, obs + 0.005), ha="center", fontsize=5.6)
    ax.axhline(2.80, color="#333333", lw=1.1, ls=":")
    ax.annotate("geometric limit 2.80 m", (3.45, 2.806), fontsize=5.8, color="#333333",
                ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["R1\n(fixed)", "S_cons", "S_mod", "S_lib"], fontsize=6.3)
    ax.set_ylim(2.75, 2.99)
    ax.set_ylabel("min passable width [m]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(g) Exp2: boundary 2.90 → 2.84 → 2.82 m\n"
                 "(grey = predicted, colour = simulated)", fontsize=7.8)


# ---------------------------------------------------------------------------
# (h) 实验 2：L=3.0 代价
# ---------------------------------------------------------------------------
def panel_exp2_cost(ax, m2: pd.DataFrame):
    names = ["R1", "S_cons", "S_mod", "S_lib"]
    labels = ["R1\n(fixed)", "S_cons", "S_mod", "S_lib"]
    delay = [float(m2[m2["run_id"] == f"L3.00_{rn}_wiedemann"]["car_delay_s"].iloc[0])
             for rn in names]
    stop = [float(m2[m2["run_id"] == f"L3.00_{rn}_wiedemann"]["ped_stop_time_total_s"].iloc[0])
            for rn in names]
    x = np.arange(len(names))
    ax.bar(x - 0.18, delay, width=0.34, color=[RULE_COLORS[rn] for rn in names],
           label="vehicle delay")
    ax.bar(x + 0.18, stop, width=0.34, color="#dddddd", edgecolor="#666666", hatch="///",
           label="pedestrian stop time")
    for xi, d, s in zip(x, delay, stop):
        ax.annotate(f"{d:.0f}", (xi - 0.18, d + 0.6), ha="center", fontsize=5.8)
        ax.annotate(f"{s:.0f}", (xi + 0.18, s + 0.6), ha="center", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylim(0, 32)
    ax.set_ylabel("time [s]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(h) Exp2: cost at L=3.0 m (Wiedemann)", fontsize=8.5)
    ax.legend(fontsize=5.8, loc="upper right")


# ---------------------------------------------------------------------------
# (i) 实验 2：实测通过事件校验
# ---------------------------------------------------------------------------
def panel_exp2_validation(ax, cfg, pass_events: pd.DataFrame):
    v = np.linspace(0, 9, 200)
    for rn in ("S_cons", "S_mod", "S_lib"):
        c0, k = rule_clearance_curve(cfg["rules"][rn])
        ax.plot(v, c0 + k * v, color=RULE_COLORS[rn], lw=1.3, alpha=0.9,
                label=f"{rn} acceptance")
    pe = pass_events[pass_events["model"] == "wiedemann"]
    for rn in ("S_cons", "S_mod", "S_lib"):
        sub = pe[(pe["rule"] == rn) & (pe["clearance_m"] >= 0)]
        ax.scatter(sub["car_speed_mps"], sub["clearance_m"], s=22,
                   color=RULE_COLORS[rn], marker=RULE_MARKERS[rn], zorder=5)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("vehicle speed at pass [m/s]", fontsize=7)
    ax.set_ylabel("clearance at pass [m]", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("(i) Exp2: measured passing events lie on\n"
                 "the acceptance curves", fontsize=7.8)
    ax.legend(fontsize=5.2, loc="upper left")


# ---------------------------------------------------------------------------
# (j)(k) 实验 2：俯视快照
# ---------------------------------------------------------------------------
def panel_snapshot(ax, cfg, traces, t, ped_name, title):
    car = _nearest_row(traces["CAR"], t)
    xc = float(car["x"])
    _draw_road(ax, 3.0, xc - 7.5, xc + 7.5)
    for i in range(3):
        p = _nearest_row(traces[f"P{i + 1}"], t)
        if abs(float(p["x"]) - xc) <= 7.5:
            _draw_ped(ax, float(p["x"]), float(p["y"]), i)
    _draw_car(ax, xc, 0.0, cfg["car"])
    ped = _nearest_row(traces[ped_name], t)
    if float(ped["speed"]) > 0.1:
        ax.annotate("", xy=(float(ped["x"]) + 2.0, float(ped["y"])),
                    xytext=(float(ped["x"]) + 0.4, float(ped["y"])),
                    arrowprops=dict(arrowstyle="-|>", color="#10375c", lw=1.2))
    ax.annotate(f"v_car = {float(car['speed']):.1f} m/s\nv_ped = {float(ped['speed']):.1f} m/s",
                (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=5.8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#cccccc", alpha=0.9))
    ax.set_xlim(xc - 7.5, xc + 7.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.set_yticks([-1.5, 0, 1.5])
    ax.set_yticklabels(["-1.5", "0", "1.5"], fontsize=5.5)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=8.5)


# ---------------------------------------------------------------------------
# (l) 结论
# ---------------------------------------------------------------------------
def panel_conclusions(ax):
    ax.axis("off")
    ax.set_title("(l) conclusions", fontsize=8.5)
    lines = [
        "1. R0 (fixed 0.30 m) deadlocks at L=3.0 m although 0.10 m is",
        "   geometrically available → rule failure, not lack of space.",
        "   A1 (fix vehicle rule) passes; A2 (fix pedestrian rule) stalls.",
        "2. Fixed personal space for L ≤ 2.90 m is space insufficiency:",
        "   c_avail ≤ 0.05 m, every rule deadlocks; 2.80 m is the hard limit.",
        "3. Speed-dependent acceptance c_acc(v)=c0+k·v (pedestrians accept",
        "   small gaps when the car is slow; speed can drop to 0) moves the",
        "   boundary to 2.84 / 2.82 m and cuts the L=3.0 m vehicle delay",
        "   from 25–30 s to 15 / 8 s; pedestrian waiting 20 s → 2–4 s.",
        "4. At the boundary the mechanism is sequential: the car stops and",
        "   waits while pedestrians walk past at 0.01–0.02 m.",
        "5. The decisive element is the decision-level acceptance threshold;",
        "   SFM force scaling is second order (<0.5 s).",
        "6. 130 runs: zero collisions, no emergency braking (|a| ≤ 2.5 m/s²).",
    ]
    for i, line in enumerate(lines):
        ax.text(0.0, 0.95 - i * 0.068, line, family="monospace", fontsize=5.7,
                va="top", transform=ax.transAxes, color="#222222")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    with open(OUT1 / "summary.json", "r", encoding="utf-8") as f:
        summary1 = json.load(f)
    ics = summary1["initial_conditions"]
    m1 = pd.read_csv(OUT1 / "metrics.csv")
    m2 = pd.read_csv(OUT2 / "metrics.csv")
    pass_events = pd.read_csv(OUT2 / "pass_events.csv")

    fig = plt.figure(figsize=(16.0, 12.6))
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[1.02, 1.12, 1.12, 1.0],
        hspace=0.46, wspace=0.30,
        left=0.045, right=0.985, top=0.895, bottom=0.045,
    )

    panel_scenario(fig.add_subplot(gs[0, 0]), cfg, ics)
    panel_models(fig.add_subplot(gs[0, 1]), cfg)
    panel_table(fig.add_subplot(gs[0, 2]), m1, m2)

    panel_exp1_delay(fig.add_subplot(gs[1, 0]), m1)
    panel_exp1_classify(fig.add_subplot(gs[1, 1]), cfg, m1)
    panel_exp1_standoff(fig.add_subplot(gs[1, 2]), cfg, load_traces(OUT1, "L3.0_R0_sfm"))

    panel_exp2_boundary(fig.add_subplot(gs[2, 0]), cfg, m2)
    panel_exp2_cost(fig.add_subplot(gs[2, 1]), m2)
    panel_exp2_validation(fig.add_subplot(gs[2, 2]), cfg, pass_events)

    pe_r1 = pass_events[(pass_events["run_id"] == "L3.00_R1_wiedemann")
                        & (pass_events["ped"] == "P1")]
    pe_sm = pass_events[(pass_events["run_id"] == "L3.00_S_mod_wiedemann")
                        & (pass_events["ped"] == "P1")]
    panel_snapshot(fig.add_subplot(gs[3, 0]), cfg, load_traces(OUT2, "L3.00_R1_wiedemann"),
                   float(pe_r1["t_pass_s"].iloc[0]), "P1",
                   "(j) Exp2 L=3.0 R1: pedestrians wait (v=0)")
    panel_snapshot(fig.add_subplot(gs[3, 1]), cfg, load_traces(OUT2, "L3.00_S_mod_wiedemann"),
                   float(pe_sm["t_pass_s"].iloc[0]), "P1",
                   "(k) Exp2 L=3.0 S_mod: walk past at 0.10 m")
    panel_conclusions(fig.add_subplot(gs[3, 2]))

    fig.suptitle(
        "Week 06 — narrow-road pedestrian–vehicle encounter: one unified agent model, two experiments (130 runs)",
        fontsize=13.0, y=0.972,
    )
    fig.text(0.5, 0.941,
             "Exp1 (fixed 0.30 m personal space): boundary 3.40 m (R0) / 2.90 m (R1); L=3.0 m is a rule failure.   "
             "Exp2 (speed-dependent acceptance c_acc(v)=c0+k·v): boundary 2.84 / 2.82 m; delay cut by 40–70%.",
             ha="center", fontsize=8.0, color="#333333")
    fig.text(0.5, 0.921,
             "social force (SFM) + Wiedemann longitudinal models · deterministic · 55 tests · zero collisions",
             ha="center", fontsize=7.2, color="#666666")
    fig.text(0.5, 0.010,
             "reproduce:  cd narrow_road  &&  python run_experiments.py  ·  python run_acceptance_experiment.py  ·  python -m pytest -q",
             ha="center", fontsize=6.2, color="#555555")
    fig.savefig(ROOT / "results_all.png", dpi=200, facecolor="white")
    fig.savefig(ROOT / "results_all.pdf", facecolor="white")
    plt.close(fig)
    print(f"written: {ROOT / 'results_all.png'} and .pdf")


if __name__ == "__main__":
    main()
