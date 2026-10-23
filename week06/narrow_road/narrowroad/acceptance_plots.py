"""实验 2 绘图：速度相关接受模型、边界、通行代价、现象验证与俯视快照。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

from .acceptance import predicted_boundary
from .common import clearance_available, load_config, rule_clearance_curve
from .plotting import COL, _draw_car, _draw_ped, _draw_road, _nearest_row, _save

RULE_COLORS = {
    "R1": COL["dead"],
    "S_cons": "#e08214",
    "S_mod": "#1b7837",
    "S_lib": "#2166ac",
}
RULE_LABELS = {
    "R1": "R1: fixed 0.30 m personal space",
    "S_cons": "S_cons: speed-dependent (0.05+0.05v)",
    "S_mod": "S_mod: speed-dependent (0.02+0.03v)",
    "S_lib": "S_lib: speed-dependent (0.01+0.02v)",
}
RULE_MARKERS = {"R1": "o", "S_cons": "s", "S_mod": "^", "S_lib": "D"}


class AccOutputs:
    """读取 outputs_acceptance/ 下的产物。"""

    def __init__(self, out_dir: str | Path, summary: dict[str, Any] | None = None):
        self.dir = Path(out_dir)
        self.metrics = pd.read_csv(self.dir / "metrics.csv")
        self.ped = pd.read_csv(self.dir / "ped_metrics.csv")
        pe = self.dir / "pass_events.csv"
        self.pass_events = pd.read_csv(pe) if pe.exists() else pd.DataFrame()
        if summary is None:
            with open(self.dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
        self.summary = summary
        self.cfg = load_config()
        self._traces: dict[str, dict[str, pd.DataFrame]] = {}

    def trace(self, rid: str) -> dict[str, pd.DataFrame]:
        if rid not in self._traces:
            df = pd.read_csv(self.dir / "trajectories" / f"{rid}.csv")
            self._traces[rid] = {
                name: g.sort_values("t").reset_index(drop=True) for name, g in df.groupby("agent")
            }
        return self._traces[rid]

    def row(self, rid: str) -> pd.Series:
        return self.metrics[self.metrics["run_id"] == rid].iloc[0]


# ---------------------------------------------------------------------------
# 图 A1：接受模型本身
# ---------------------------------------------------------------------------
def figA1_model(out: AccOutputs, out_dir: Path) -> None:
    cfg = out.cfg
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    v = np.linspace(0, 9.0, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8))

    # (a) 接受阈值 c_acc(v) 与各宽度的几何余量
    ax = axes[0]
    rules = ["S_cons", "S_mod", "S_lib"]
    for rn in rules:
        c0, k = rule_clearance_curve(cfg["rules"][rn])
        ax.plot(v, c0 + k * v, color=RULE_COLORS[rn], lw=1.8,
                label=f"{rn}: c_acc = {c0:.2f} + {k:.2f}·v")
    ax.axhline(float(cfg["pedestrians"]["c_pass_m"]), color=COL["dead"], lw=1.4, ls="--",
               label="R1 pedestrians: fixed 0.30 m")
    label_pos = {4.0: (8.9, "right"), 3.5: (8.9, "right"), 3.0: (8.9, "right"),
                 2.90: (0.15, "left"), 2.84: (1.7, "left"), 2.80: (3.3, "left")}
    for width in (4.0, 3.5, 3.0, 2.90, 2.84, 2.80):
        c = clearance_available(width, car_w, ped_w)
        if c <= 0.65:
            ax.axhline(c, color="#777777", lw=0.7, ls=":")
            lx, ha = label_pos[width]
            ax.annotate(f"L={width:.2f}", (lx, c + 0.006), fontsize=6.0, va="bottom",
                        ha=ha, color="#555555")
    ax.annotate(
        "L=3.0: S_mod accepts up to\nv = (0.10 - 0.02)/0.03 = 2.7 m/s",
        (2.65, 0.10), xytext=(4.3, 0.095), fontsize=7.0, color=RULE_COLORS["S_mod"],
        arrowprops=dict(arrowstyle="->", lw=0.8, color=RULE_COLORS["S_mod"]),
    )
    ax.annotate(
        "R1: 0.30 m is never reached at L=3.0\n-> pedestrians always wait",
        (1.1, 0.30), xytext=(4.8, 0.40), fontsize=7.0, color=COL["dead"],
        arrowprops=dict(arrowstyle="->", lw=0.8, color=COL["dead"]),
    )
    ax.set_xlabel("vehicle speed v [m/s]")
    ax.set_ylabel("accepted lateral clearance c_acc [m]")
    ax.set_title("(a) speed-dependent personal space:  c_acc(v) = c0 + k·v")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=6.6, loc="upper left")

    # (b) 各规则允许的车辆通过速度 vs 道路宽度
    ax = axes[1]
    widths = np.linspace(2.78, 4.05, 300)
    for rn in ("R1", "S_cons", "S_mod", "S_lib"):
        rule = cfg["rules"][rn]
        c0, k = rule_clearance_curve(rule)
        sf = float(rule.get("car_speed_factor", 1.0))
        cap = np.maximum(0.0, sf * (widths / 2 - car_w / 2 - ped_w - c0) / k)
        ax.plot(widths, cap, color=RULE_COLORS[rn], lw=1.8,
                label=f"{rn}: boundary L={2*(c0+car_w/2+ped_w):.2f} m")
    ax.axvline(2.80, color="#555", lw=0.9, ls=":")
    ax.annotate("geometric limit 2.80 m", (2.815, 9.2), fontsize=6.6, color="#555")
    ax.annotate("L=3.0 m: 1.0 / 2.4 / 4.0 m/s", (3.02, 2.2), fontsize=6.6, color="#333333")
    ax.set_xlabel("road width L [m]")
    ax.set_ylabel("allowed vehicle passing speed [m/s]")
    ax.set_title("(b) implied passing speed  v_allow = 0.9·(c_avail − c0)/k")
    ax.set_xlim(2.78, 4.05)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=6.6, loc="upper left")
    fig.suptitle("What model fits 'pedestrians squeeze past a very slow car'?  "
                 "A shared, speed-dependent gap-acceptance relation", fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "figA1_model")


# ---------------------------------------------------------------------------
# 图 A2：可通行边界
# ---------------------------------------------------------------------------
def figA2_boundary(out: AccOutputs, out_dir: Path) -> None:
    cfg = out.cfg
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    l_min, l_max = 2.78, 4.02
    xs = np.linspace(l_min, l_max, 400)

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.8), sharex=True, sharey=True)
    for ax, rn in zip(axes.flat, ["R1", "S_cons", "S_mod", "S_lib"]):
        rule = cfg["rules"][rn]
        color = RULE_COLORS[rn]
        ax.plot(xs, xs / 2 - car_w / 2 - ped_w, color="#333333", lw=1.4, label="c_avail(L)")
        b = predicted_boundary(cfg, rn)
        if rule.get("ped_acceptance") == "speed":
            c0 = float(rule["car_c_min_m"])
            ax.axhline(c0, color=color, lw=1.1, ls="--",
                       label=f"pedestrian floor c0 = {c0:.2f} m")
        else:
            ax.axhline(float(cfg["pedestrians"]["c_pass_m"]), color=color, lw=1.1, ls="--",
                       label="pedestrian threshold 0.30 m")
            ax.axhline(b["c_req_m"], color="#888888", lw=0.9, ls=":",
                       label=f"vehicle needs c > {b['c_req_m']:.2f}")
        ax.axvline(b["l_crit_m"], color=color, lw=1.2, ls="-.",
                   label=f"predicted boundary {b['l_crit_m']:.2f} m")
        ax.axvline(2.80, color="#555555", lw=0.8, ls=":")

        sub = out.metrics[(out.metrics["rule"] == rn) & (out.metrics["model"] == "wiedemann")]
        for _, r in sub.iterrows():
            passed = r["outcome"] == "PASS"
            ax.scatter([float(r["width_m"])], [float(r["c_avail_m"])], marker=RULE_MARKERS[rn],
                       s=48, facecolor=color if passed else "white", edgecolor=color,
                       linewidth=1.2, zorder=5)
            if not passed:
                ax.scatter([float(r["width_m"])], [float(r["c_avail_m"])], marker="x", s=26,
                           color=color, zorder=6, linewidth=1.1)
        ax.set_title(f"{RULE_LABELS[rn]}   →  boundary {b['l_crit_m']:.2f} m", fontsize=7.8)
        ax.set_ylim(-0.01, 0.65)
        ax.set_xlim(l_min, l_max)
        ax.legend(fontsize=5.8, loc="upper left", framealpha=0.9)
    for ax in axes[1]:
        ax.set_xlabel("road width L [m]")
    for ax in axes[:, 0]:
        ax.set_ylabel("available clearance [m]")
    fig.suptitle("Passability boundary per rule (Wiedemann): filled marker = PASS, ✕ = deadlock, "
                 "dash-dot = predicted boundary", fontsize=9, y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "figA2_boundary")


# ---------------------------------------------------------------------------
# 图 A3：通行代价
# ---------------------------------------------------------------------------
def figA3_cost(out: AccOutputs, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), sharex=True)
    for ax, col, ylabel, title in (
        (axes[0], "car_delay_s", "vehicle delay [s]", "(a) vehicle passage cost"),
        (axes[1], "ped_stop_time_total_s", "total pedestrian stop time [s]",
         "(b) pedestrian waiting"),
    ):
        for rn in ("R1", "S_cons", "S_mod", "S_lib"):
            for model, ls in (("wiedemann", "-"), ("sfm", "--")):
                sub = out.metrics[(out.metrics["rule"] == rn)
                                  & (out.metrics["model"] == model)
                                  & (out.metrics["outcome"] == "PASS")].sort_values("width_m")
                sub = sub[sub[col].notna()]
                ax.plot(sub["width_m"], sub[col], color=RULE_COLORS[rn], lw=1.2,
                        ls=ls, marker=RULE_MARKERS[rn], ms=3.5, alpha=0.85,
                        label=rn if model == "wiedemann" else None)
        ax.set_xlabel("road width L [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(fontsize=6.8, title="rule (solid: Wiedemann, dashed: SFM)",
                   title_fontsize=6.5)
    axes[0].annotate("L=3.0 m:  25-30 s -> 15 s -> 8 s", (3.25, 19.0), fontsize=7,
                     color="#333333")
    fig.suptitle("Cost of passing: the speed-dependent acceptance model lowers the delay and the waiting",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "figA3_cost")


# ---------------------------------------------------------------------------
# 图 A4：现象验证（L = 3.0）
# ---------------------------------------------------------------------------
def figA4_phenomenon(out: AccOutputs, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2))

    # (a)(b)(c) L=3.0, Wiedemann
    ax = axes[0, 0]
    for rn in ("R1", "S_cons", "S_mod", "S_lib"):
        rid = f"L3.00_{rn}_wiedemann"
        car = out.trace(rid)["CAR"]
        ax.plot(car["t"], car["speed"], color=RULE_COLORS[rn], lw=1.6, label=rn)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("vehicle speed [m/s]")
    ax.set_title("(a) L = 3.0 m: vehicle speed")
    ax.legend(fontsize=6.8)

    ax = axes[0, 1]
    for rn in ("R1", "S_cons", "S_mod", "S_lib"):
        rid = f"L3.00_{rn}_wiedemann"
        p = out.trace(rid)["P1"]
        ax.plot(p["t"], p["speed"], color=RULE_COLORS[rn], lw=1.4, label=rn)
    ax.annotate("P1 speed reaches 0 (yields)\nthen walks past the slow car",
                (24.0, 0.35), fontsize=7, color="#333333")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("P1 speed [m/s]")
    ax.set_title("(b) pedestrian P1 speed: can drop to 0")
    ax.legend(fontsize=6.8)

    ax = axes[1, 0]
    for rn in ("R1", "S_cons", "S_mod", "S_lib"):
        rid = f"L3.00_{rn}_wiedemann"
        car = out.trace(rid)["CAR"]
        ax.plot(car["t"], car["x"], color=RULE_COLORS[rn], lw=1.6, label=rn)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("vehicle x [m]")
    ax.set_title("(c) L = 3.0 m: vehicle position")
    ax.legend(fontsize=6.8)

    # (d) 实测通过时刻：车速 vs 余量，叠上接受曲线
    ax = axes[1, 1]
    v = np.linspace(0, 9, 200)
    for rn in ("S_cons", "S_mod", "S_lib"):
        c0, k = rule_clearance_curve(out.cfg["rules"][rn])
        ax.plot(v, c0 + k * v, color=RULE_COLORS[rn], lw=1.3, ls="-", alpha=0.9,
                label=f"{rn} acceptance")
    passes = out.pass_events
    if not passes.empty:
        ok = passes.merge(
            out.metrics[["run_id", "outcome"]], on="run_id", how="left"
        )
        ok = ok[(ok["outcome"] == "PASS") & (ok["gap_m"] < 1.0)]
        for rn in ("S_cons", "S_mod", "S_lib"):
            sub = ok[(ok["rule"] == rn) & (ok["model"] == "wiedemann")]
            ax.scatter(sub["car_speed_mps"], sub["clearance_m"], s=26,
                       color=RULE_COLORS[rn], marker=RULE_MARKERS[rn], alpha=0.9,
                       label=f"{rn} observed pass")
    ax.set_xlabel("vehicle speed at the passing moment [m/s]")
    ax.set_ylabel("lateral clearance at the passing moment [m]")
    ax.set_title("(d) observed passing events lie on the acceptance curves")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=6.2, ncol=2)

    fig.suptitle("L = 3.0 m: pedestrians yield only when needed, then walk close past the slow vehicle",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "figA4_phenomenon")


# ---------------------------------------------------------------------------
# 图 A5：俯视快照（R1 vs S_mod，L=3.0）
# ---------------------------------------------------------------------------
def figA5_snapshots(out: AccOutputs, out_dir: Path) -> None:
    width = 3.0
    rows = [("L3.00_R1_wiedemann", "R1:\nwaiting", ""),
            ("L3.00_S_mod_wiedemann", "S_mod:\nwalking past", "")]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 4.8))
    for r, (rid, label, note) in enumerate(rows):
        traces = out.trace(rid)
        events = out.pass_events[out.pass_events["run_id"] == rid]
        for c, ped_name in enumerate(("P1", "P2", "P3")):
            t = float(events[events["ped"] == ped_name]["t_pass_s"].iloc[0])
            ax = axes[r, c]
            car = _nearest_row(traces["CAR"], t)
            ped = _nearest_row(traces[ped_name], t)
            view = 8.5
            xc = float(car["x"])
            _draw_road(ax, width, xc - view, xc + view)
            for i in range(3):
                p = _nearest_row(traces[f"P{i + 1}"], t)
                if abs(float(p["x"]) - xc) <= view:
                    _draw_ped(ax, float(p["x"]), float(p["y"]), i)
            _draw_car(ax, xc, float(car["y"]), out.cfg["car"])
            if float(ped["speed"]) > 0.1:
                ax.annotate(
                    "", xy=(float(ped["x"]) + 2.2, float(ped["y"])),
                    xytext=(float(ped["x"]) + 0.4, float(ped["y"])),
                    arrowprops=dict(arrowstyle="-|>", color="#10375c", lw=1.2),
                )
            txt = (f"t = {t:4.1f} s\nv_car = {float(car['speed']):4.1f} m/s\n"
                   f"v_{ped_name} = {float(ped['speed']):4.1f} m/s")
            ax.annotate(txt, (0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
                        fontsize=6.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9))
            ax.set_xlim(xc - view, xc + view)
            ax.set_ylim(-width / 2 - 0.5, width / 2 + 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([-1.5, 0, 1.5])
            ax.set_yticklabels(["-1.5", "0", "1.5"], fontsize=6)
            ax.grid(False)
            if r == 0:
                ax.set_title(f"passing {ped_name}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{label}     y [m]", fontsize=7.4)
    fig.suptitle("Top-down snapshots at L = 3.0 m (Wiedemann): the same geometry, two behaviours",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "figA5_snapshots")


# ---------------------------------------------------------------------------
# 一页总结
# ---------------------------------------------------------------------------
def figA_summary(out: AccOutputs, out_dir: Path) -> None:
    cfg = out.cfg
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    fig = plt.figure(figsize=(11.6, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.45, wspace=0.3)

    # (a) 接受曲线
    ax = fig.add_subplot(gs[0, 0:2])
    v = np.linspace(0, 9, 200)
    ax.axhline(0.30, color=COL["dead"], lw=1.4, ls="--", label="R1: fixed 0.30 m")
    for rn in ("S_cons", "S_mod", "S_lib"):
        c0, k = rule_clearance_curve(cfg["rules"][rn])
        ax.plot(v, c0 + k * v, color=RULE_COLORS[rn], lw=1.8,
                label=f"{rn}: {c0:.2f}+{k:.2f}·v")
    for width, txt in ((4.0, "4.0"), (3.5, "3.5"), (3.0, "3.0"), (2.90, "2.90"), (2.84, "2.84")):
        c = clearance_available(width, car_w, ped_w)
        ax.axhline(c, color="#888", lw=0.7, ls=":")
        ax.annotate(f"L={txt}", (8.95, c), fontsize=6.2, va="bottom", ha="right", color="#555")
    ax.set_xlabel("vehicle speed [m/s]")
    ax.set_ylabel("accepted clearance [m]")
    ax.set_title("(a) speed-dependent personal space")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=6.5, loc="upper left")

    # (b) 边界：预测线 vs 仿真通过/僵持
    ax = fig.add_subplot(gs[0, 2])
    names = ["R1", "S_cons", "S_mod", "S_lib"]
    for yi, rn in enumerate(names):
        b = predicted_boundary(cfg, rn)
        color = RULE_COLORS[rn]
        sub = out.metrics[(out.metrics["rule"] == rn)
                          & (out.metrics["model"] == "wiedemann")
                          & (out.metrics["width_m"] <= 3.0)]
        for _, r in sub.iterrows():
            passed = r["outcome"] == "PASS"
            ax.scatter([float(r["width_m"])], [yi], marker=RULE_MARKERS[rn], s=34,
                       facecolor=color if passed else "white", edgecolor=color,
                       linewidth=1.0, zorder=5)
            if not passed:
                ax.scatter([float(r["width_m"])], [yi], marker="x", s=18, color=color,
                           zorder=6, linewidth=1.0)
        ax.plot([b["l_crit_m"], b["l_crit_m"]], [yi - 0.35, yi + 0.35], color=color, lw=1.4)
        ax.annotate(f"{b['l_crit_m']:.2f}", (b["l_crit_m"], yi + 0.42), ha="center",
                    fontsize=6.2, color=color)
    ax.axvline(2.80, color="#333333", lw=1.0, ls=":")
    ax.annotate("2.80 m\nlimit", (2.803, -0.62), fontsize=6.0, color="#333333")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.8, len(names) - 0.45)
    ax.set_xlim(2.78, 3.02)
    ax.set_xlabel("width [m]")
    ax.set_title("(b) predicted boundary vs simulated pass/✕")
    ax.grid(axis="y", alpha=0.1)

    # (c) L=3.0 延误与行人等待
    ax = fig.add_subplot(gs[1, 0])
    labels, delay, stop = [], [], []
    for rn in names:
        r = out.row(f"L3.00_{rn}_wiedemann")
        labels.append(rn)
        delay.append(float(r["car_delay_s"]))
        stop.append(float(r["ped_stop_time_total_s"]))
    x = np.arange(len(names))
    ax.bar(x - 0.18, delay, width=0.34, color=[RULE_COLORS[n] for n in names],
           label="vehicle delay")
    ax.bar(x + 0.18, stop, width=0.34, color="#cccccc", edgecolor="#666",
           hatch="///", label="pedestrian stop time")
    for xi, d, s in zip(x, delay, stop):
        ax.annotate(f"{d:.0f}", (xi - 0.18, d + 0.5), ha="center", fontsize=6.5)
        ax.annotate(f"{s:.0f}", (xi + 0.18, s + 0.5), ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("time [s]")
    ax.set_title("(c) cost at L = 3.0 m (Wiedemann)")
    ax.legend(fontsize=6.2)
    ax.set_ylim(0, 34)

    # (d) 边界处“等行人走过再走”的快照
    ax = fig.add_subplot(gs[1, 1:])
    rid = "L2.84_S_mod_wiedemann"
    tr = out.trace(rid)
    row = out.row(rid)
    ev = out.pass_events[(out.pass_events["run_id"] == rid)
                         & (out.pass_events["ped"] == "P2")]
    t_show = float(ev["t_pass_s"].iloc[0]) if not ev.empty else float(tr["CAR"]["t"].max()) - 10.0
    car = _nearest_row(tr["CAR"], t_show)
    xc = float(car["x"])
    _draw_road(ax, 2.84, xc - 12, xc + 12)
    for i in range(3):
        p = _nearest_row(tr[f"P{i + 1}"], t_show)
        _draw_ped(ax, float(p["x"]), float(p["y"]), i)
    _draw_car(ax, xc, float(car["y"]), cfg["car"])
    ax.set_xlim(xc - 12, xc + 12)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.set_yticks([-1.42, 0, 1.42])
    ax.set_xlabel("x [m]")
    ax.set_title(f"(d) L = 2.84 m, S_mod: vehicle waits ({float(row['car_delay_s']):.0f} s delay), "
                 f"pedestrians walk past at 0.02 m")
    ax.grid(False)

    fig.suptitle(
        "Speed-dependent acceptance:  boundary 2.90 → 2.84 → 2.82 m (limit 2.80);  "
        "at L=3.0 vehicle delay 25–30 s → 15 s → 8 s, pedestrian waiting 20 s → 4 s → 2 s",
        fontsize=8.6, y=0.99,
    )
    _save(fig, out_dir, "summary2")


def make_acceptance_figures(out_dir: str | Path, summary: dict[str, Any] | None = None) -> None:
    out_dir = Path(out_dir)
    out = AccOutputs(out_dir, summary)
    figA1_model(out, out_dir)
    figA2_boundary(out, out_dir)
    figA3_cost(out, out_dir)
    figA4_phenomenon(out, out_dir)
    figA5_snapshots(out, out_dir)
    figA_summary(out, out_dir)


