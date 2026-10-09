"""Create the required one-page A3 scientific PDF and 300-DPI PNG."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

from .edie_calculation import parallelogram_polygon, rectangle_polygon
from .fundamental_diagram import model_curve


COLORS = {"Rectangular": "#0072B2", "Parallelogram": "#D55E00", "Adaptive": "#009E73"}
MARKERS = {"Rectangular": "o", "Parallelogram": "s", "Adaptive": "^"}
LINESTYLES = {"Rectangular": "-", "Parallelogram": "--", "Adaptive": "-."}


def _trajectory_lines(frame: pd.DataFrame, bounds: tuple[float, float, float, float] | None = None) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    for _, group in frame.groupby("trajectory_segment_id", sort=False):
        values = group[["time_s", "s_clean_m"]].to_numpy(float)
        if bounds is not None:
            t_min, t_max, s_min, s_max = bounds
            if values[:, 0].max() < t_min or values[:, 0].min() > t_max:
                continue
            if values[:, 1].max() < s_min or values[:, 1].min() > s_max:
                continue
        if len(values) >= 2:
            lines.append(values)
    return lines


def _window_polygon(row: pd.Series) -> np.ndarray:
    if float(row["wave_speed_m_s"]) == 0.0:
        return rectangle_polygon(row["t0_s"], row["s0_m"], row["duration_s"], row["length_m"])
    return parallelogram_polygon(
        row["t0_s"], row["s0_m"], row["duration_s"], row["length_m"], row["wave_speed_m_s"]
    )


def _representative_rows(states: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    rect = states["Rectangular"]
    valid = rect.loc[rect["valid"].astype(bool)]
    target_t = float(valid["t0_s"].median())
    target_s = float(valid["s0_m"].median())
    distance = (valid["t0_s"] - target_t).abs() + (valid["s0_m"] - target_s).abs()
    anchor_id = int(valid.loc[distance.idxmin(), "anchor_id"])
    chosen = {}
    for method, table in states.items():
        matched = table.loc[table["anchor_id"].eq(anchor_id)]
        if matched.empty:
            matched = table.loc[table["valid"].astype(bool)].head(1)
        chosen[method] = matched.iloc[0]
    return chosen


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.tick_params(direction="out", length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def _plot_time_space(ax: plt.Axes, trajectories: pd.DataFrame, rows: dict[str, pd.Series]) -> None:
    line_collection = LineCollection(
        _trajectory_lines(trajectories), colors="#4D4D4D", linewidths=0.22, alpha=0.26, rasterized=True
    )
    ax.add_collection(line_collection)
    ax.autoscale()
    ax.set(xlabel="Time (s)", ylabel="Longitudinal position (m)", title="Lane-3 trajectories and aggregation geometry")
    _style_axis(ax)
    polygons = {}
    for method, row in rows.items():
        polygon = _window_polygon(row)
        polygons[method] = polygon
        ax.add_patch(
            Polygon(
                polygon, closed=True, facecolor=COLORS[method], edgecolor=COLORS[method],
                linewidth=1.4, alpha=0.18, linestyle=LINESTYLES[method], label=method,
            )
        )
    ax.legend(loc="upper left", fontsize=6.5, frameon=True, ncol=3)

    all_vertices = np.vstack(list(polygons.values()))
    margin_t = max(5.0, float(np.ptp(all_vertices[:, 0])) * 0.20)
    margin_s = max(10.0, float(np.ptp(all_vertices[:, 1])) * 0.15)
    bounds = (
        float(all_vertices[:, 0].min() - margin_t), float(all_vertices[:, 0].max() + margin_t),
        float(all_vertices[:, 1].min() - margin_s), float(all_vertices[:, 1].max() + margin_s),
    )
    zoom = inset_axes(ax, width="45%", height="47%", loc="lower right", borderpad=1.0)
    zoom.add_collection(
        LineCollection(
            _trajectory_lines(trajectories, bounds), colors="#3A3A3A", linewidths=0.42, alpha=0.42,
            rasterized=True,
        )
    )
    for method, polygon in polygons.items():
        zoom.add_patch(
            Polygon(
                polygon, closed=True, fill=False, edgecolor=COLORS[method], linewidth=1.2,
                linestyle=LINESTYLES[method],
            )
        )
    zoom.set_xlim(bounds[0], bounds[1])
    zoom.set_ylim(bounds[2], bounds[3])
    zoom.set_title("Local geometry", fontsize=6.5, pad=2)
    zoom.tick_params(labelsize=5.5, length=2)
    zoom.grid(True, color="#E5E5E5", linewidth=0.35)
    adaptive = rows["Adaptive"]
    zoom.text(
        0.02, 0.02,
        f"L={adaptive['length_m']:.0f} m, T={adaptive['duration_s']:.0f} s\n"
        f"w={adaptive['wave_speed_m_s']:.2f} m/s; A=L×T",
        transform=zoom.transAxes, fontsize=5.5, va="bottom",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#B0B0B0", "pad": 2},
    )
    _panel_label(ax, "(a)")


def _scatter_panel(
    ax: plt.Axes,
    method: str,
    states: pd.DataFrame,
    fit: dict[str, Any],
    limits: tuple[float, float],
    label: str,
) -> None:
    valid = states.loc[states["valid"].astype(bool)]
    ax.scatter(
        valid["density_veh_km"], valid["flow_veh_h"], s=12, alpha=0.40,
        color=COLORS[method], marker=MARKERS[method], linewidths=0, rasterized=True,
    )
    curve = model_curve(fit)
    if len(curve):
        ax.plot(curve["density_veh_km"], curve["flow_veh_h"], color="#1A1A1A", linewidth=1.0)
    ax.set(xlim=(0, limits[0]), ylim=(0, limits[1]), xlabel="Density (veh/km/lane)", ylabel="Flow (veh/h/lane)")
    ax.set_title(f"{method} (n={len(valid)})", fontsize=8, pad=3)
    _style_axis(ax)
    _panel_label(ax, label)


def _comparison_panel(
    ax: plt.Axes,
    relation: str,
    fit_details: dict[str, Any],
    label: str | None = None,
) -> None:
    for method in ["Rectangular", "Parallelogram", "Adaptive"]:
        curve = model_curve(fit_details[method]["full_fit"])
        if curve.empty:
            continue
        if relation == "qk":
            x, y = curve["density_veh_km"], curve["flow_veh_h"]
            xlabel, ylabel, title = "Density (veh/km)", "Flow (veh/h)", "Flow-density fits"
        elif relation == "vk":
            x, y = curve["density_veh_km"], curve["speed_km_h"]
            xlabel, ylabel, title = "Density (veh/km)", "Speed (km/h)", "Speed-density fits"
        else:
            x, y = curve["speed_km_h"], curve["flow_veh_h"]
            xlabel, ylabel, title = "Speed (km/h)", "Flow (veh/h)", "Flow-speed fits"
        ax.plot(
            x, y, color=COLORS[method], linestyle=LINESTYLES[method], linewidth=1.45,
            label=method,
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    _style_axis(ax)
    if relation == "qk":
        ax.legend(fontsize=5.5, loc="best", frameon=True)
    if label:
        _panel_label(ax, label)


def _format_number(value: Any, digits: int = 0) -> str:
    return "N/A" if value is None or not np.isfinite(float(value)) else f"{float(value):,.{digits}f}"


def _key_findings(metrics: pd.DataFrame, wave_speed_m_s: float) -> str:
    finite = metrics.dropna(subset=["test_rmse_veh_h"])
    if finite.empty:
        comparison = "No method produced a valid held-out error estimate."
    else:
        best = finite.loc[finite["test_rmse_veh_h"].idxmin()]
        comparison = (
            f"{best['method']} had the lowest held-out flow RMSE "
            f"({_format_number(best['test_rmse_veh_h'])} veh/h) under the common time-block split."
        )
    coverage = ", ".join(
        f"{row.method}: {int(row.valid_samples)}/{int(row.candidate_windows)}"
        for row in metrics.itertuples(index=False)
    )
    return (
        f"Wave-aligned windows used an empirical w={wave_speed_m_s:.2f} m/s; the adaptive method "
        "selected size and orientation without discarding acceleration/deceleration states.\n"
        f"{comparison}\nValid windows ({coverage}). Lower scatter or RMSE alone does not prove more accurate "
        "state extraction; results are limited to one lane and one 15-min period."
    )


def generate_report(
    trajectories: pd.DataFrame,
    states: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    fit_details: dict[str, Any],
    wave_speed_m_s: float,
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "NGSIM_I80_Fundamental_Diagram_Report.pdf"
    png_path = output / "NGSIM_I80_Fundamental_Diagram_Report.png"
    manifest_path = output / "NGSIM_I80_Fundamental_Diagram_Report.manifest.json"
    report_cfg = config["report"]

    valid_all = pd.concat([table.loc[table["valid"].astype(bool)] for table in states.values()])
    k_limit = float(valid_all["density_veh_km"].max() * 1.05)
    q_limit = float(valid_all["flow_veh_h"].max() * 1.05)
    representative = _representative_rows(states)

    style = {
        "font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 8.0,
        "axes.labelsize": 6.8, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
        "legend.fontsize": 6.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    with mpl.rc_context(style):
        fig = plt.figure(
            figsize=(float(report_cfg["page_width_in"]), float(report_cfg["page_height_in"])),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            3, 6, left=0.045, right=0.985, bottom=0.065, top=0.895,
            wspace=0.72, hspace=0.62, height_ratios=[1.08, 0.92, 0.58],
        )
        ax_a = fig.add_subplot(grid[0:2, 0:3])
        ax_b = fig.add_subplot(grid[0, 3])
        ax_c = fig.add_subplot(grid[0, 4])
        ax_d = fig.add_subplot(grid[0, 5])
        ax_e1 = fig.add_subplot(grid[1, 3])
        ax_e2 = fig.add_subplot(grid[1, 4])
        ax_e3 = fig.add_subplot(grid[1, 5])
        ax_f = fig.add_subplot(grid[2, 0:3])
        ax_g = fig.add_subplot(grid[2, 3:6])

        _plot_time_space(ax_a, trajectories, representative)
        labels = {"Rectangular": "(b)", "Parallelogram": "(c)", "Adaptive": "(d)"}
        axes = {"Rectangular": ax_b, "Parallelogram": ax_c, "Adaptive": ax_d}
        for method in axes:
            _scatter_panel(
                axes[method], method, states[method], fit_details[method]["full_fit"],
                (k_limit, q_limit), labels[method],
            )
        _comparison_panel(ax_e1, "qk", fit_details, "(e)")
        _comparison_panel(ax_e2, "vk", fit_details)
        _comparison_panel(ax_e3, "qv", fit_details)

        ax_f.axis("off")
        _panel_label(ax_f, "(f)")
        ax_f.set_title("Quantitative comparison (held-out final 30% time blocks)", loc="left", fontsize=8, pad=2)
        table_rows = []
        for method in ["Rectangular", "Parallelogram", "Adaptive"]:
            row = metrics.loc[metrics["method"].eq(method)].iloc[0]
            table_rows.append(
                [method, int(row["valid_samples"]), _format_number(row["test_mae_veh_h"]),
                 _format_number(row["test_rmse_veh_h"]), _format_number(row["capacity_veh_h"]),
                 _format_number(row["capacity_stability_cv_percent"], 1) + ("%" if np.isfinite(row["capacity_stability_cv_percent"]) else "")]
            )
        table = ax_f.table(
            cellText=table_rows,
            colLabels=["Method", "Samples", "MAE\n(veh/h)", "RMSE\n(veh/h)", "Capacity\n(veh/h)", "Stability\nCV"],
            cellLoc="center", colLoc="center", loc="center", bbox=[0.0, 0.02, 1.0, 0.78],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.2)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor("#B8B8B8")
            cell.set_linewidth(0.5)
            cell.set_facecolor("#F0F0F0" if row_index == 0 else "white")
            if row_index == 0:
                cell.set_text_props(weight="bold")

        ax_g.axis("off")
        _panel_label(ax_g, "(g)")
        ax_g.set_title("Key findings", loc="left", fontsize=8, pad=2)
        findings = "\n".join(
            textwrap.fill(paragraph, width=104, break_long_words=False)
            for paragraph in _key_findings(metrics, wave_speed_m_s).splitlines()
        )
        ax_g.text(0.0, 0.82, findings, va="top", ha="left", fontsize=7.0, linespacing=1.35)

        fig.suptitle(str(report_cfg["title"]), x=0.045, y=0.965, ha="left", fontsize=15, fontweight="bold")
        fig.text(
            0.045, 0.927,
            "Objective: compare exact Edie states from rectangular, wave-aligned, and exploratory adaptive windows "
            "on a common lane/time-space domain.",
            ha="left", fontsize=8.2,
        )
        fig.text(
            0.045, 0.022,
            "Lane 3; SI units; exact segment-boundary interpolation. Parallelogram geometry is a simplified implementation "
            "inspired by He & Wu (2025), not a full reproduction. Greenshields fits are reference models, not extraction accuracy.",
            ha="left", fontsize=6.2, color="#333333",
        )
        metadata = {"Title": str(report_cfg["title"]), "Author": "Reproducible Python workflow"}
        fig.savefig(pdf_path, format="pdf", dpi=int(report_cfg["dpi"]), metadata=metadata)
        fig.savefig(
            png_path, format="png", dpi=int(report_cfg["dpi"]), facecolor="white",
            transparent=False, pil_kwargs={"compress_level": 6},
        )
        plt.close(fig)
    with Image.open(png_path) as rendered:
        rendered.convert("RGB").save(
            png_path, format="PNG", dpi=(int(report_cfg["dpi"]), int(report_cfg["dpi"])),
            compress_level=6,
        )

    manifest = {
        "page": "A3 landscape",
        "size_inches": [float(report_cfg["page_width_in"]), float(report_cfg["page_height_in"])],
        "png_dpi": int(report_cfg["dpi"]),
        "background": "opaque white",
        "palette": COLORS,
        "raw_points_shown": True,
        "common_flow_density_limits": [k_limit, q_limit],
        "simplification_note": "Wave-aligned polygon follows the task-specified s0+w(t-t0) boundaries.",
        "sources": [
            "He, Z., & Wu, C. (2025). arXiv:2507.09648v2.",
            "Kassis et al. (2026). Scientific Agent Skills. arXiv:2609.00065.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pdf_path, png_path, manifest_path


def generate_method_figures(
    states: dict[str, pd.DataFrame],
    fit_details: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Save one three-relation raw-scatter diagnostic figure per method."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    valid_all = pd.concat([table.loc[table["valid"].astype(bool)] for table in states.values()])
    limits = {
        "k": float(valid_all["density_veh_km"].max() * 1.05),
        "q": float(valid_all["flow_veh_h"].max() * 1.05),
        "v": float(valid_all["speed_km_h"].max() * 1.05),
    }
    paths: list[Path] = []
    style = {
        "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.0,
        "axes.labelsize": 9.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "figure.facecolor": "white", "axes.facecolor": "white",
    }
    for method in ["Rectangular", "Parallelogram", "Adaptive"]:
        valid = states[method].loc[states[method]["valid"].astype(bool)]
        curve = model_curve(fit_details[method]["full_fit"])
        with mpl.rc_context(style):
            fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
            scatter_options = {
                "s": 14, "alpha": 0.42, "color": COLORS[method],
                "marker": MARKERS[method], "linewidths": 0, "rasterized": True,
            }
            axes[0].scatter(valid["density_veh_km"], valid["flow_veh_h"], **scatter_options)
            axes[1].scatter(valid["density_veh_km"], valid["speed_km_h"], **scatter_options)
            axes[2].scatter(valid["speed_km_h"], valid["flow_veh_h"], **scatter_options)
            if len(curve):
                axes[0].plot(curve["density_veh_km"], curve["flow_veh_h"], color="#111111", linewidth=1.2)
                axes[1].plot(curve["density_veh_km"], curve["speed_km_h"], color="#111111", linewidth=1.2)
                axes[2].plot(curve["speed_km_h"], curve["flow_veh_h"], color="#111111", linewidth=1.2)
            axes[0].set(xlim=(0, limits["k"]), ylim=(0, limits["q"]), xlabel="Density (veh/km/lane)", ylabel="Flow (veh/h/lane)", title="Flow-density")
            axes[1].set(xlim=(0, limits["k"]), ylim=(0, limits["v"]), xlabel="Density (veh/km/lane)", ylabel="Speed (km/h)", title="Speed-density")
            axes[2].set(xlim=(0, limits["v"]), ylim=(0, limits["q"]), xlabel="Speed (km/h)", ylabel="Flow (veh/h/lane)", title="Flow-speed")
            for ax in axes:
                _style_axis(ax)
                ax.text(0.98, 0.96, f"n={len(valid)}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
            fig.suptitle(f"{method} empirical fundamental diagrams", fontsize=13, fontweight="bold")
            path = output / f"{method}_Fundamental_Diagrams.png"
            fig.savefig(path, dpi=int(config["report"]["dpi"]), facecolor="white", transparent=False)
            plt.close(fig)
        with Image.open(path) as rendered:
            rendered.convert("RGB").save(
                path, format="PNG", dpi=(int(config["report"]["dpi"]), int(config["report"]["dpi"])),
                compress_level=6,
            )
        paths.append(path)
    return paths
