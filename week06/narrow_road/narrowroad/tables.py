"""参数表生成：从 config + metrics 自动生成 Markdown 表格，保证报告与配置一致。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(v: float, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def write_tables(summary: dict[str, Any], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = _load_cfg()
    _write_geometry(cfg, out)
    _write_rules(cfg, out)
    _write_models(cfg, out)
    _write_results(summary, out)


def _load_cfg() -> dict[str, Any]:
    from .common import load_config

    return load_config()


def _write_geometry(cfg: dict[str, Any], out: Path) -> None:
    road, ped, car = cfg["road"], cfg["pedestrians"], cfg["car"]
    rows = []
    for name, p in (("行人", ped), ("车辆", car)):
        rows.append(
            [
                name,
                _fmt(p["length_m"]),
                _fmt(p["width_m"]),
                f"{_fmt(p['x_range_m'][0])}–{_fmt(p['x_range_m'][1])}"
                if "x_range_m" in p
                else f"{_fmt(p['start_x_m'])}（固定）",
                _fmt(p["v0_range_mps"][0]) + f"–{_fmt(p['v0_range_mps'][1])}"
                if "v0_range_mps" in p
                else _fmt(p["v0_mps"]),
                _fmt(p["tau_s"]),
                _fmt(p["a_max_mps2"]),
                _fmt(p["b_comfort_mps2"]),
                _fmt(p["b_max_mps2"]),
                "1（可侧移）" if p is ped else "0（锁死）",
            ]
        )
    header = ["主体", "车长/体深 L", "车宽/肩宽 W", "初始 x [m]", "期望速度 [m/s]",
              "τ [s]", "a_max", "b_comfort", "b_max", "横向机动 α"]
    text = "# 表 1  几何与运动参数（单位：m, s, m/s）\n\n" + _table(rows, header)
    text += (
        f"\n- 道路：x ∈ [{_fmt(road['x_min_m'])}, {_fmt(road['x_max_m'])}] m，"
        f"路中线 y = 0，宽度 L 取 {', '.join(f'{w:.1f}' for w in road['widths_m'])} m（扫描另含"
        f" {', '.join(f'{w:.1f}' for w in road['sweep_widths_m'])} m）。\n"
        f"- 车辆沿 −x 行驶、固定在 y = {_fmt(car['y_m'])}，不可侧移；行人沿 +x 步行、可侧移。\n"
        f"- 行人初始条件由种子 {cfg['meta']['seed']} 一次抽取后冻结，所有宽度/规则/模型共用。\n"
    )
    (out / "01_geometry_and_motion.md").write_text(text, encoding="utf-8")


def _write_rules(cfg: dict[str, Any], out: Path) -> None:
    ped = cfg["pedestrians"]
    lines = [
        "# 表 2  交互规则（决策层）",
        "",
        "行人与车辆使用同一套规则引擎，只是角色参数不同。",
        "",
        "## 2.1 行人",
        "",
        "| 条件 | 行为 | 期望速度 v_target | 横向目标 y_target |",
        "|---|---|---|---|",
        f"| 车辆尚未驶过，且横向余量 c < {ped['c_pass_m']:.2f} m，且纵向净距 g ≤ {ped['d_alert_m']:.0f} m，g > {ped['d_wait_m']:.0f} m | SIDESTEP | "
        f"{ped['sidestep_speed_factor']:.2f}·v0 | 就近路缘 ±(L/2 − Wp/2) |",
        f"| 同上但 g ≤ {ped['d_wait_m']:.0f} m | WAIT | 0 | R0/A1：就地僵住；R1/A2：继续移向路缘 |",
        "| 其余情况（车辆已驶过，或横向余量已足够） | WALK | v0 | 若已让过路则沿路缘继续 |",
        f"| 前方 {ped['follow_range_m']:.0f} m 内有同行行人且横向偏离 < {ped['follow_width_m']:.1f} m | 跟驰限速 | ≤ √(2·b·max(g−{ped['follow_gap_m']:.1f}, 0)) | — |",
        "",
        "## 2.2 车辆",
        "",
        "| 约束 | 公式 | 说明 |",
        "|---|---|---|",
        f"| 物理安全 | v ≤ √(2·b_plan·max(g − {cfg['car']['g_stop_m']:.1f}, 0)) − v_ped | 行人仍在路径内（横向重叠）时必须在 {cfg['car']['g_stop_m']:.1f} m 处停住；v_ped 为行人迎面走来的速度 |",
        f"| 规则允许速度 | R0: c ≥ {cfg['rules']['R0']['car_c_req_m']:.2f} m 放行 v0，否则 0；R1: v_clear = (c − {cfg['rules']['R1']['car_c_min_m']:.2f}) / {cfg['rules']['R1']['car_c_per_mps']:.2f} | c 为车辆与行人的横向身体余量 |",
        f"| 停车（v_clear = 0） | v ≤ √(2·b_plan·max(g − {cfg['car']['d_standoff_m']:.1f}, 0)) − v_ped | 在行人前方 {cfg['car']['d_standoff_m']:.1f} m 停住 |",
        f"| 低速通过（0 < v_clear < v0） | v ≤ √(v_clear² + 2·b_plan·max(g − {cfg['car']['d_near_m']:.1f}, 0)) | 以 v_clear 从行人侧方蹭行通过 |",
        f"| 并行通过 | v ≤ v_clear | 车头超过行人后直到车尾完全越过，保持低速 |",
        "",
        "## 2.3 规则集",
        "",
        "| 规则 | 名称 | 行人 WAIT 时 | 车辆余量判据 | 允许通过的临界宽度 |",
        "|---|---|---|---|---|",
        f"| R0 | 基准 | 完全僵住 | 固定 c_req = {cfg['rules']['R0']['car_c_req_m']:.2f} m | L ≥ 3.40 m |",
        f"| R1 | 修改 | 继续移向路缘 | 速度相关 v_clear = (c − {cfg['rules']['R1']['car_c_min_m']:.2f})/{cfg['rules']['R1']['car_c_per_mps']:.2f} | L > 2.90 m |",
        "| A1 | 消融：只改车 | 完全僵住 | 同 R1 | L > 2.90 m |",
        "| A2 | 消融：只改人 | 继续移向路缘 | 同 R0 | L ≥ 3.40 m |",
        "",
        "> 临界宽度由几何关系 L = 2·(c_req + Wc/2 + Wp) 得到：R0 为 2·(0.30+0.90+0.50)=3.40 m；"
        "R1 取几何最小余量 c_min=0.05 m，为 2·(0.05+0.90+0.50)=2.90 m。",
        "",
    ]
    (out / "02_rules.md").write_text("\n".join(lines), encoding="utf-8")


def _write_models(cfg: dict[str, Any], out: Path) -> None:
    sfm, wie = cfg["models"]["sfm"], cfg["models"]["wiedemann"]
    lines = [
        "# 表 3  两种运动模型（控制层）",
        "",
        "## 3.1 社会力模型（SFM）",
        "",
        "对行人 i（全向移动体）：",
        "",
        "```",
        "a_i = (v_target·e_i − v_i)/τ  +  Σ_j  A_ij·exp(−d_ij^clear/B_ij)·w(φ_ij)·n_ij",
        "w(φ) = λ + (1−λ)(1+cosφ)/2        # 各向异性：正前方 1，正后方 λ",
        "```",
        "",
        "对车辆（不可侧移）：只保留斥力沿行驶方向的制动分量，且斥力不允许把车往前推。"
        "车辆用三个圆近似矩形。",
        "",
        "| 参数 | 值 | 含义 |",
        "|---|---|---|",
        f"| 车辆三圆 | r={sfm['car_circles']['radius_m']:.2f} m, 偏移 {sfm['car_circles']['offsets_m']} m | 近似 4.5×1.8 m 矩形 |",
        f"| 行人↔行人 | A={sfm['ped_ped']['A_mps2']}, B={sfm['ped_ped']['B_m']}, λ={sfm['ped_ped']['aniso_lambda']} | 指数斥力（加速度量纲） |",
        f"| 行人←车辆 | A={sfm['ped_car']['A_mps2']}, B={sfm['ped_car']['B_m']}, λ={sfm['ped_car']['aniso_lambda']} | 行人避车 |",
        f"| 车辆←行人 | A={sfm['car_ped']['A_mps2']}, B={sfm['car_ped']['B_m']}, λ={sfm['car_ped']['aniso_lambda']} | 车辆制动分量 |",
        "",
        "## 3.2 Wiedemann 心理-物理阈值模型（简化实现）",
        "",
        "```",
        "s_des   = SDX + T·v                # 期望净距（SDX = s0）",
        "s_brake = s_des + u²/(2·b)         # BX 制动阈值",
        "s_react = s_des + u·t_react        # AX 接近阈值",
        "若 s ≤ s_brake          -> 制动 regime: a = clip(−u²/(2·max(s−SDX,0.3)), −b_max, −b_comfort)",
        "若 s ≤ s_react 且 u>SDV -> 接近 regime: 同上，允许较轻的减速",
        "若 s < s_des            -> 跟驰 regime: a = clip(k_gap·(s−s_des) − k_v·u, −b_comfort, a_max)",
        "否则                     -> 自由行驶: 向 v_target 收敛",
        "```",
        "",
        "其中 s 为车头-车尾净距，u 为接近速度，SDV 为速度差感知阈值；"
        "交互项只允许让车/人多减速，不越过决策层给出的期望速度。",
        "",
        "| 参数 | 车辆 | 行人 | 含义 |",
        "|---|---|---|---|",
        f"| T [s] | {wie['car']['T_s']} | {wie['ped']['T_s']} | 期望时间车头距 |",
        f"| SDX=s0 [m] | {wie['s0_m']} | {wie['s0_m']} | 静止最小净距 |",
        f"| t_react [s] | {wie['car']['t_react_s']} | {wie['ped']['t_react_s']} | 接近阈值时间 |",
        f"| SDV [m/s] | {wie['dv_th_mps']} | {wie['dv_th_mps']} | 速度差感知阈值 |",
        f"| k_gap / k_v / k_acc | {wie['k_gap_1ps']} / {wie['k_v_1ps']} / {wie['k_acc_1ps']} | 同 | 跟驰/自由行驶增益 |",
        "",
        "> 说明：原版 Wiedemann 74 用于同向跟驰；本实验把它推广到“迎面行人”情形——"
        "把行人视为行驶路径内的前车，接近速度 u 取两者沿行驶方向的相对速度。"
        "这是为完成“人车同一模型”要求所做的显式简化，见报告局限性一节。",
        "",
    ]
    (out / "03_models.md").write_text("\n".join(lines), encoding="utf-8")


def _write_results(summary: dict[str, Any], out: Path) -> None:
    rows = []
    for r in summary["key_runs"]:
        delay = r.get("car_delay_s")
        rows.append(
            [
                r["run_id"],
                f"{r['width_m']:.1f}",
                r["rule"],
                r["model"],
                r["outcome"],
                _fmt(delay, 1) if delay is not None else "—",
                _fmt(r.get("car_min_speed_mps"), 2),
                _fmt(r.get("min_pass_clearance_m"), 2),
                _fmt(r.get("min_body_distance_m"), 2),
                _fmt(r.get("car_stop_time_s"), 1),
                f"{int(r['car_stop_count'])}",
                _fmt(r.get("deadlock_t_s"), 1) if r.get("deadlock_t_s") else "—",
                _fmt(r.get("standoff_gap_m"), 2) if r.get("standoff_gap_m") else "—",
            ]
        )
    header = ["run", "L [m]", "规则", "模型", "结果", "车辆延误 [s]", "最小车速 [m/s]",
              "通过余量 [m]", "最小身体距离 [m]", "停车时长 [s]", "停车次数", "僵持时刻 [s]", "对峙净距 [m]"]
    text = "# 表 4  关键运行结果\n\n" + _table(rows, header)
    text += (
        "\n- “通过余量”= 人车并行时横向身体余量的最小值（负值表示身体重叠）。\n"
        "- “最小身体距离”= 行人外接圆到车辆矩形的最小距离；“对峙净距”= 僵持时车头到最近行人身体的净距。\n"
        "- 全部数值来自 outputs/metrics.csv（本次真实运行）。\n"
    )
    (out / "04_results.md").write_text(text, encoding="utf-8")
