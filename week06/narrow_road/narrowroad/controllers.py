"""控制层：两种可插拔的纵向运动模型（行人与车辆共用）。

    SFM       : 社会力模型（Helbing & Molnár 1995 形式，加速度量纲）
                驱动力 (v_target - v)/tau  +  指数型交互斥力（各向异性）
     Wiedemann: 心理-物理阈值模型（Wiedemann 1974 的简化实现）
                自由行驶 / 接近 / 制动 / 跟驰 / 静止保持 五个 regime，
                以感知阈值 AX、BX、SDX、SDV 等组织（见下方注释与报告参数表）

两者都只输出加速度；积分、道路边界与不重叠约束由 simulate.py 统一处理。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .agent import Agent, World
from .decision import Decision


# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------
def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _aniso(e: np.ndarray, to_other: np.ndarray, lam: float) -> float:
    """各向异性权重：正前方为 1，正后方为 lam。"""
    cos_phi = _clip(float(np.dot(e, to_other)), -1.0, 1.0)
    return lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi)


def car_circles(car: Agent, cfg: dict[str, Any]) -> list[tuple[float, float, float]]:
    """用三个圆近似车辆矩形（供社会力计算）。"""
    c = cfg["models"]["sfm"]["car_circles"]
    r = float(c["radius_m"])
    return [(car.x + float(off), car.y, r) for off in c["offsets_m"]]


# ---------------------------------------------------------------------------
# 社会力模型
# ---------------------------------------------------------------------------
def sfm_force(agent: Agent, world: World, cfg: dict[str, Any]) -> np.ndarray:
    """作用于 agent 的社会力合力（加速度量纲，世界坐标）。"""
    sfm = cfg["models"]["sfm"]
    e = np.array([agent.heading, 0.0])
    force = np.zeros(2)

    if agent.role == "ped":
        # --- 车辆对行人的斥力（车辆用三圆近似，取最近圆）------------------
        best: tuple[float, float, float, float] | None = None
        for cx, cy, r in car_circles(world.car, cfg):
            d = math.hypot(agent.x - cx, agent.y - cy)
            clear = d - (r + agent.radius)
            if best is None or clear < best[0]:
                best = (clear, cx, cy, d)
        assert best is not None
        clear, cx, cy, d = best
        d = max(d, 1e-6)
        to_other = np.array([(cx - agent.x) / d, (cy - agent.y) / d])
        p = sfm["ped_car"]
        w = _aniso(e, to_other, float(p["aniso_lambda"]))
        mag = float(p["A_mps2"]) * math.exp(-max(clear, -0.3) / float(p["B_m"]))
        # 实验 2：车辆对行人的感知风险随车速缩放（慢车斥力弱，行人可贴近穿行）
        if world.rule is not None and world.rule.get("risk_scaling"):
            rs = sfm.get("risk_scaling", {})
            v_car = abs(world.car.vx)
            rho = (v_car / max(world.car.v0, 1e-9)) ** float(rs.get("exponent", 0.7))
            rho = min(1.0, max(float(rs.get("floor", 0.15)), rho))
            mag *= rho
        force += mag * w * (-to_other)

        # --- 行人之间的斥力 ------------------------------------------------
        p2 = sfm["ped_ped"]
        for other in world.peds:
            if other is agent:
                continue
            d = math.hypot(agent.x - other.x, agent.y - other.y)
            if d < 1e-9:
                continue
            clear = d - (agent.radius + other.radius)
            if clear > 3.0:
                continue
            to_other = np.array([(other.x - agent.x) / d, (other.y - agent.y) / d])
            w = _aniso(e, to_other, float(p2["aniso_lambda"]))
            mag = float(p2["A_mps2"]) * math.exp(-max(clear, -0.3) / float(p2["B_m"]))
            force += mag * w * (-to_other)
        return force

    # --- 车辆：行人对其的斥力，只保留纵向制动分量（不可侧移）-------------
    p = sfm["car_ped"]
    for ped in world.peds:
        best_c: tuple[float, float, float, float] | None = None
        for cx, cy, r in car_circles(world.car, cfg):
            d = math.hypot(ped.x - cx, ped.y - cy)
            clear = d - (r + ped.radius)
            if best_c is None or clear < best_c[0]:
                best_c = (clear, cx, cy, d)
        assert best_c is not None
        clear, cx, cy, d = best_c
        d = max(d, 1e-6)
        to_other = np.array([(ped.x - cx) / d, (ped.y - cy) / d])
        w = _aniso(e, to_other, float(p["aniso_lambda"]))
        mag = float(p["A_mps2"]) * math.exp(-max(clear, -0.3) / float(p["B_m"]))
        force += mag * w * (-to_other)

    a_long_from_force = float(np.dot(force, e))
    if a_long_from_force > 0.0:          # 斥力不允许把车往前推
        a_long_from_force = 0.0
    return a_long_from_force * e


def control_sfm(agent: Agent, world: World, decision: Decision, cfg: dict[str, Any]):
    """社会力控制层，返回 (a_world_x, a_world_y, a_long)。"""
    ped_cfg = cfg["pedestrians"]
    if agent.role == "ped":
        a_long_drive = _clip(
            (decision.v_target - agent.vx) / agent.tau, -agent.b_comfort, agent.a_max
        )
        v_lat_des = _clip(
            float(ped_cfg["k_lat_1ps"]) * (decision.y_target - agent.y),
            -float(ped_cfg["v_lat_max_mps"]),
            float(ped_cfg["v_lat_max_mps"]),
        )
        a_lat_drive = _clip(
            (v_lat_des - agent.vy) / float(ped_cfg["tau_lat_s"]),
            -float(ped_cfg["a_lat_max_mps2"]),
            float(ped_cfg["a_lat_max_mps2"]),
        )
        f = sfm_force(agent, world, cfg)
        ax = _clip(a_long_drive + f[0], -agent.b_max, agent.a_max)
        ay = _clip(a_lat_drive + f[1], -float(ped_cfg["a_lat_max_mps2"]),
                   float(ped_cfg["a_lat_max_mps2"]))
        return ax, ay, ax

    # 车辆
    a_long_drive = _clip(
        (decision.v_target - agent.long_speed) / agent.tau, -agent.b_comfort, agent.a_max
    )
    f = sfm_force(agent, world, cfg)
    ax_world = agent.heading * a_long_drive + f[0]
    a_long = _clip(agent.heading * ax_world, -agent.b_max, agent.a_max)
    return agent.heading * a_long, 0.0, a_long


# ---------------------------------------------------------------------------
# Wiedemann 心理-物理阈值模型
# ---------------------------------------------------------------------------
def _free_accel(
    speed: float, v_target: float, a_max: float, b_comfort: float, dv_th: float, k: float
) -> float:
    """自由行驶 regime：向期望速度平滑收敛。

    Wiedemann 的自由行驶段是“以最大加速度加速到期望速度、超过后减速”，
    这里用带死区的比例项实现，避免离散时间步下的 bang-bang 抖动。
    注意：当期望速度本身接近 0（要求停车）时，死区不能阻止制动，否则车辆会
    以一个极小速度永久蠕行——这会掩盖本来应当发生的停车/僵持。
    """
    if v_target <= 0.5 * dv_th:                      # 期望停车
        return 0.0 if speed < 0.02 else -b_comfort
    if speed < v_target - dv_th:
        return min(a_max, max(0.2, k * (v_target - speed)))
    if speed > v_target + dv_th:
        return max(-b_comfort, k * (v_target - speed))
    return 0.0


def wiedemann_longitudinal(
    agent: Agent, world: World, decision: Decision, cfg: dict[str, Any]
) -> float:
    """Wiedemann 纵向加速度（沿自身行驶方向）。

    记号与阈值（与 config.yaml 对应）：
        s      : 到前车的车头-车尾净距
        u      : 接近速度（= d(gap)/dt 的相反数）
        SDX=s0 : 静止最小净距
        SDV    : 速度差感知阈值 dv_th
        s_des  = SDX + T*v          (T = 期望时间车头距)
        s_brake= s_des + u^2/(2b)   (BX 制动阈值)
        s_react= s_des + u*t_react  (AX 接近阈值)
    规则：
        s <= s_brake           -> 制动 regime: 按停车所需减速度减速（[-b_max, -b_comfort]）
        s <= s_react 且 u>SDV  -> 接近 regime: 同上但允许较轻的减速度
        s < s_des              -> 跟驰 regime: 比例项维持期望净距
        else                   -> 自由行驶
    交互项只能让车“多减速”，不能越过决策层给出的期望速度。
    """
    w = cfg["models"]["wiedemann"]
    role_over = w.get(agent.role, {})
    T = float(role_over.get("T_s", w["T_s"]))
    t_react = float(role_over.get("t_react_s", w["t_react_s"]))
    s0 = float(w["s0_m"])
    dv_th = float(w["dv_th_mps"])

    speed = agent.long_speed
    a_free = _free_accel(
        speed, decision.v_target, agent.a_max, agent.b_comfort, dv_th, float(w["k_acc_1ps"])
    )

    leader = world.nearest_leader(agent)
    if leader is None:
        return a_free

    gap = agent.gap_to(leader)
    u = agent.closing_speed_to(leader)
    s_des = s0 + T * speed
    s_brake = s_des + u * u / (2.0 * agent.b_comfort)
    s_react = s_des + u * t_react
    a_req = -u * u / (2.0 * max(gap - s0, 0.3))

    if gap <= s_brake:
        a_int = _clip(a_req, -agent.b_max, -agent.b_comfort)
    elif gap <= s_react and u > dv_th:
        a_int = _clip(a_req, -agent.b_max, -0.5)
    elif gap < s_des:
        a_int = _clip(
            float(w["k_gap_1ps"]) * (gap - s_des) - float(w["k_v_1ps"]) * u,
            -agent.b_comfort,
            agent.a_max,
        )
    else:
        a_int = None

    a_long = a_free if a_int is None else min(a_int, a_free)

    # 静止保持：已经停住且与前车距离小于静止净距时保持不动
    if speed < 0.05 and gap < s0 + 0.3:
        a_long = 0.0
    return a_long


def control_wiedemann(agent: Agent, world: World, decision: Decision, cfg: dict[str, Any]):
    """Wiedemann 控制层，返回 (a_world_x, a_world_y, a_long)。"""
    ped_cfg = cfg["pedestrians"]
    a_long = wiedemann_longitudinal(agent, world, decision, cfg)
    if agent.role == "ped":
        v_lat_des = _clip(
            float(ped_cfg["k_lat_1ps"]) * (decision.y_target - agent.y),
            -float(ped_cfg["v_lat_max_mps"]),
            float(ped_cfg["v_lat_max_mps"]),
        )
        ay = _clip(
            (v_lat_des - agent.vy) / float(ped_cfg["tau_lat_s"]),
            -float(ped_cfg["a_lat_max_mps2"]),
            float(ped_cfg["a_lat_max_mps2"]),
        )
        return a_long, ay, a_long
    return agent.heading * a_long, 0.0, a_long


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def control(
    agent: Agent,
    world: World,
    decision: Decision,
    model: str,
    cfg: dict[str, Any],
) -> tuple[float, float, float]:
    """按模型名调用对应控制层。"""
    if model == "sfm":
        return control_sfm(agent, world, decision, cfg)
    if model == "wiedemann":
        return control_wiedemann(agent, world, decision, cfg)
    raise ValueError(f"unknown model: {model!r}")
