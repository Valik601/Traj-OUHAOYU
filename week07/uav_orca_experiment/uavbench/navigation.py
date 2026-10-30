"""导航：状态机常量、waypoint 与期望速度（handoff §4.2、§4.5）。

本模块为纯函数集合；状态迁移由 ``simulation.py`` 统一执行。
"""

from __future__ import annotations

import numpy as np

from . import geometry

STAGE_WAITING: int = 0
STAGE_TO_CENTER: int = 1
STAGE_TO_GOAL: int = 2
STAGE_ARRIVED: int = 3

STAGE_NAMES: dict[int, str] = {
    STAGE_WAITING: "waiting",
    STAGE_TO_CENTER: "to_center",
    STAGE_TO_GOAL: "to_goal",
    STAGE_ARRIVED: "arrived",
}

#: 冻结阶段（WAITING 与 ARRIVED：命令速度恒为零，但保留在邻居与安全检测中）
FROZEN_STAGES: frozenset[int] = frozenset({STAGE_WAITING, STAGE_ARRIVED})


def desired_velocity(
    waypoint: np.ndarray, position: np.ndarray, v_max: float, dt: float
) -> np.ndarray:
    """共享期望速度，handoff §4.5。

    ``d = ||q - p||``；``d = 0 → 0``；否则
    ``v_pref = min(v_max, d/dt) * (q - p) / d``。
    """
    wp = np.asarray(waypoint, dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)
    delta = wp - pos
    d = float(np.sqrt(geometry.sq_norm(delta)))
    if d <= 0.0:
        return np.zeros(3, dtype=np.float64)
    speed = min(float(v_max), d / float(dt))
    return speed * delta / d


def desired_velocities(
    waypoints: np.ndarray, positions: np.ndarray, v_max: float, dt: float
) -> np.ndarray:
    """批量版本：``waypoints``/``positions`` 形状 ``(A,3)``。"""
    wps = np.asarray(waypoints, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    if wps.size == 0:
        return np.zeros_like(wps)
    delta = wps - pos
    d = np.sqrt(np.maximum(geometry.sq_norm(delta), 0.0))
    speed = np.minimum(float(v_max), np.divide(d, float(dt), out=np.zeros_like(d), where=d > 0.0))
    out = np.zeros_like(delta)
    nz = d > 0.0
    out[nz] = speed[nz, None] * delta[nz] / d[nz, None]
    return out


def waypoint_for_stage(stage: int, goal: np.ndarray) -> np.ndarray | None:
    """当前阶段的 waypoint：TO_CENTER → 原点；TO_GOAL → 目标；等待/到达 → None。"""
    if stage == STAGE_TO_CENTER:
        return np.zeros(3, dtype=np.float64)
    if stage == STAGE_TO_GOAL:
        return np.asarray(goal, dtype=np.float64)
    return None


def waypoints_for_stages(
    stages: np.ndarray, goals: np.ndarray, active_ids: np.ndarray
) -> np.ndarray:
    """批量得到活动机 waypoint，返回 ``(A,3)``。活动机阶段只能是 TO_CENTER/TO_GOAL。"""
    st = np.asarray(stages, dtype=np.int8)[active_ids]
    goals_act = np.asarray(goals, dtype=np.float64)[active_ids]
    out = np.zeros_like(goals_act)
    center_mask = st == STAGE_TO_CENTER
    goal_mask = st == STAGE_TO_GOAL
    if np.any(center_mask):
        out[center_mask] = 0.0
    if np.any(goal_mask):
        out[goal_mask] = goals_act[goal_mask]
    if np.any(~(center_mask | goal_mask)):
        raise ValueError("active agents must be in TO_CENTER or TO_GOAL stage")
    return out


def stage_durations(
    t_grid: np.ndarray,
    stages: np.ndarray,
    *,
    final_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每机在三个可见阶段的持续时长（等待/去中心/去目标）。

    ``t_grid`` ``(K+1,)``，``stages`` ``(K+1,N)``。返回 ``(wait, to_center, to_goal)`` 各 ``(N,)``。
    时长由阶段序列在网格上的驻留步数乘以相邻网格间隔累加；末步后的时间不计。
    """
    st = np.asarray(stages)
    n_agents = st.shape[1]
    wait = np.zeros(n_agents, dtype=np.float64)
    to_center = np.zeros(n_agents, dtype=np.float64)
    to_goal = np.zeros(n_agents, dtype=np.float64)
    if st.shape[0] < 2:
        return wait, to_center, to_goal
    dt = float(t_grid[1] - t_grid[0])
    # 每一段 [t_k, t_{k+1}] 归因于 t_k 时刻的阶段（常数速度段）
    prev = st[:-1]
    wait += np.sum(prev == STAGE_WAITING, axis=0) * dt
    to_center += np.sum(prev == STAGE_TO_CENTER, axis=0) * dt
    to_goal += np.sum(prev == STAGE_TO_GOAL, axis=0) * dt
    return wait, to_center, to_goal


def center_entry_time(
    position_t: np.ndarray, displacement: np.ndarray, center_tolerance: float
) -> float | None:
    """单机在一步常速位移内首次进入中心容差球的真实时刻（相对本步起点的秒数）。"""
    return geometry.segment_sphere_enter_time(
        position_t, displacement, center_tolerance, closed=True
    )


def arrival_mask(
    stages_next: np.ndarray,
    scenario: str,
    center_done: np.ndarray,
    positions_next: np.ndarray,
    goals: np.ndarray,
    goal_tolerance: float,
) -> np.ndarray:
    """``t_{k+1}`` 网格上的到达判定（handoff §4.4 第 8 步）。

    - 阶段必须已经是 ``TO_GOAL``（本步不得从 TO_CENTER 直接到达）；
    - B 场景还要求中心子任务已经完成（``center_done``）；
    - 到达是 "步末在容差球内"，不是"步内扫过目标球"。
    """
    st = np.asarray(stages_next)
    eligible = st == STAGE_TO_GOAL
    if scenario == "B":
        eligible = eligible & np.asarray(center_done, dtype=bool)
    pos = np.asarray(positions_next, dtype=np.float64)
    g = np.asarray(goals, dtype=np.float64)
    d = np.sqrt(geometry.sq_norm(pos - g))
    return eligible & (d <= float(goal_tolerance))
