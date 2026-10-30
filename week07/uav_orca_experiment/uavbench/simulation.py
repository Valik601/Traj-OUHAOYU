"""同步更新仿真器：主实验唯一状态更新者（handoff §4.4）。

每步严格按以下顺序执行：

1. 依据冻结的放行表更新本步可出发状态；
2. 建立全体快照（位置、当前速度、阶段、是否冻结；冻结者当前速度为零）；
3. 为全部活动机生成当前 waypoint 与期望速度；
4. 基于同一快照求解全部命令（后算者看不到前算者新速度）；
5. 每个命令在整段 ``[t_k, t_{k+1}]`` 保持恒定并统一推进 ``p_next = p + v*dt``；
6. 用该实际常速段做中心进入检测（真实事件时刻）；
7. 从 ``t_{k+1}`` 起切换到 ``TO_GOAL``；
8. 在 ``t_{k+1}`` 检查目标容差（B 还要求已完成中心子任务）；
9. 记录轨迹、事件与诊断。

禁止步内切换速度、瞬移、事后裁剪或把位置吸附到目标中心。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import dynamics, geometry, navigation
from .config import RELEASE_MODELS, Config, MODEL_BASE
from .initialization import Scene
from .planners import build_planner
from .planners.base import Diagnostics, PlanResult, PlanState
from .scheduling import ReleaseTable, build_release_table

__all__ = [
    "SimulationError",
    "SimulationResult",
    "build_neighbor_lists",
    "run_simulation",
]


class SimulationError(RuntimeError):
    """仿真契约被破坏（NaN、超速、非法状态等），运行必须标记为无效。"""


DIAG_BOOL_FIELDS = (
    "sample_exhausted",
    "used_fallback",
    "solver_infeasible",
    "numerical_error",
)
DIAG_INT_FIELDS = ("candidate_count_tested", "initial_overlap_pairs")
DIAG_FLOAT_FIELDS = ("constraint_violation_max",)
_FROZEN_LIST = [navigation.STAGE_WAITING, navigation.STAGE_ARRIVED]


@dataclass
class SimulationResult:
    """一次运行的完整轨迹与事件（不包含指标；指标由 metrics.py 离线重算）。"""

    t: np.ndarray  # (K+1,)
    positions: np.ndarray  # (K+1,N,3)
    velocities: np.ndarray  # (K,N,3) 实际常速段命令速度
    stages: np.ndarray  # (K+1,N) int8
    frozen: np.ndarray  # (K+1,N) bool
    active: np.ndarray  # (K,N) bool
    center_entry_times: np.ndarray  # (N,) float64, NaN=未访问
    center_entry_steps: np.ndarray  # (N,) int64, -1=未访问
    arrival_times: np.ndarray  # (N,) float64, NaN=未到达
    release_planned: np.ndarray  # (N,) float64, NaN=A 场景/无放行
    release_actual: np.ndarray  # (N,) float64, NaN=A 场景/无放行
    planning_step_seconds: np.ndarray  # (K,) float64
    diag: dict[str, np.ndarray]
    model: str
    planner_name: str
    scenario: str
    seed: int
    scene_id: str
    release_order: str | None
    backend: str
    valid: bool
    numerical_error: bool
    error_message: str | None
    n_steps_executed: int
    wall_seconds: float

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {
            "t": np.asarray(self.t, dtype=np.float64),
            "positions": np.asarray(self.positions, dtype=np.float64),
            "velocities": np.asarray(self.velocities, dtype=np.float64),
            "stages": np.asarray(self.stages, dtype=np.int8),
            "frozen": np.asarray(self.frozen, dtype=bool),
            "active": np.asarray(self.active, dtype=bool),
            "center_entry_times": np.asarray(self.center_entry_times, dtype=np.float64),
            "center_entry_steps": np.asarray(self.center_entry_steps, dtype=np.int64),
            "arrival_times": np.asarray(self.arrival_times, dtype=np.float64),
            "release_planned": np.asarray(self.release_planned, dtype=np.float64),
            "release_actual": np.asarray(self.release_actual, dtype=np.float64),
            "planning_step_seconds": np.asarray(self.planning_step_seconds, dtype=np.float64),
            "n_steps_executed": np.asarray(int(self.n_steps_executed), dtype=np.int64),
            "status_numerical_error": np.asarray(bool(self.numerical_error)),
            "status_valid": np.asarray(bool(self.valid)),
        }
        # 诊断数组统一加 diag_ 前缀，避免与轨迹键（t/step）冲突
        d.update({f"diag_{k}": np.asarray(v) for k, v in self.diag.items()})
        return d


def build_neighbor_lists(
    cfg: Config, positions: np.ndarray, active_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """构造活动机的邻机表（主实验：全部其余 N-1 机，含等待/到达的冻结体）。

    返回 ``(neighbor_ids (A,M) int64, neighbor_mask (A,M) bool)``，邻居按稳定 ID 升序。
    """
    n = int(positions.shape[0])
    a = int(active_ids.size)
    if a == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=bool)
    mode = cfg.simulation.neighbor_mode
    if mode == "all":
        m = n - 1
        ids = np.broadcast_to(np.arange(n, dtype=np.int64), (a, n)).copy()
        keep = ids != active_ids[:, None]
        neighbor_ids = ids[keep].reshape(a, m)
        return neighbor_ids, np.ones((a, m), dtype=bool)
    if mode == "range":
        r = float(cfg.simulation.neighbor_range)  # type: ignore[arg-type]
        pos_act = positions[active_ids]
        d = np.linalg.norm(positions[None, :, :] - pos_act[:, None, :], axis=2)
        d[np.arange(a), active_ids] = np.inf
        keep = d <= r
        counts = keep.sum(axis=1)
        m = int(counts.max()) if a else 0
        neighbor_ids = np.full((a, m), -1, dtype=np.int64)
        mask = np.zeros((a, m), dtype=bool)
        for i in range(a):
            idx = np.flatnonzero(keep[i])
            if idx.size:
                neighbor_ids[i, : idx.size] = idx
                mask[i, : idx.size] = True
        return neighbor_ids, mask
    raise SimulationError(f"unknown neighbor_mode: {mode}")


def _zero_diag_arrays() -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    for name in DIAG_BOOL_FIELDS:
        d[name] = np.zeros(0, dtype=bool)
    for name in DIAG_INT_FIELDS:
        d[name] = np.zeros(0, dtype=np.int32)
    for name in DIAG_FLOAT_FIELDS:
        d[name] = np.zeros(0, dtype=np.float64)
    d["step"] = np.zeros(0, dtype=np.int64)
    d["t"] = np.zeros(0, dtype=np.float64)
    d["agent_id"] = np.zeros(0, dtype=np.int64)
    return d


def _check_diag(diag: Diagnostics, a: int) -> None:
    for name in DIAG_BOOL_FIELDS:
        if np.asarray(getattr(diag, name)).shape != (a,):
            raise SimulationError(f"diagnostic {name} must have shape ({a},)")
    for name in DIAG_INT_FIELDS + DIAG_FLOAT_FIELDS:
        if np.asarray(getattr(diag, name)).shape != (a,):
            raise SimulationError(f"diagnostic {name} must have shape ({a},)")
    if diag.extra:
        for name, arr in diag.extra.items():
            if np.asarray(arr).shape != (a,):
                raise SimulationError(f"extra diagnostic {name} must have shape ({a},)")


def _pad_diag(
    buf: dict[str, np.ndarray], step: int, t: float, ids: np.ndarray, diag: Diagnostics
) -> None:
    buf["step"] = np.concatenate([buf["step"], np.full(ids.size, step, dtype=np.int64)])
    buf["t"] = np.concatenate([buf["t"], np.full(ids.size, t, dtype=np.float64)])
    buf["agent_id"] = np.concatenate([buf["agent_id"], ids.astype(np.int64)])
    for name in DIAG_BOOL_FIELDS:
        buf[name] = np.concatenate([buf[name], np.asarray(getattr(diag, name), dtype=bool)])
    for name in DIAG_INT_FIELDS:
        buf[name] = np.concatenate([buf[name], np.asarray(getattr(diag, name), dtype=np.int32)])
    for name in DIAG_FLOAT_FIELDS:
        buf[name] = np.concatenate([buf[name], np.asarray(getattr(diag, name), dtype=np.float64)])


def _pad_static_remainder(
    *,
    positions: np.ndarray,
    velocities: np.ndarray,
    stages: np.ndarray,
    frozen_hist: np.ndarray,
    active_hist: np.ndarray,
    planning_step_seconds: np.ndarray,
    start_step: int,
) -> None:
    """数值错误后把剩余步填为静态冻结段（运行直接标记无效，不冒充超时）。"""
    if start_step + 1 < positions.shape[0]:
        positions[start_step + 1 :] = positions[start_step]
        stages[start_step + 1 :] = stages[start_step]
        frozen_hist[start_step + 1 :] = frozen_hist[start_step]
    if start_step < velocities.shape[0]:
        velocities[start_step:] = 0.0
        active_hist[start_step:] = False


def run_simulation(
    cfg: Config,
    scene: Scene,
    model: str,
    scenario: str,
    *,
    release_table: ReleaseTable | None = None,
    candidate_set: Any | None = None,
    planner: Any | None = None,
) -> SimulationResult:
    """执行一次完整运行（含 120 s 观察窗，全体冻结后仍推进静态尾段）。"""
    sim = cfg.simulation
    dt = float(cfg.experiment.dt)
    k_steps = int(cfg.n_steps)
    n = int(scene.n_agents)
    if n != sim.n_agents:
        raise SimulationError(f"scene n_agents={n} != config n_agents={sim.n_agents}")
    if scenario not in ("A", "B"):
        raise SimulationError(f"unknown scenario: {scenario}")
    if scenario == "A" and model in RELEASE_MODELS:
        raise SimulationError("release models are only defined for scenario B")
    base_model = MODEL_BASE.get(model)
    if base_model is None:
        if planner is None:
            raise SimulationError(f"unknown model: {model}")
        base_model = "custom"  # 仅测试用注入规划器的场景

    starts = np.asarray(scene.starts, dtype=np.float64)
    goals = np.asarray(scene.goals, dtype=np.float64)
    if not (np.all(np.isfinite(starts)) and np.all(np.isfinite(goals))):
        raise SimulationError("scene contains non-finite coordinates")

    if planner is None:
        planner = build_planner(model, cfg, candidate_set)
    backend = str(getattr(planner, "backend", "numpy") or "numpy")

    use_release = bool(model in RELEASE_MODELS and scenario == "B")
    if use_release:
        if release_table is None:
            release_table = build_release_table(cfg, scene)
        release_planned = np.asarray(release_table.planned, dtype=np.float64)
        release_actual = np.asarray(release_table.actual, dtype=np.float64)
        release_order: str | None = release_table.order
    else:
        release_planned = np.full(n, np.nan, dtype=np.float64)
        release_actual = np.full(n, np.nan, dtype=np.float64)
        release_order = None

    t_grid = np.arange(k_steps + 1, dtype=np.float64) * dt
    positions = np.empty((k_steps + 1, n, 3), dtype=np.float64)
    velocities = np.zeros((k_steps, n, 3), dtype=np.float64)
    stages = np.empty((k_steps + 1, n), dtype=np.int8)
    frozen_hist = np.zeros((k_steps + 1, n), dtype=bool)
    active_hist = np.zeros((k_steps, n), dtype=bool)
    planning_step_seconds = np.zeros(k_steps, dtype=np.float64)

    center_entry_times = np.full(n, np.nan, dtype=np.float64)
    center_entry_steps = np.full(n, -1, dtype=np.int64)
    center_done = np.zeros(n, dtype=bool)
    arrival_times = np.full(n, np.nan, dtype=np.float64)

    positions[0] = starts
    if scenario == "A":
        stages[0] = navigation.STAGE_TO_GOAL
    elif use_release:
        stages[0] = navigation.STAGE_WAITING
    else:
        stages[0] = navigation.STAGE_TO_CENTER
    frozen_hist[0] = np.isin(stages[0], _FROZEN_LIST)

    diag_buf = _zero_diag_arrays()
    error_message: str | None = None
    numerical_error = False
    n_executed = 0
    wall_start = time.perf_counter()
    k = 0

    try:
        while k < k_steps:
            t = float(t_grid[k])
            t_next = float(t_grid[k + 1])
            step_start = time.perf_counter()
            stage_k = stages[k]
            # ---- 1. 放行 --------------------------------------------------
            if use_release:
                assert release_table is not None
                releasable = (stage_k == navigation.STAGE_WAITING) & (
                    t >= release_actual - 1e-12
                )
                if np.any(releasable):
                    stage_k = stage_k.copy()
                    stage_k[releasable] = navigation.STAGE_TO_CENTER
                    stages[k] = stage_k
            frozen_k = np.isin(stage_k, _FROZEN_LIST)
            frozen_hist[k] = frozen_k
            active_ids = np.flatnonzero(~frozen_k)
            # ---- 2. 快照 --------------------------------------------------
            pos_k = positions[k]
            vel_k = velocities[k - 1].copy() if k > 0 else np.zeros((n, 3), dtype=np.float64)
            vel_k[frozen_k] = 0.0  # 冻结者当前速度为零
            waypoints = np.zeros((0, 3), dtype=np.float64)
            v_pref = np.zeros((0, 3), dtype=np.float64)
            result: PlanResult | None = None
            if active_ids.size:
                bad_stage = ~np.isin(
                    stage_k[active_ids],
                    [navigation.STAGE_TO_CENTER, navigation.STAGE_TO_GOAL],
                )
                if np.any(bad_stage):
                    raise SimulationError("active agent in illegal stage")
                # ---- 3. waypoint / 期望速度 --------------------------------
                waypoints = navigation.waypoints_for_stages(stage_k, goals, active_ids)
                v_pref = navigation.desired_velocities(
                    waypoints, pos_k[active_ids], sim.v_max, dt
                )
                neighbor_ids, neighbor_mask = build_neighbor_lists(cfg, pos_k, active_ids)
                snapshot = np.array(pos_k, copy=True)
                velocities_snap = np.array(vel_k, copy=True)
                state = PlanState(
                    t=t,
                    dt=dt,
                    positions=snapshot,
                    velocities=velocities_snap,
                    stages=np.array(stage_k, copy=True),
                    frozen=np.array(frozen_k, copy=True),
                    active_ids=active_ids.copy(),
                    waypoints=np.array(waypoints, copy=True),
                    v_pref=np.array(v_pref, copy=True),
                    neighbor_ids=neighbor_ids,
                    neighbor_mask=neighbor_mask,
                )
                for arr in (
                    state.positions,
                    state.velocities,
                    state.stages,
                    state.frozen,
                    state.active_ids,
                    state.waypoints,
                    state.v_pref,
                    state.neighbor_ids,
                    state.neighbor_mask,
                ):
                    arr.setflags(write=False)
                # ---- 4. 同步求解 ------------------------------------------
                result = planner.plan(state)
                if not isinstance(result, PlanResult):
                    raise SimulationError("planner.plan must return PlanResult")
                _check_diag(result.diagnostics, active_ids.size)
                planned = np.array(result.velocities, dtype=np.float64, copy=True)
            else:
                planned = np.zeros((0, 3), dtype=np.float64)
            # ---- 5. 全体命令速度（冻结者为零）与推进 ---------------------
            if active_ids.size:
                v_full = dynamics.command_velocities(
                    n,
                    active_ids,
                    planned,
                    dt=dt,
                    positions=pos_k,
                    velocities=vel_k,
                    frozen=frozen_k,
                    v_max=sim.v_max,
                )
            else:
                v_full = np.zeros((n, 3), dtype=np.float64)
            p_next = dynamics.advance_positions(pos_k, v_full, dt)
            if not (np.all(np.isfinite(p_next)) and np.all(np.isfinite(v_full))):
                raise SimulationError("non-finite positions/velocities after integration")
            planning_step_seconds[k] = time.perf_counter() - step_start

            # ---- 6. 中心进入（真实事件时刻，步内线段求根） ----------------
            stage_next = stage_k.copy()
            if scenario == "B":
                center_candidates = np.flatnonzero(
                    (stage_k == navigation.STAGE_TO_CENTER) & (~center_done)
                )
                for idx in center_candidates:
                    th = navigation.center_entry_time(
                        pos_k[idx], p_next[idx] - pos_k[idx], sim.center_tolerance
                    )
                    if th is not None:
                        center_done[idx] = True
                        center_entry_times[idx] = t + float(th) * dt
                        center_entry_steps[idx] = k
                        stage_next[idx] = navigation.STAGE_TO_GOAL
                stale = np.flatnonzero(
                    (stage_next == navigation.STAGE_TO_CENTER) & center_done
                )
                stage_next[stale] = navigation.STAGE_TO_GOAL

            # ---- 7. 到达判定（t_{k+1} 网格） -----------------------------
            arrived_now = navigation.arrival_mask(
                stage_next,
                scenario,
                center_done,
                p_next,
                goals,
                sim.goal_tolerance,
            )
            if np.any(arrived_now):
                stage_next[arrived_now] = navigation.STAGE_ARRIVED
                arrival_times[arrived_now] = t_next

            # ---- 8. 记录 --------------------------------------------------
            positions[k + 1] = p_next
            velocities[k] = v_full
            stages[k + 1] = stage_next
            frozen_hist[k + 1] = np.isin(stage_next, _FROZEN_LIST)
            active_hist[k] = ~frozen_k
            if result is not None:
                _pad_diag(diag_buf, k, t, active_ids, result.diagnostics)
            n_executed = k + 1
            k += 1
    except Exception as exc:  # noqa: BLE001 - 数值/实现错误必须如实记录为无效运行
        numerical_error = True
        error_message = f"{type(exc).__name__}: {exc}"
        _pad_static_remainder(
            positions=positions,
            velocities=velocities,
            stages=stages,
            frozen_hist=frozen_hist,
            active_hist=active_hist,
            planning_step_seconds=planning_step_seconds,
            start_step=k,
        )

    wall_seconds = time.perf_counter() - wall_start
    return SimulationResult(
        t=t_grid,
        positions=positions,
        velocities=velocities,
        stages=stages,
        frozen=frozen_hist,
        active=active_hist,
        center_entry_times=center_entry_times,
        center_entry_steps=center_entry_steps,
        arrival_times=arrival_times,
        release_planned=release_planned,
        release_actual=release_actual,
        planning_step_seconds=planning_step_seconds,
        diag=diag_buf,
        model=model,
        planner_name=str(getattr(planner, "name", base_model)),
        scenario=scenario,
        seed=int(scene.seed),
        scene_id=scene.scene_id,
        release_order=release_order,
        backend=backend,
        valid=not numerical_error,
        numerical_error=numerical_error,
        error_message=error_message,
        n_steps_executed=n_executed,
        wall_seconds=wall_seconds,
    )
