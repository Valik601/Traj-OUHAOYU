"""离线连续时间指标模块（CONTRACTS.md §11 / handoff §6）。

本模块 **只从保存的实际轨迹离线重算指标**，不使用算法自报的“安全”标记。
入口为 :func:`compute_metrics`（纯计算）与 :func:`write_metrics`（落盘 6 个产物）。

轨迹输入
--------
``traj`` 为 ``trajectories.npz`` 读入后的 dict（或等价 mapping），必需键：

===================  ==========================  ==========================================
键                    形状                        含义
===================  ==========================  ==========================================
``t``                 ``(K+1,)`` float64          均匀网格时刻，``t[k] = k*dt``
``positions``         ``(K+1,N,3)`` float64       网格真实位置
``velocities``        ``(K,N,3)`` float64         每个常速段的实际命令速度
``stages``            ``(K+1,N)`` int8            导航阶段 (0 waiting,1 to_center,2 to_goal,3 arrived)
===================  ==========================  ==========================================

可选键（缺失时安全降级）：

* ``frozen`` ``(K+1,N)`` bool；``active`` ``(K,N)`` bool；``dt`` float；``n_agents`` int；
  ``observation_time`` float。
* ``arrival_times`` ``(N,)`` float64（NaN=从未到达）。若提供则作为 ``agents.csv`` 与聚合指标的
  权威到达时刻，并与 ``stages`` 推导值交叉校验（阈值 1e-9 s，差异写入
  ``metrics.consistency``）。
* ``center_entry_times`` ``(N,)`` float64、``center_entry_steps`` ``(N,)`` int64（-1=从未）。
  也可作为 ``compute_metrics`` 的显式参数传入（参数优先）。中心进入时刻只由仿真提供
  （步内事件），本模块用实际轨迹校验：``||p(t_entry)|| <= center_tolerance + tol``，
  并与轨迹重算结果比较；不一致写入 ``metrics.consistency``。
* ``release_planned`` / ``release_actual`` ``(N,)`` float64（场景 A 为 NaN）。
  也可作为显式参数传入（参数优先）。
* 规划诊断（Lead 冻结的每调用一行的布局）::

      diag_step (C,) int64, diag_t (C,) float64, diag_agent_id (C,) int64,
      diag_sample_exhausted (C,) bool, diag_used_fallback (C,) bool,
      diag_solver_infeasible (C,) bool, diag_constraint_violation_max (C,) float64,
      diag_candidate_count_tested (C,) int32, diag_initial_overlap_pairs (C,) int32,
      diag_numerical_error (C,) bool

  同时兼容 ``(K,N)`` 掩码布局（配合 ``diag_active_mask``/``diag_active_ids``）与
  ``(K,)`` 逐步聚合布局；全部缺失时按 0 处理（``active_planning_calls`` 退化为
  ``active.sum()``）。
* ``planning_step_seconds`` ``(K,)`` float64（逐步“邻居准备+期望速度+求解+放行”墙钟秒；
  兼容旧名 ``planning_time_s``）。
* ``status_numerical_error`` / ``status_valid`` 布尔标量，``n_steps_executed`` int64。
* ``experiment_id`` str（也可作为 :func:`compute_metrics` 关键字参数传入，参数优先）。

连续安全定义（handoff §6.1，统一 ``d_eval = d_safe - eps_distance``）
---------------------------------------------------------------
遍历全部 ``N(N-1)/2`` 机对；每步 ``r = p_i - p_j``、``w = v_i - v_j``，
``t* = clip(-(r·w)/|w|^2, 0, dt)``，``d_min_step = ||r + t* w||``（``w≈0`` 用静态距离）。
严格侵犯区间为 ``{t: ||r + t w||^2 < d_eval^2}`` 与本步 ``[0,dt]`` 的交（相切不算）。
相邻步延续的区间合并为一个事件；首侵犯时间取区间下确界（进入边界时刻）。
边界接触：全程最小距离落入 ``[d_safe-eps_distance, d_safe+GEOM_EPS]`` 且从未严格侵犯的机对。

若干字段口径（规范未逐字定义处，在此固定并保持稳定）
----------------------------------------------------
* ``min_distance_curve.csv`` 每行对应一个常速段区间 ``[t_k, t_{k+1}]``：``min_distance`` 为该
  区间的连续最小机间距；``violating_pair_count`` 为在该区间内存在严格侵犯的唯一机对数；
  ``boundary_pair_count`` 为该区间连续最小距离落入边界带的机对数。``t`` 取区间起点 ``t_k``。
* ``boundary_contacts.csv`` 的 ``min_time_inside_band`` 定义为该机对中心距处于边界带
  ``[d_safe-eps, d_safe+eps]`` 内的连续总时长（秒）。
* ``t90``：完成数达到 ``ceil(0.9*N)`` 的最早网格时刻（N=100 时即 90 架）；不足则为 null。
* ``mean_path_efficiency_excess`` 只对完成者计算：``mean(L_i/(2R-eps_goal) - 1)``，
  下界 ``2R - goal_tolerance``（默认 39.7 m）；``mean_path_over_40_minus_1`` 用
  ``average_path_length/(2R) - 1``（默认分母 40 m），包含未完成者，仅作描述量。
* ``wait_duration``：场景 B 为 ``release_actual``，场景 A 为 0。
  ``post_release_travel_time`` = ``arrival - release``（完成者）；场景 A 的虚拟放行时刻为 0，
  故等于到达时刻。
* ``safe_completed``：在 ``[0,T]`` 内到达且 **整个观察窗内从未参与任何侵犯**（含到达后/等待中）。
  ``safe_completion_rate`` 与 B 场景主要终点一致。
* 停滞（handoff §6.3）：已释放且未完成、同一阶段持续 ``>= stall_window``、最后
  ``stall_window`` 秒累计路程 ``< stall_path_threshold`` 且 waypoint 距离下降
  ``< stall_waypoint_threshold``；排除 WAITING/ARRIVED 阶段，阶段切换重置窗口。
  ``low_progress_mobile`` 为同样窗口内“有移动但 waypoint 无进展”的徘徊。停滞不是死锁证明。
* 固定观察窗：所有安全指标覆盖轨迹给出的整个 ``[0, t_K]``（主实验要求 ``t_K = T = 120 s``）；
  若轨迹被提前截断，``consistency.grid_complete=false`` 并如实按现有网格计算。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import geometry
from .config import Config
from .navigation import (
    STAGE_ARRIVED,
    STAGE_TO_CENTER,
    STAGE_TO_GOAL,
    STAGE_WAITING,
    stage_durations,
)

__all__ = [
    "MetricsResult",
    "compute_metrics",
    "write_metrics",
    "AGENT_COLUMNS",
    "VIOLATION_EVENT_COLUMNS",
    "BOUNDARY_CONTACT_COLUMNS",
    "MIN_DISTANCE_CURVE_COLUMNS",
    "ARRIVAL_CURVE_COLUMNS",
]

#: 相对速度被视为“零”的速度平方阈值（m²/s²），与 geometry.VEL_EPS 一致。
_REL_VEL_EPS_SQ: float = geometry.VEL_EPS ** 2
#: 时间比较容差（秒）。
_TIME_TOL: float = 1e-9
#: 中心进入时刻一致性比较容差（秒）。
_CENTER_TIME_TOL: float = 1e-6
#: 到达时刻（stages 推导 vs 仿真提供）一致性比较容差（秒）。
_ARRIVAL_TIME_TOL: float = 1e-9
#: 中心进入物理校验（||p(t_entry)|| 超出容差）附加容差（米）。
_CENTER_POS_TOL: float = 1e-6
#: 机对分块大小：每块最多 P_CHUNK 个机对 × K 步，控制峰值内存（~50 MB 量级）。
#: 100 机 × 1200 步时峰值约 40 MB，单线程重算 < 2 s。
_PAIR_CHUNK: int = 96

#: ``agents.csv`` 表头（CONTRACTS §11.3，冻结）。
AGENT_COLUMNS: tuple[str, ...] = (
    "agent_id",
    "release_rank",
    "release_planned",
    "release_actual",
    "wait_duration",
    "center_entry_time",
    "center_crossing",
    "arrival_time",
    "post_release_travel_time",
    "phase_wait_duration",
    "phase_to_center_duration",
    "phase_to_goal_duration",
    "path_length",
    "final_goal_distance",
    "final_stage",
    "violated",
    "safe_completed",
    "stalled",
    "low_progress_mobile",
)

#: ``violation_events.csv`` 表头（CONTRACTS §11.3，冻结）。每行一个合并后机对侵犯区间。
VIOLATION_EVENT_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "agent_i",
    "agent_j",
    "start_time",
    "end_time",
    "duration",
    "min_distance",
    "min_distance_time",
)

#: ``boundary_contacts.csv`` 表头（CONTRACTS §11.3，冻结）。
BOUNDARY_CONTACT_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "agent_i",
    "agent_j",
    "min_distance",
    "min_distance_time",
    "min_time_inside_band",
)

#: ``min_distance_curve.csv`` 表头（CONTRACTS §11.3，冻结）。
MIN_DISTANCE_CURVE_COLUMNS: tuple[str, ...] = (
    "step",
    "t",
    "min_distance",
    "violating_pair_count",
    "boundary_pair_count",
)

#: ``arrival_curve.csv`` 表头（CONTRACTS §11.3，冻结）。
ARRIVAL_CURVE_COLUMNS: tuple[str, ...] = (
    "step",
    "t",
    "cumulative_arrivals",
    "cumulative_center_visits",
    "cumulative_safe_completions_known_later",
)

# =============================================================================
# 结果容器
# =============================================================================
@dataclass
class MetricsResult:
    """``compute_metrics`` 的返回值（CONTRACTS §11）。"""

    metrics: dict[str, Any]
    agents: pd.DataFrame
    violation_events: pd.DataFrame
    boundary_contacts: pd.DataFrame
    min_distance_curve: pd.DataFrame
    arrival_curve: pd.DataFrame


# =============================================================================
# 通用小工具
# =============================================================================
def _opt(source: Any, *names: str) -> Any:
    """从 dict / npz-like 中取第一个存在的键；都不存在返回 ``None``。"""
    for name in names:
        try:
            return source[name]
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return None


def _required(source: Any, name: str) -> Any:
    value = _opt(source, name)
    if value is None:
        raise ValueError(f"trajectory is missing required key {name!r}")
    return value


def _as_bool(value: Any) -> bool:
    """把标量或 size-1 数组安全转成 Python bool。"""
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def _finite_float(value: Any) -> float | None:
    """float 值，非有限（NaN/Inf）→ ``None``（JSON null）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _json_safe(value: Any) -> Any:
    """递归转换为 JSON 可序列化对象；NaN/Infinity → ``None``，numpy 标量 → Python 标量。"""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.str_, str)):
        return str(value)
    return value


def _empty_frame(columns: tuple[str, ...], dtypes: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({col: np.array([], dtype=dtypes[col]) for col in columns})


# =============================================================================
# 轨迹规范化
# =============================================================================
@dataclass
class _Trajectory:
    t: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    stages: np.ndarray
    frozen: np.ndarray
    active: np.ndarray
    dt: float
    observation_time: float
    n_agents: int
    n_steps: int
    finite_data: bool
    raw: Any


def _normalize_traj(traj: Mapping[str, Any], cfg: Config) -> _Trajectory:
    t = np.asarray(_required(traj, "t"), dtype=np.float64).reshape(-1)
    positions = np.asarray(_required(traj, "positions"), dtype=np.float64)
    velocities = np.asarray(_required(traj, "velocities"), dtype=np.float64)
    stages_raw = np.asarray(_required(traj, "stages"))

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"positions must have shape (K+1,N,3), got {positions.shape}")
    n_steps = int(positions.shape[0] - 1)
    n_agents = int(positions.shape[1])
    if n_steps < 0:
        raise ValueError("positions must contain at least one grid point")
    if t.shape != (n_steps + 1,):
        raise ValueError(f"t shape {t.shape} != ({n_steps + 1},)")
    if velocities.shape != (n_steps, n_agents, 3):
        raise ValueError(
            f"velocities shape {velocities.shape} != ({n_steps},{n_agents},3)"
        )
    if stages_raw.shape != (n_steps + 1, n_agents):
        raise ValueError(
            f"stages shape {stages_raw.shape} != ({n_steps + 1},{n_agents})"
        )
    stages = stages_raw.astype(np.int8, copy=False)

    dt_raw = _opt(traj, "dt")
    if dt_raw is None:
        dt = float(t[1] - t[0]) if n_steps > 0 else float(cfg.dt)
    else:
        dt = float(dt_raw)
    if not (dt > 0.0) or not math.isfinite(dt):
        raise ValueError(f"dt must be positive and finite, got {dt!r}")
    if n_steps >= 1:
        diffs = np.diff(t)
        if not np.all(np.isfinite(diffs)) or np.max(np.abs(diffs - dt)) > 1e-9:
            raise ValueError("trajectory time grid must be uniform with spacing dt")

    frozen_raw = _opt(traj, "frozen")
    if frozen_raw is None:
        frozen = np.isin(stages, (STAGE_WAITING, STAGE_ARRIVED))
    else:
        frozen = np.asarray(frozen_raw)
        if frozen.shape != (n_steps + 1, n_agents):
            raise ValueError(f"frozen shape {frozen.shape} != ({n_steps + 1},{n_agents})")
        frozen = frozen.astype(bool, copy=False)

    active_raw = _opt(traj, "active")
    if active_raw is None:
        # 保守推导：非冻结机的下一步才会被规划；仅用于诊断分母等可选统计。
        active = ~frozen[:-1]
    else:
        active = np.asarray(active_raw)
        if active.shape != (n_steps, n_agents):
            raise ValueError(f"active shape {active.shape} != ({n_steps},{n_agents})")
        active = active.astype(bool, copy=False)

    obs_raw = _opt(traj, "observation_time")
    observation_time = float(obs_raw) if obs_raw is not None else float(t[-1] if t.size else cfg.observation_time)

    n_agents_raw = _opt(traj, "n_agents")
    if n_agents_raw is not None and int(n_agents_raw) != n_agents:
        raise ValueError(f"n_agents={int(n_agents_raw)} != positions N={n_agents}")

    finite_data = bool(
        np.all(np.isfinite(t)) and np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities))
    )
    return _Trajectory(
        t=t,
        positions=positions,
        velocities=velocities,
        stages=stages,
        frozen=frozen,
        active=active,
        dt=dt,
        observation_time=observation_time,
        n_agents=n_agents,
        n_steps=n_steps,
        finite_data=finite_data,
        raw=traj,
    )


# =============================================================================
# 连续安全扫描
# =============================================================================
@dataclass
class _PairScan:
    pair_count: int
    event_pair: np.ndarray
    event_start: np.ndarray
    event_end: np.ndarray
    event_duration: np.ndarray
    event_min_distance: np.ndarray
    event_min_time: np.ndarray
    pair_min_distance: np.ndarray
    pair_min_time: np.ndarray
    violated_pair_mask: np.ndarray
    step_min_distance: np.ndarray
    step_violating_pair_count: np.ndarray
    step_boundary_pair_count: np.ndarray
    band_pair: np.ndarray
    band_min_distance: np.ndarray
    band_min_time: np.ndarray
    band_time_inside: np.ndarray
    min_separation: float | None
    min_separation_time: float | None


def _sublevel_duration(a: np.ndarray, b: np.ndarray, c: np.ndarray, dt: float) -> np.ndarray:
    """``{t in [0,dt] : a t^2 + 2 b t + c < 0}`` 的连续长度（秒）。

    ``a``、``b``、``c`` 形状相同；逐个二次式独立求根（凸二次，子水平集为区间）。
    """
    moving = a > _REL_VEL_EPS_SQ
    disc = b * b - a * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    inv_a = np.divide(1.0, a, out=np.zeros_like(a), where=moving)
    t1 = (-b - sq) * inv_a
    t2 = (-b + sq) * inv_a
    lo = np.clip(t1, 0.0, dt)
    hi = np.clip(t2, 0.0, dt)
    dur_moving = np.where(moving, np.where(disc > 0.0, np.maximum(hi - lo, 0.0), 0.0), 0.0)
    dur_static = np.where((~moving) & (c < 0.0), dt, 0.0)
    return np.where(moving, dur_moving, dur_static)


def _time_in_band(
    t: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    dt: float,
    band_lo: float,
    band_hi: float,
) -> np.ndarray:
    """机对中心距处于 ``[band_lo, band_hi]`` 的连续总时长（秒）。

    长度 = ``length{d < band_hi} - length{d < band_lo}``（凸二次子水平集长度之差）。
    仅对“全程最小距离落在带内”的机对调用，故无相消问题。
    """
    count = int(pair_i.size)
    out = np.zeros(count, dtype=np.float64)
    if count == 0 or t.size < 2:
        return out
    lo_sq = float(band_lo) * float(band_lo)
    hi_sq = float(band_hi) * float(band_hi)
    for start in range(0, count, _PAIR_CHUNK):
        stop = min(start + _PAIR_CHUNK, count)
        ii = pair_i[start:stop]
        jj = pair_j[start:stop]
        r = positions[:-1, ii, :] - positions[:-1, jj, :]
        w = velocities[:, ii, :] - velocities[:, jj, :]
        a = np.einsum("kci,kci->kc", w, w)
        b = np.einsum("kci,kci->kc", r, w)
        r2 = np.einsum("kci,kci->kc", r, r)
        dur_lo = _sublevel_duration(a, b, r2 - lo_sq, dt)
        dur_hi = _sublevel_duration(a, b, r2 - hi_sq, dt)
        out[start:stop] = np.maximum(dur_hi - dur_lo, 0.0).sum(axis=0)
    return out


def _scan_pairs(
    t: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    d_safe: float,
    eps_distance: float,
) -> _PairScan:
    """全部 ``N(N-1)/2`` 机对的连续最近距离、侵犯区间（跨步合并）与边界接触。"""
    n_agents = int(positions.shape[1])
    n_steps = int(positions.shape[0] - 1)
    n_pairs = n_agents * (n_agents - 1) // 2
    d_eval = float(d_safe) - float(eps_distance)
    # handoff §2.2：边界接触带为 [d_safe-eps_distance, d_safe+eps_distance]。
    band_lo = d_eval
    band_hi = float(d_safe) + float(eps_distance)

    pair_min2 = np.full(n_pairs, np.inf, dtype=np.float64)
    pair_min_t = np.full(n_pairs, np.nan, dtype=np.float64)
    step_min2 = np.full(n_steps, np.inf, dtype=np.float64)
    step_viol = np.zeros(n_steps, dtype=np.int64)
    step_band = np.zeros(n_steps, dtype=np.int64)
    ev_pair: list[np.ndarray] = []
    ev_start: list[np.ndarray] = []
    ev_end: list[np.ndarray] = []
    ev_min: list[np.ndarray] = []
    ev_time: list[np.ndarray] = []

    if n_pairs == 0:
        event_pair = np.array([], dtype=np.int64)
        return _PairScan(
            pair_count=0,
            event_pair=event_pair,
            event_start=np.array([], dtype=np.float64),
            event_end=np.array([], dtype=np.float64),
            event_duration=np.array([], dtype=np.float64),
            event_min_distance=np.array([], dtype=np.float64),
            event_min_time=np.array([], dtype=np.float64),
            pair_min_distance=np.array([], dtype=np.float64),
            pair_min_time=np.array([], dtype=np.float64),
            violated_pair_mask=np.array([], dtype=bool),
            step_min_distance=np.full(n_steps, np.nan),
            step_violating_pair_count=step_viol,
            step_boundary_pair_count=step_band,
            band_pair=np.array([], dtype=np.int64),
            band_min_distance=np.array([], dtype=np.float64),
            band_min_time=np.array([], dtype=np.float64),
            band_time_inside=np.array([], dtype=np.float64),
            min_separation=None,
            min_separation_time=None,
        )

    iu, ju = np.triu_indices(n_agents, 1)
    if n_steps == 0:
        diff = positions[0, iu, :] - positions[0, ju, :]
        pair_min2 = np.einsum("pi,pi->p", diff, diff)
        pair_min_t[:] = float(t[0]) if t.size else 0.0
    else:
        with np.errstate(all="ignore"):
            for start in range(0, n_pairs, _PAIR_CHUNK):
                stop = min(start + _PAIR_CHUNK, n_pairs)
                ii = iu[start:stop]
                jj = ju[start:stop]
                count = stop - start

                r = positions[:-1, ii, :] - positions[:-1, jj, :]
                w = velocities[:, ii, :] - velocities[:, jj, :]
                a = np.einsum("kci,kci->kc", w, w)
                b = np.einsum("kci,kci->kc", r, w)
                r2 = np.einsum("kci,kci->kc", r, r)

                moving = a > _REL_VEL_EPS_SQ
                inv_a = np.divide(1.0, a, out=np.zeros_like(a), where=moving)
                tstar = np.where(moving, np.clip(-b * inv_a, 0.0, dt), 0.0)
                # 直接向量求 |r + t* w|²：避免二次展开在近零距离处的相消误差。
                closest = r + tstar[..., None] * w
                dmin2 = np.maximum(np.einsum("kci,kci->kc", closest, closest), 0.0)
                dmin = np.sqrt(dmin2)
                del closest

                # ---- 每机对全程最小距离与时刻 ---------------------------------
                arg_step = np.argmin(dmin, axis=0)
                cols = np.arange(count)
                pair_min2[start:stop] = dmin2[arg_step, cols]
                pair_min_t[start:stop] = t[arg_step] + tstar[arg_step, cols]

                # ---- 每步全局最小距离 / 边界带计数 -----------------------------
                np.minimum(step_min2, np.min(dmin2, axis=1), out=step_min2)
                in_band = (dmin >= band_lo) & (dmin <= band_hi)
                step_band += in_band.sum(axis=1)

                # ---- 严格侵犯区间（d_eval 阈值）-------------------------------
                c = r2 - d_eval * d_eval
                disc = b * b - a * c
                sq = np.sqrt(np.maximum(disc, 0.0))
                t1 = (-b - sq) * inv_a
                t2 = (-b + sq) * inv_a
                static_inside = (~moving) & (c < 0.0)
                enter = np.where(moving, np.maximum(t1, 0.0), np.where(static_inside, 0.0, 0.0))
                exit_ = np.where(moving, np.minimum(t2, dt), np.where(static_inside, dt, 0.0))
                valid = moving & (disc > 0.0) & (enter < exit_) | static_inside
                step_viol += valid.sum(axis=1)

                # ---- 收集侵犯段并跨步/并段合并 --------------------------------
                ci, ki = np.nonzero(valid.T)  # (C,K) 行主序：同机对按时间升序
                if ci.size == 0:
                    continue
                ent = enter.T[ci, ki]
                ext = exit_.T[ci, ki]
                gstart = t[ki] + ent
                gend = t[ki] + ext
                tstar_s = np.clip(tstar.T[ci, ki], ent, ext)
                # 段内最小距离同样用直接向量求，避免近零距离的相消误差。
                r_pk = r.transpose(1, 0, 2)  # (C,K,3)，按机对优先索引
                w_pk = w.transpose(1, 0, 2)
                seg_off = r_pk[ci, ki] + tstar_s[:, None] * w_pk[ci, ki]
                seg_d = np.sqrt(
                    np.maximum(np.einsum("si,si->s", seg_off, seg_off), 0.0)
                )
                seg_t = t[ki] + tstar_s

                new_group = np.ones(ci.size, dtype=bool)
                if ci.size > 1:
                    same_pair = ci[1:] == ci[:-1]
                    contiguous = gstart[1:] <= gend[:-1] + _TIME_TOL
                    new_group[1:] = ~(same_pair & contiguous)
                group_starts = np.flatnonzero(new_group)
                counts = np.diff(np.append(group_starts, ci.size))
                group_lasts = group_starts + counts - 1
                group_id = np.repeat(np.arange(group_starts.size), counts)

                group_min = np.minimum.reduceat(seg_d, group_starts)
                group_min_time = np.full(group_starts.size, np.nan, dtype=np.float64)
                hit = seg_d == np.repeat(group_min, counts)
                hit_idx = np.flatnonzero(hit)
                if hit_idx.size:
                    hit_groups = group_id[hit_idx]
                    first = np.r_[True, hit_groups[1:] != hit_groups[:-1]]
                    group_min_time[hit_groups[first]] = seg_t[hit_idx[first]]
                missing = ~np.isfinite(group_min_time)
                if np.any(missing):
                    group_min_time[missing] = seg_t[group_starts[missing]]

                ev_pair.append((start + ci[group_starts]).astype(np.int64))
                ev_start.append(gstart[group_starts])
                ev_end.append(gend[group_lasts])
                ev_min.append(group_min)
                ev_time.append(group_min_time)

    if ev_pair:
        event_pair = np.concatenate(ev_pair)
        event_start = np.concatenate(ev_start)
        event_end = np.concatenate(ev_end)
        event_min_distance = np.concatenate(ev_min)
        event_min_time = np.concatenate(ev_time)
        order = np.lexsort((event_pair, event_start))
        event_pair = event_pair[order]
        event_start = event_start[order]
        event_end = event_end[order]
        event_duration = event_end - event_start
        event_min_distance = event_min_distance[order]
        event_min_time = event_min_time[order]
    else:
        event_pair = np.array([], dtype=np.int64)
        event_start = np.array([], dtype=np.float64)
        event_end = np.array([], dtype=np.float64)
        event_duration = np.array([], dtype=np.float64)
        event_min_distance = np.array([], dtype=np.float64)
        event_min_time = np.array([], dtype=np.float64)

    violated_pair_mask = np.zeros(n_pairs, dtype=bool)
    if event_pair.size:
        violated_pair_mask[event_pair] = True

    pair_min_distance = np.sqrt(np.maximum(pair_min2, 0.0))
    with np.errstate(all="ignore"):
        best_pair = int(np.argmin(pair_min2)) if n_pairs else -1
        if best_pair >= 0 and np.isfinite(pair_min2[best_pair]):
            min_separation: float | None = float(np.sqrt(pair_min2[best_pair]))
            min_separation_time: float | None = _finite_float(pair_min_t[best_pair])
        else:
            min_separation = None
            min_separation_time = None

    step_min_distance = np.sqrt(np.maximum(step_min2, 0.0))
    step_min_distance[~np.isfinite(step_min2)] = np.nan

    # ---- 边界接触：全程最小距离落在带内且从未严格侵犯 ---------------------
    band_mask = (
        (~violated_pair_mask)
        & np.isfinite(pair_min_distance)
        & (pair_min_distance >= band_lo)
        & (pair_min_distance <= band_hi)
    )
    band_pair = np.flatnonzero(band_mask).astype(np.int64)
    band_min_distance = pair_min_distance[band_pair] if band_pair.size else np.array([], dtype=np.float64)
    band_min_time = pair_min_t[band_pair] if band_pair.size else np.array([], dtype=np.float64)
    band_time_inside = _time_in_band(
        t, positions, velocities, iu[band_pair], ju[band_pair], dt, band_lo, band_hi
    )

    return _PairScan(
        pair_count=n_pairs,
        event_pair=event_pair,
        event_start=event_start,
        event_end=event_end,
        event_duration=event_duration,
        event_min_distance=event_min_distance,
        event_min_time=event_min_time,
        pair_min_distance=pair_min_distance,
        pair_min_time=pair_min_t,
        violated_pair_mask=violated_pair_mask,
        step_min_distance=step_min_distance,
        step_violating_pair_count=step_viol,
        step_boundary_pair_count=step_band,
        band_pair=band_pair,
        band_min_distance=band_min_distance,
        band_min_time=band_min_time,
        band_time_inside=band_time_inside,
        min_separation=min_separation,
        min_separation_time=min_separation_time,
    )


# =============================================================================
# 中心首次进入（从实际轨迹重算 / 校验）
# =============================================================================
def _first_center_entry(
    t: np.ndarray, positions: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """从轨迹重算每机首次进入 ``||p|| <= radius``（闭球）的时刻与步索引。

    使用与 ``geometry.segment_sphere_enter_time(..., closed=True)`` 一致的几何约定；
    返回 ``(times(N,), steps(N,))``，未进入为 ``NaN`` / ``-1``。
    """
    n_steps = int(positions.shape[0] - 1)
    n_agents = int(positions.shape[1])
    times = np.full(n_agents, np.nan, dtype=np.float64)
    steps = np.full(n_agents, -1, dtype=np.int64)
    if n_agents == 0 or positions.shape[0] == 0:
        return times, steps
    eps = float(geometry.GEOM_EPS)
    radius = float(radius)

    r2_0 = np.einsum("ni,ni->n", positions[0], positions[0])
    inside0 = r2_0 <= (radius + eps) ** 2
    times[inside0] = float(t[0])
    steps[inside0] = 0
    todo = ~inside0
    if n_steps == 0 or not np.any(todo):
        return times, steps

    with np.errstate(all="ignore"):
        p = positions[:-1]
        dp = positions[1:] - p
        a = np.einsum("kni,kni->kn", dp, dp)
        b = np.einsum("kni,kni->kn", p, dp)
        r2 = np.einsum("kni,kni->kn", p, p)
        c = r2 - radius * radius
        moving = a > eps * eps
        disc = b * b - a * c
        sq = np.sqrt(np.maximum(disc, 0.0))
        inv_a = np.divide(1.0, a, out=np.zeros_like(a), where=moving)
        s1 = (-b - sq) * inv_a
        candidate = moving & (disc >= 0.0) & (s1 >= -1e-12) & (s1 <= 1.0 + 1e-12)
        s = np.clip(s1, 0.0, 1.0)
        big = np.where(candidate, s, np.inf)
        first_step = np.argmin(big, axis=0)
        first_s = big[first_step, np.arange(n_agents)]
        found = np.isfinite(first_s) & todo
        if np.any(found):
            idx = np.flatnonzero(found)
            step_idx = first_step[idx]
            step_dt = t[step_idx + 1] - t[step_idx]
            times[idx] = t[step_idx] + first_s[idx] * step_dt
            steps[idx] = step_idx
    return times, steps


def _interp_positions(
    t: np.ndarray, positions: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    """在常速段轨迹上线性插值 ``(len(query_times), 3)``。"""
    q = np.asarray(query_times, dtype=np.float64).reshape(-1)
    n_agents = int(positions.shape[1])
    out = np.full((q.size, 3), np.nan, dtype=np.float64)
    valid = np.isfinite(q)
    if not np.any(valid) or t.size == 0:
        return out
    if t.size == 1:
        out[valid] = positions[0, :3][None, :] if n_agents == 1 else np.nan
        return out
    qv = q[valid]
    idx = np.clip(np.searchsorted(t, qv, side="right") - 1, 0, t.size - 2)
    step_dt = t[idx + 1] - t[idx]
    frac = np.clip(np.divide(qv - t[idx], step_dt, out=np.zeros_like(qv), where=step_dt > 0.0), 0.0, 1.0)
    agents = np.arange(qv.size)
    out[valid] = positions[idx, agents] + frac[:, None] * (positions[idx + 1, agents] - positions[idx, agents])
    return out


# =============================================================================
# 到达时刻 / 停滞 / 诊断
# =============================================================================
def _arrival_from_stages(t: np.ndarray, stages: np.ndarray, limit: float) -> np.ndarray:
    """由 ``stages`` 首次变为 ARRIVED 的网格时刻推出到达时间（NaN=未到达）。"""
    n_agents = int(stages.shape[1])
    times = np.full(n_agents, np.nan, dtype=np.float64)
    if stages.shape[0] == 0:
        return times
    arrived = stages == STAGE_ARRIVED
    has = arrived.any(axis=0)
    if np.any(has):
        first = np.argmax(arrived, axis=0)
        raw = t[first]
        times = np.where(has & np.isfinite(raw) & (raw <= limit + _TIME_TOL), raw, np.nan)
    return times


def _detect_stall(
    traj: _Trajectory,
    goals: np.ndarray,
    arrivals: np.ndarray,
    release: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """handoff §6.3 停滞指示与低进展徘徊指示，逐机返回 bool。"""
    n_agents = traj.n_agents
    stalled = np.zeros(n_agents, dtype=bool)
    low_mobile = np.zeros(n_agents, dtype=bool)
    n_steps = traj.n_steps
    if n_steps <= 0 or n_agents == 0:
        return stalled, low_mobile

    window = float(cfg.metrics.stall_window)
    path_thr = float(cfg.metrics.stall_path_threshold)
    waypoint_thr = float(cfg.metrics.stall_waypoint_threshold)
    window_steps = int(round(window / traj.dt))
    if window_steps <= 0 or n_steps < window_steps:
        return stalled, low_mobile

    t = traj.t
    positions = traj.positions
    stages = traj.stages

    seg = np.linalg.norm(np.diff(positions, axis=0), axis=-1)  # (K,N)
    cum_path = np.concatenate(
        [np.zeros((1, n_agents), dtype=np.float64), np.cumsum(seg, axis=0)], axis=0
    )

    # 各网格点的 waypoint 距离（WAITING/ARRIVED 无 waypoint → NaN）
    goal_rows = np.broadcast_to(goals[None, :, :], positions.shape)
    dist_goal = np.linalg.norm(positions - goal_rows, axis=-1)
    dist_center = np.linalg.norm(positions, axis=-1)
    waypoint_dist = np.full((n_steps + 1, n_agents), np.nan, dtype=np.float64)
    to_center = stages == STAGE_TO_CENTER
    to_goal = stages == STAGE_TO_GOAL
    waypoint_dist[to_center] = dist_center[to_center]
    waypoint_dist[to_goal] = dist_goal[to_goal]

    # 同阶段已持续步数（阶段切换重置）
    change = np.zeros((n_steps + 1, n_agents), dtype=bool)
    change[0] = True
    if n_steps >= 1:
        change[1:] = stages[1:] != stages[:-1]
    index_grid = np.arange(n_steps + 1, dtype=np.int64)[:, None]
    last_change = np.maximum.accumulate(np.where(change, index_grid, 0), axis=0)
    run_steps = index_grid - last_change

    end_idx = np.arange(window_steps, n_steps + 1, dtype=np.int64)
    start_idx = end_idx - window_steps
    stage_ok = np.isin(stages[end_idx], (STAGE_TO_CENTER, STAGE_TO_GOAL))
    run_ok = run_steps[end_idx] >= window_steps
    not_arrived = ~np.isfinite(arrivals)[None, :]
    window_path = cum_path[end_idx] - cum_path[start_idx]
    waypoint_drop = waypoint_dist[start_idx] - waypoint_dist[end_idx]

    times_end = t[end_idx][:, None]
    rel = np.asarray(release, dtype=np.float64).reshape(1, -1)
    released = np.where(np.isfinite(rel), times_end >= rel, True)

    candidate = stage_ok & run_ok & not_arrived & released & np.isfinite(waypoint_drop)
    slow = window_path < path_thr
    no_progress = waypoint_drop < waypoint_thr
    stalled = np.any(candidate & slow & no_progress, axis=0)
    low_mobile = np.any(candidate & (~slow) & no_progress, axis=0)
    return stalled, low_mobile


def _diag_active_mask(raw: Any, active: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """返回 ``(K,N)`` 活动掩码，用于 ``(K,N)`` 布局的诊断数组。"""
    n_steps, n_agents = shape
    mask = np.asarray(active, dtype=bool)
    if mask.shape != shape:
        mask = np.ones(shape, dtype=bool)
    diag_mask = _opt(raw, "diag_active_mask")
    if diag_mask is not None:
        arr = np.asarray(diag_mask, dtype=bool)
        if arr.shape == shape:
            return arr
    ids = _opt(raw, "diag_active_ids")
    if ids is not None:
        arr = np.asarray(ids)
        if arr.ndim == 2:
            if arr.shape == shape:
                return np.isfinite(arr) & (arr >= 0)
            if arr.shape == (n_agents, n_steps):
                return (np.isfinite(arr) & (arr >= 0)).T
            if arr.shape[0] == n_steps:
                return np.isfinite(arr) & (arr >= 0)
    return mask


def _aggregate_planner_diagnostics(raw: Any, active: np.ndarray) -> tuple[dict[str, Any], int | None]:
    """聚合规划诊断；缺失键按 0 处理。返回 ``(diagnostics_dict, per_call_count)``。"""
    n_steps = int(active.shape[0])
    n_agents = int(active.shape[1])
    bool_keys = (
        "diag_sample_exhausted",
        "diag_used_fallback",
        "diag_solver_infeasible",
        "diag_numerical_error",
    )
    out: dict[str, Any] = {
        "sample_exhausted_count": 0,
        "solver_fallback_count": 0,
        "solver_infeasible_count": 0,
        "constraint_violation_max": 0.0,
        "initial_overlap_pairs_total": 0,
        "numerical_error_count": 0,
        "active_planning_calls": 0,
        "candidate_count_tested_total": 0,
    }

    step_ids = _opt(raw, "diag_step")
    agent_ids = _opt(raw, "diag_agent_id")
    per_call_count: int | None = None
    if step_ids is not None:
        per_call_count = int(np.asarray(step_ids).reshape(-1).size)
    elif agent_ids is not None:
        per_call_count = int(np.asarray(agent_ids).reshape(-1).size)

    if per_call_count is None:
        # 兼容 (K,N) 掩码布局 / (K,) 逐步聚合布局
        mask = _diag_active_mask(raw, active, (n_steps, n_agents))
        layout_2d = False
        for key in bool_keys + ("diag_constraint_violation_max", "diag_initial_overlap_pairs", "diag_candidate_count_tested"):
            arr = _opt(raw, key)
            if arr is None:
                continue
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape == (n_steps, n_agents):
                layout_2d = True
                break
        if layout_2d:
            for key in bool_keys:
                arr = _opt(raw, key)
                if arr is not None:
                    a = np.asarray(arr, dtype=bool)
                    if a.shape != (n_steps, n_agents):
                        raise ValueError(f"{key} shape {a.shape} != ({n_steps},{n_agents})")
                    out[_bool_count_key(key)] = int(np.sum(a & mask))
            arr = _opt(raw, "diag_constraint_violation_max")
            if arr is not None:
                a = np.asarray(arr, dtype=np.float64)
                vals = a[mask]
                if vals.size and np.any(np.isfinite(vals)):
                    out["constraint_violation_max"] = float(np.nanmax(vals))
            arr = _opt(raw, "diag_initial_overlap_pairs")
            if arr is not None:
                a = np.asarray(arr, dtype=np.float64)
                out["initial_overlap_pairs_total"] = int(np.nansum(a[mask]))
            arr = _opt(raw, "diag_candidate_count_tested")
            if arr is not None:
                a = np.asarray(arr, dtype=np.float64)
                out["candidate_count_tested_total"] = int(np.nansum(a[mask]))
            out["active_planning_calls"] = int(mask.sum())
        else:
            # 1-D：优先解释为 (K,) 逐步聚合，否则解释为 (C,) 每调用
            sizes = []
            for key in bool_keys + ("diag_constraint_violation_max", "diag_initial_overlap_pairs"):
                arr = _opt(raw, key)
                if arr is not None and np.asarray(arr).ndim == 1:
                    sizes.append(int(np.asarray(arr).size))
            layout_per_step = any(size == n_steps for size in sizes)
            if layout_per_step:
                # 1-D 逐步聚合布局：数值按计数求和（bool 视为 0/1），避免把计数>1 当 True。
                for key in bool_keys:
                    arr = _opt(raw, key)
                    if arr is not None:
                        out[_bool_count_key(key)] = int(
                            np.sum(np.asarray(arr, dtype=np.float64))
                        )
                arr = _opt(raw, "diag_constraint_violation_max")
                if arr is not None and np.asarray(arr).size:
                    out["constraint_violation_max"] = float(np.nanmax(np.asarray(arr, dtype=np.float64)))
                arr = _opt(raw, "diag_initial_overlap_pairs")
                if arr is not None:
                    out["initial_overlap_pairs_total"] = int(np.nansum(np.asarray(arr, dtype=np.float64)))
                arr = _opt(raw, "diag_candidate_count_tested")
                if arr is not None:
                    out["candidate_count_tested_total"] = int(np.nansum(np.asarray(arr, dtype=np.float64)))
                out["active_planning_calls"] = int(active.sum())
            elif sizes and len(set(sizes)) == 1:
                per_call_count = sizes[0]
    if per_call_count is not None:
        for key in bool_keys:
            arr = _opt(raw, key)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=bool).reshape(-1)
            if a.size != per_call_count:
                raise ValueError(f"{key} has {a.size} entries, expected {per_call_count} planning calls")
            out[_bool_count_key(key)] = int(np.sum(a))
        arr = _opt(raw, "diag_constraint_violation_max")
        if arr is not None:
            a = np.asarray(arr, dtype=np.float64).reshape(-1)
            if a.size != per_call_count:
                raise ValueError("diag_constraint_violation_max length does not match planning calls")
            finite = a[np.isfinite(a)]
            out["constraint_violation_max"] = float(np.max(finite)) if finite.size else 0.0
        arr = _opt(raw, "diag_initial_overlap_pairs")
        if arr is not None:
            a = np.asarray(arr, dtype=np.float64).reshape(-1)
            if a.size != per_call_count:
                raise ValueError("diag_initial_overlap_pairs length does not match planning calls")
            out["initial_overlap_pairs_total"] = int(np.nansum(a))
        arr = _opt(raw, "diag_candidate_count_tested")
        if arr is not None:
            a = np.asarray(arr, dtype=np.float64).reshape(-1)
            if a.size != per_call_count:
                raise ValueError("diag_candidate_count_tested length does not match planning calls")
            out["candidate_count_tested_total"] = int(np.nansum(a))
        # 每调用一行的诊断若为空（例如 Direct 不产生诊断行），分母退回 active 计数。
        out["active_planning_calls"] = (
            int(per_call_count) if per_call_count > 0 else int(active.sum())
        )
    else:
        out["active_planning_calls"] = int(active.sum())
    return out, per_call_count


def _bool_count_key(name: str) -> str:
    mapping = {
        "diag_sample_exhausted": "sample_exhausted_count",
        "diag_used_fallback": "solver_fallback_count",
        "diag_solver_infeasible": "solver_infeasible_count",
        "diag_numerical_error": "numerical_error_count",
    }
    return mapping[name]


def _aggregate_planning(
    raw: Any, active: np.ndarray, calls: int | None
) -> dict[str, Any]:
    """规划墙钟聚合（CONTRACTS §11.2 ``planning``）。"""
    step_seconds = _opt(raw, "planning_step_seconds", "planning_time_s")
    mean_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None
    total_s: float | None = None
    if step_seconds is not None:
        arr = np.asarray(step_seconds, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr) & (arr >= 0.0)]
        if finite.size:
            mean_ms = float(np.mean(finite) * 1000.0)
            p95_ms = float(np.percentile(finite, 95.0) * 1000.0)
            max_ms = float(np.max(finite) * 1000.0)
            total_s = float(np.sum(finite))
    per_active_ms: float | None = None
    if total_s is not None and calls:
        per_active_ms = float(total_s * 1000.0 / float(calls))
    return {
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "max_ms": max_ms,
        "total_s": total_s,
        "calls": None if calls is None else int(calls),
        "per_active_call_ms": per_active_ms,
    }


def _release_rank(release_planned: np.ndarray) -> np.ndarray:
    """由计划放行时刻给出 0 基排名；NaN 保持 NaN。"""
    ranks = np.full(release_planned.size, np.nan, dtype=np.float64)
    finite = np.isfinite(release_planned)
    if np.any(finite):
        order = np.argsort(release_planned[finite], kind="stable")
        sorted_ranks = np.empty(int(finite.sum()), dtype=np.float64)
        sorted_ranks[order] = np.arange(int(finite.sum()), dtype=np.float64)
        ranks[finite] = sorted_ranks
    return ranks


# =============================================================================
# 主计算
# =============================================================================
def compute_metrics(
    traj: Mapping[str, Any],
    scene: Any,
    cfg: Config,
    *,
    model: str,
    scenario: str,
    seed: int,
    scene_hash: str,
    config_hash: str,
    center_entry_times: np.ndarray | None,
    center_entry_steps: np.ndarray | None = None,
    release_planned: np.ndarray | None = None,
    release_actual: np.ndarray | None = None,
    status: dict[str, Any] | None = None,
    experiment_id: str | None = None,
) -> MetricsResult:
    """从保存轨迹离线重算全部指标（CONTRACTS §11）。

    参数形状见模块 docstring 与 CONTRACTS §11；``center_entry_times`` 为仿真提供的
    步内首次中心进入时刻（``NaN`` = 未访问），可用轨迹校验其一致性。
    """
    scenario = str(scenario).upper()
    if scenario not in ("A", "B"):
        raise ValueError(f"scenario must be 'A' or 'B', got {scenario!r}")

    tr = _normalize_traj(traj, cfg)
    t = tr.t
    positions = tr.positions
    velocities = tr.velocities
    stages = tr.stages
    n_agents = tr.n_agents
    n_steps = tr.n_steps
    dt = tr.dt
    observation_time = float(cfg.observation_time)
    d_safe = float(cfg.simulation.d_safe)
    eps_distance = float(cfg.metrics.eps_distance)
    center_tolerance = float(cfg.simulation.center_tolerance)
    init_radius = float(cfg.simulation.init_sphere_radius)
    goal_tolerance = float(cfg.simulation.goal_tolerance)

    # ---- 场景数组（鸭子类型：只需要 goals/agent_ids/scene_id）-------------
    goals = np.asarray(getattr(scene, "goals"), dtype=np.float64)
    agent_ids_raw = np.asarray(getattr(scene, "agent_ids", np.arange(n_agents)))
    scene_id_raw = getattr(scene, "scene_id", None)
    if goals.shape != (n_agents, 3):
        raise ValueError(f"scene.goals shape {goals.shape} != ({n_agents},3)")
    if agent_ids_raw.shape != (n_agents,):
        raise ValueError(f"scene.agent_ids shape {agent_ids_raw.shape} != ({n_agents},)")
    agent_ids = agent_ids_raw.astype(np.int64, copy=False)
    scene_id = None if scene_id_raw is None else str(scene_id_raw)

    # ---- 连续安全扫描 -----------------------------------------------------
    scan = _scan_pairs(t, positions, velocities, dt, d_safe, eps_distance)

    # ---- 到达时刻：stages 推导 + 仿真提供值交叉校验 -----------------------
    arrival_derived = _arrival_from_stages(t, stages, observation_time)
    arrival_provided_raw = _opt(traj, "arrival_times")
    arrival_checked = 0
    arrival_max_diff: float | None = None
    arrival_mismatch = 0
    if arrival_provided_raw is not None:
        provided = np.asarray(arrival_provided_raw, dtype=np.float64).reshape(-1)
        if provided.size != n_agents:
            raise ValueError(f"arrival_times size {provided.size} != N={n_agents}")
        provided = np.where(np.isfinite(provided), provided, np.nan)
        provided = np.where(provided <= observation_time + _TIME_TOL, provided, np.nan)
        both = np.isfinite(provided) & np.isfinite(arrival_derived)
        arrival_checked = int(np.sum(both))
        if arrival_checked:
            diffs = np.abs(provided[both] - arrival_derived[both])
            arrival_max_diff = float(np.max(diffs))
            arrival_mismatch += int(np.sum(diffs > _ARRIVAL_TIME_TOL))
        arrival_mismatch += int(np.sum(np.isfinite(provided) & ~np.isfinite(arrival_derived)))
        arrival_mismatch += int(np.sum(~np.isfinite(provided) & np.isfinite(arrival_derived)))
        arrival = provided
    else:
        arrival = arrival_derived
    arrival = np.asarray(arrival, dtype=np.float64).reshape(-1)
    arrived_mask = np.isfinite(arrival)
    arrival_count = int(np.sum(arrived_mask))
    completed_all = arrival_count == n_agents

    # ---- 中心首次进入 -----------------------------------------------------
    recomputed_center_times, recomputed_center_steps = _first_center_entry(
        t, positions, center_tolerance
    )
    if center_entry_times is not None:
        center_provided = np.asarray(center_entry_times, dtype=np.float64).reshape(-1)
        if center_provided.size != n_agents:
            raise ValueError(f"center_entry_times size {center_provided.size} != N={n_agents}")
        center_provided = np.where(np.isfinite(center_provided), center_provided, np.nan)
        center_source = "provided"
    else:
        traj_center = _opt(traj, "center_entry_times")
        if traj_center is not None:
            center_provided = np.asarray(traj_center, dtype=np.float64).reshape(-1)
            if center_provided.size != n_agents:
                raise ValueError(f"traj['center_entry_times'] size != N={n_agents}")
            center_provided = np.where(np.isfinite(center_provided), center_provided, np.nan)
            center_source = "provided"
        else:
            center_provided = np.full(n_agents, np.nan, dtype=np.float64)
            center_source = "recomputed"

    center_used = center_provided.copy()
    if scenario == "A":
        # 场景 A 无中心子任务：仿真不记录时用轨迹重算值仅作诊断，rate 仍为 null。
        center_used = np.where(np.isfinite(center_provided), center_provided, recomputed_center_times)
    elif center_source == "recomputed":
        center_used = recomputed_center_times.copy()

    steps_provided_raw = center_entry_steps if center_entry_steps is not None else _opt(traj, "center_entry_steps")
    center_step_checked = 0
    center_step_mismatch = 0
    if steps_provided_raw is not None:
        steps_prov = np.asarray(steps_provided_raw).reshape(-1)
        if steps_prov.size != n_agents:
            raise ValueError(f"center_entry_steps size {steps_prov.size} != N={n_agents}")
        known = steps_prov >= 0
        known_recomp = recomputed_center_steps >= 0
        center_step_checked = int(np.sum(known & known_recomp))
        if center_step_checked:
            center_step_mismatch += int(
                np.sum(steps_prov[known & known_recomp] != recomputed_center_steps[known & known_recomp])
            )
        center_step_mismatch += int(np.sum(known & ~known_recomp))
        center_step_mismatch += int(np.sum(~known & known_recomp))

    center_checked = 0
    center_max_diff: float | None = None
    center_mismatch = 0
    if center_source == "provided":
        both_known = np.isfinite(center_provided) & np.isfinite(recomputed_center_times)
        center_checked = int(np.sum(both_known))
        if center_checked:
            diffs = np.abs(center_provided[both_known] - recomputed_center_times[both_known])
            center_max_diff = float(np.max(diffs))
            center_mismatch += int(np.sum(diffs > _CENTER_TIME_TOL))
        center_mismatch += int(
            np.sum(np.isfinite(center_provided) & ~np.isfinite(recomputed_center_times))
        )
        if scenario == "B":
            center_mismatch += int(
                np.sum(~np.isfinite(center_provided) & np.isfinite(recomputed_center_times))
            )
    center_pos_violations = 0
    finite_center = np.isfinite(center_provided)
    if np.any(finite_center):
        tq = center_provided[finite_center]
        pos_q = _interp_positions(t, positions, tq)
        norms = np.linalg.norm(pos_q, axis=-1)
        in_grid = (tq >= t[0] - _TIME_TOL) & (tq <= t[-1] + _TIME_TOL)
        bad = ~in_grid | ~np.isfinite(norms) | (norms > center_tolerance + _CENTER_POS_TOL)
        center_pos_violations = int(np.sum(bad))
    center_crossing_count = int(np.sum(np.isfinite(center_used)))

    # ---- 路径 / 阶段 ------------------------------------------------------
    if n_steps > 0:
        seg_len = np.linalg.norm(np.diff(positions, axis=0), axis=-1)  # (K,N)
        path_length = seg_len.sum(axis=0)
    else:
        path_length = np.zeros(n_agents, dtype=np.float64)
    final_goal_distance = np.linalg.norm(positions[-1] - goals, axis=-1)
    final_stage = stages[-1].astype(np.int64, copy=True)

    phase_wait, phase_center, phase_goal = stage_durations(
        t, stages, final_time=float(t[-1]) if t.size else 0.0
    )

    # ---- 放行 / 等待 ------------------------------------------------------
    if scenario == "A":
        release_eff = np.zeros(n_agents, dtype=np.float64)
        wait_duration = np.zeros(n_agents, dtype=np.float64)
    else:
        release_actual_raw = release_actual if release_actual is not None else _opt(traj, "release_actual")
        if release_actual_raw is not None:
            release_eff = np.asarray(release_actual_raw, dtype=np.float64).reshape(-1)
            if release_eff.size != n_agents:
                raise ValueError(f"release_actual size {release_eff.size} != N={n_agents}")
            release_eff = np.where(np.isfinite(release_eff), release_eff, np.nan)
            wait_duration = release_eff.copy()
        else:
            release_eff = np.full(n_agents, np.nan, dtype=np.float64)
            wait_duration = np.full(n_agents, np.nan, dtype=np.float64)

    planned_raw = release_planned if release_planned is not None else _opt(traj, "release_planned")
    if planned_raw is not None:
        release_planned_arr = np.asarray(planned_raw, dtype=np.float64).reshape(-1)
        if release_planned_arr.size != n_agents:
            raise ValueError(f"release_planned size {release_planned_arr.size} != N={n_agents}")
        release_planned_arr = np.where(np.isfinite(release_planned_arr), release_planned_arr, np.nan)
    else:
        release_planned_arr = np.full(n_agents, np.nan, dtype=np.float64)
    release_rank = _release_rank(release_planned_arr)

    rel_for_travel = release_eff if scenario == "B" else np.zeros(n_agents, dtype=np.float64)
    post_release = np.full(n_agents, np.nan, dtype=np.float64)
    both_arr = arrived_mask & np.isfinite(rel_for_travel)
    post_release[both_arr] = arrival[both_arr] - rel_for_travel[both_arr]
    mean_post_release = _mean_or_none(post_release)

    wait_finite = wait_duration[np.isfinite(wait_duration)]
    mean_wait = float(np.mean(wait_finite)) if wait_finite.size else None
    total_wait = float(np.sum(wait_finite)) if wait_finite.size else None

    # ---- 侵犯参与与安全完成 ----------------------------------------------
    violated_agents = np.zeros(n_agents, dtype=bool)
    if scan.event_pair.size:
        iu, ju = np.triu_indices(n_agents, 1)
        violated_agents[iu[scan.event_pair]] = True
        violated_agents[ju[scan.event_pair]] = True
    safe_completed = arrived_mask & (~violated_agents)
    safe_completion_count = int(np.sum(safe_completed))
    team_safe_success = bool(safe_completion_count == n_agents and n_agents > 0)

    # ---- 时间指标 ---------------------------------------------------------
    all_arrival_time: float | None = None
    if completed_all and n_agents > 0:
        all_arrival_time = float(np.max(arrival))

    t90_threshold = int(math.ceil(0.9 * n_agents)) if n_agents > 0 else 0
    t90: float | None = None
    if t90_threshold > 0 and arrival_count >= t90_threshold:
        ordered = np.sort(arrival[arrived_mask])
        value = float(ordered[t90_threshold - 1])
        if t.size:
            idx = int(np.searchsorted(t, value, side="left"))
            if idx >= t.size:
                idx = t.size - 1
            t90 = float(t[idx])
        else:  # pragma: no cover - t 至少含一个网格点
            t90 = value

    completion_limited = np.where(arrived_mask, np.minimum(arrival, observation_time), observation_time)
    mean_completion_time_limited = float(np.mean(completion_limited)) if n_agents > 0 else None

    # ---- 路程指标 ---------------------------------------------------------
    average_path_length = float(np.mean(path_length)) if n_agents > 0 else None
    lower_bound = 2.0 * init_radius - goal_tolerance
    completed_mask = arrived_mask
    completed_count_for_path_metrics = int(np.sum(completed_mask))
    mean_path_efficiency_excess: float | None = None
    if completed_count_for_path_metrics > 0 and lower_bound > 0.0:
        mean_path_efficiency_excess = float(
            np.mean(path_length[completed_mask] / lower_bound - 1.0)
        )
    mean_path_over_40_minus_1: float | None = None
    if average_path_length is not None and init_radius > 0.0:
        mean_path_over_40_minus_1 = float(average_path_length / (2.0 * init_radius) - 1.0)

    # ---- 停滞 -------------------------------------------------------------
    stalled, low_progress_mobile = _detect_stall(tr, goals, arrival, release_eff, cfg)

    # ---- 规划诊断与墙钟 ---------------------------------------------------
    planner_diagnostics, per_call_count = _aggregate_planner_diagnostics(traj, tr.active)
    planning = _aggregate_planning(
        traj, tr.active, planner_diagnostics.get("active_planning_calls")
    )
    diag_active_call_mismatch: int | None = None
    if per_call_count is not None:
        diag_active_call_mismatch = int(per_call_count - int(tr.active.sum()))

    # ---- 未完成分类 -------------------------------------------------------
    if scenario == "B":
        released_by_T = np.where(
            np.isfinite(release_eff), release_eff <= observation_time + _TIME_TOL, True
        )
    else:
        released_by_T = np.ones(n_agents, dtype=bool)
    uncompleted_mask = ~arrived_mask
    not_released = int(np.sum(uncompleted_mask & ~released_by_T))
    if scenario == "B":
        center_done = np.isfinite(center_used)
        not_center = int(np.sum(uncompleted_mask & released_by_T & ~center_done))
        center_done_not_arrived = int(np.sum(uncompleted_mask & released_by_T & center_done))
    else:
        not_center = 0
        center_done_not_arrived = int(np.sum(uncompleted_mask & released_by_T))

    # ---- 状态 -------------------------------------------------------------
    status_in: dict[str, Any] = dict(status) if status is not None else {}
    if "numerical_error" not in status_in:
        raw_ne = _opt(traj, "status_numerical_error")
        status_in["numerical_error"] = _as_bool(raw_ne) if raw_ne is not None else False
    if "valid" not in status_in:
        raw_valid = _opt(traj, "status_valid")
        status_in["valid"] = _as_bool(raw_valid) if raw_valid is not None else True
    if "completed" not in status_in:
        status_in["completed"] = bool(completed_all)
    if "timeout" not in status_in:
        status_in["timeout"] = bool(not status_in["completed"])
    numerical_error = bool(status_in.get("numerical_error", False)) or (not tr.finite_data)
    valid = bool(status_in.get("valid", True)) and tr.finite_data
    status_out = {
        "completed": bool(status_in.get("completed", completed_all)),
        "timeout": bool(status_in.get("timeout", not completed_all)),
        "numerical_error": numerical_error,
        "valid": valid,
    }

    n_steps_executed_raw = _opt(traj, "n_steps_executed")
    n_steps_executed = int(n_steps_executed_raw) if n_steps_executed_raw is not None else None
    expected_n_steps = int(round(observation_time / dt)) if dt > 0 else None
    grid_end = float(t[-1]) if t.size else None
    grid_complete = bool(
        grid_end is not None and abs(grid_end - observation_time) <= max(1e-9, dt * 1e-6)
    )

    consistency = {
        "grid_steps": int(n_steps),
        "expected_grid_steps": expected_n_steps,
        "grid_end_time": _finite_float(grid_end) if grid_end is not None else None,
        "observation_time": float(observation_time),
        "grid_complete": grid_complete,
        "non_finite_data": bool(not tr.finite_data),
        "n_steps_executed": n_steps_executed,
        "center_entry_source": center_source,
        "center_entry_checked": int(center_checked),
        "center_entry_max_abs_diff_s": _finite_float(center_max_diff),
        "center_entry_mismatch_count": int(center_mismatch),
        "center_entry_position_violations": int(center_pos_violations),
        "center_entry_step_checked": int(center_step_checked),
        "center_entry_step_mismatch_count": int(center_step_mismatch),
        "arrival_checked": int(arrival_checked),
        "arrival_max_abs_diff_s": _finite_float(arrival_max_diff),
        "arrival_mismatch_count": int(arrival_mismatch),
        "diag_active_call_mismatch": diag_active_call_mismatch,
    }

    # ---- 事件 / 边界接触 DataFrame ---------------------------------------
    iu, ju = np.triu_indices(n_agents, 1)
    if scan.event_pair.size:
        violation_events = pd.DataFrame(
            {
                "pair_id": scan.event_pair.astype(np.int64),
                "agent_i": agent_ids[iu[scan.event_pair]],
                "agent_j": agent_ids[ju[scan.event_pair]],
                "start_time": scan.event_start,
                "end_time": scan.event_end,
                "duration": scan.event_duration,
                "min_distance": scan.event_min_distance,
                "min_distance_time": scan.event_min_time,
            }
        )
    else:
        violation_events = _empty_frame(
            VIOLATION_EVENT_COLUMNS,
            {
                "pair_id": np.int64,
                "agent_i": np.int64,
                "agent_j": np.int64,
                "start_time": np.float64,
                "end_time": np.float64,
                "duration": np.float64,
                "min_distance": np.float64,
                "min_distance_time": np.float64,
            },
        )

    if scan.band_pair.size:
        boundary_contacts = pd.DataFrame(
            {
                "pair_id": scan.band_pair.astype(np.int64),
                "agent_i": agent_ids[iu[scan.band_pair]],
                "agent_j": agent_ids[ju[scan.band_pair]],
                "min_distance": scan.band_min_distance,
                "min_distance_time": scan.band_min_time,
                "min_time_inside_band": scan.band_time_inside,
            }
        )
    else:
        boundary_contacts = _empty_frame(
            BOUNDARY_CONTACT_COLUMNS,
            {
                "pair_id": np.int64,
                "agent_i": np.int64,
                "agent_j": np.int64,
                "min_distance": np.float64,
                "min_distance_time": np.float64,
                "min_time_inside_band": np.float64,
            },
        )

    # ---- 曲线 -------------------------------------------------------------
    if n_steps > 0:
        min_distance_curve = pd.DataFrame(
            {
                "step": np.arange(n_steps, dtype=np.int64),
                "t": t[:-1],
                "min_distance": scan.step_min_distance,
                "violating_pair_count": scan.step_violating_pair_count.astype(np.int64),
                "boundary_pair_count": scan.step_boundary_pair_count.astype(np.int64),
            }
        )
    else:
        min_distance_curve = _empty_frame(
            MIN_DISTANCE_CURVE_COLUMNS,
            {
                "step": np.int64,
                "t": np.float64,
                "min_distance": np.float64,
                "violating_pair_count": np.int64,
                "boundary_pair_count": np.int64,
            },
        )

    if t.size:
        grid = t
        arrival_curve = pd.DataFrame(
            {
                "step": np.arange(t.size, dtype=np.int64),
                "t": grid,
                "cumulative_arrivals": _cumulative_counts(grid, arrival[arrived_mask]),
                "cumulative_center_visits": _cumulative_counts(
                    grid, center_used[np.isfinite(center_used)]
                ),
                "cumulative_safe_completions_known_later": _cumulative_counts(
                    grid, arrival[safe_completed & arrived_mask]
                ),
            }
        )
    else:  # pragma: no cover - t 至少含一个网格点
        arrival_curve = _empty_frame(
            ARRIVAL_CURVE_COLUMNS,
            {
                "step": np.int64,
                "t": np.float64,
                "cumulative_arrivals": np.int64,
                "cumulative_center_visits": np.int64,
                "cumulative_safe_completions_known_later": np.int64,
            },
        )

    # ---- agents DataFrame -------------------------------------------------
    release_planned_col = release_planned_arr
    if scenario == "B":
        release_actual_col = release_eff.copy()
        wait_col = wait_duration.copy()
    else:
        release_actual_col = (
            np.asarray(release_actual, dtype=np.float64).reshape(-1).copy()
            if release_actual is not None
            else np.full(n_agents, np.nan, dtype=np.float64)
        )
        wait_col = np.zeros(n_agents, dtype=np.float64)
    agents = pd.DataFrame(
        {
            "agent_id": agent_ids,
            "release_rank": release_rank,
            "release_planned": release_planned_col,
            "release_actual": release_actual_col,
            "wait_duration": wait_col,
            "center_entry_time": center_used,
            "center_crossing": np.isfinite(center_used),
            "arrival_time": arrival,
            "post_release_travel_time": post_release,
            "phase_wait_duration": phase_wait,
            "phase_to_center_duration": phase_center,
            "phase_to_goal_duration": phase_goal,
            "path_length": path_length,
            "final_goal_distance": final_goal_distance,
            "final_stage": final_stage,
            "violated": violated_agents,
            "safe_completed": safe_completed,
            "stalled": stalled,
            "low_progress_mobile": low_progress_mobile,
        }
    )

    # ---- metrics dict -----------------------------------------------------
    violation_pair_seconds = (
        float(np.sum(scan.event_duration)) if scan.event_duration.size else 0.0
    )
    first_violation_time = (
        float(np.min(scan.event_start)) if scan.event_start.size else None
    )
    center_crossing_rate = (
        (center_crossing_count / n_agents) if scenario == "B" and n_agents > 0 else None
    )
    experiment_id_resolved = experiment_id
    if experiment_id_resolved is None:
        experiment_id_resolved = _opt(traj, "experiment_id")
    metrics: dict[str, Any] = {
        "experiment_id": None if experiment_id_resolved is None else str(experiment_id_resolved),
        "model": str(model),
        "scenario": scenario,
        "scene_id": scene_id,
        "seed": int(seed),
        "scene_hash": None if scene_hash is None else str(scene_hash),
        "config_hash": None if config_hash is None else str(config_hash),
        "status": status_out,
        "n_agents": int(n_agents),
        "arrival_count": int(arrival_count),
        "arrival_rate": float(arrival_count / n_agents) if n_agents > 0 else None,
        "completed_all": bool(completed_all),
        "safe_completion_count": int(safe_completion_count),
        "safe_completion_rate": float(safe_completion_count / n_agents) if n_agents > 0 else None,
        "center_crossing_count": int(center_crossing_count),
        "center_crossing_rate": center_crossing_rate,
        "team_safe_success": team_safe_success,
        "min_separation": scan.min_separation,
        "min_separation_time": scan.min_separation_time,
        "violating_pair_count": int(np.sum(scan.violated_pair_mask)),
        "violation_event_count": int(scan.event_pair.size),
        "violation_pair_seconds": violation_pair_seconds,
        "first_violation_time": first_violation_time,
        "all_arrival_time": all_arrival_time,
        "t90": t90,
        "mean_completion_time_limited": mean_completion_time_limited,
        "average_path_length": average_path_length,
        "mean_path_efficiency_excess": mean_path_efficiency_excess,
        "mean_path_over_40_minus_1": mean_path_over_40_minus_1,
        "completed_count_for_path_metrics": int(completed_count_for_path_metrics),
        "waiting": {
            "mean_wait": mean_wait,
            "total_wait": total_wait,
            "mean_post_release_travel": mean_post_release,
        },
        "planning": planning,
        "planner_diagnostics": planner_diagnostics,
        "stall": {
            "stalled_agent_count": int(np.sum(stalled)),
            "low_progress_mobile_count": int(np.sum(low_progress_mobile)),
        },
        "uncompleted": {
            "not_released": int(not_released),
            "not_center": int(not_center),
            "center_done_not_arrived": int(center_done_not_arrived),
        },
        "consistency": consistency,
    }

    return MetricsResult(
        metrics=metrics,
        agents=agents,
        violation_events=violation_events,
        boundary_contacts=boundary_contacts,
        min_distance_curve=min_distance_curve,
        arrival_curve=arrival_curve,
    )


def _cumulative_counts(grid: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """``#(event_time <= grid_j)``，逐网格点计数。"""
    times = np.sort(np.asarray(event_times, dtype=np.float64))
    times = times[np.isfinite(times)]
    if times.size == 0:
        return np.zeros(grid.size, dtype=np.int64)
    return np.searchsorted(times, grid, side="right").astype(np.int64)


# =============================================================================
# 落盘
# =============================================================================
def write_metrics(result_dir: Path, result: MetricsResult) -> None:
    """把 ``MetricsResult`` 写成 CONTRACTS §11 规定的 6 个产物。

    ``metrics.json`` 用标准 JSON ``null`` 表示缺失（不含 NaN/Infinity）；
    空事件表仍保留表头。
    """
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    payload = _json_safe(result.metrics)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")

    result.agents.to_csv(
        out_dir / "agents.csv", index=False, columns=list(AGENT_COLUMNS), lineterminator="\n"
    )
    result.violation_events.to_csv(
        out_dir / "violation_events.csv",
        index=False,
        columns=list(VIOLATION_EVENT_COLUMNS),
        lineterminator="\n",
    )
    result.boundary_contacts.to_csv(
        out_dir / "boundary_contacts.csv",
        index=False,
        columns=list(BOUNDARY_CONTACT_COLUMNS),
        lineterminator="\n",
    )
    result.min_distance_curve.to_csv(
        out_dir / "min_distance_curve.csv",
        index=False,
        columns=list(MIN_DISTANCE_CURVE_COLUMNS),
        lineterminator="\n",
    )
    result.arrival_curve.to_csv(
        out_dir / "arrival_curve.csv",
        index=False,
        columns=list(ARRIVAL_CURVE_COLUMNS),
        lineterminator="\n",
    )
