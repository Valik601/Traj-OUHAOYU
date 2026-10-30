"""恒速段动力学：每步命令速度在整段 ``[t_k, t_{k+1}]`` 保持恒定（handoff §4.4）。

主实验允许瞬时改变速度方向与大小；不考虑加速度、姿态、传感与通信延迟。
"""

from __future__ import annotations

import numpy as np

from . import geometry


def advance_positions(
    positions: np.ndarray, velocities: np.ndarray, dt: float
) -> np.ndarray:
    """``p_next = p + v*dt``；``velocities`` 中冻结者必须已为零。"""
    p = np.asarray(positions, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    if p.shape != v.shape:
        raise ValueError(f"positions {p.shape} and velocities {v.shape} shape mismatch")
    return p + v * float(dt)


def command_velocities(
    n_agents: int,
    active_ids: np.ndarray,
    planned: np.ndarray,
    *,
    dt: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    frozen: np.ndarray,
    v_max: float,
    tol: float = geometry.VEL_EPS,
) -> np.ndarray:
    """把规划器输出写入全体命令速度（冻结者为零），并校验有限性与速度上限。"""
    n = int(n_agents)
    full = np.zeros((n, 3), dtype=np.float64)
    act = np.asarray(active_ids, dtype=np.int64)
    planned = np.asarray(planned, dtype=np.float64)
    if planned.shape != (act.size, 3):
        raise ValueError(f"planned velocities shape {planned.shape} != ({act.size}, 3)")
    if not np.all(np.isfinite(planned)):
        raise FloatingPointError("planner returned non-finite velocity")
    norms = np.sqrt(geometry.sq_norm(planned))
    if np.any(norms > float(v_max) + tol):
        raise ValueError(
            f"planner exceeded v_max ({v_max}): max {float(np.max(norms)) if norms.size else 0.0}"
        )
    full[act] = planned
    if np.any(frozen) and np.any(np.abs(full[np.asarray(frozen, dtype=bool)]) > 0.0):
        raise ValueError("frozen agents must have zero command velocity")
    return full


def path_lengths(positions: np.ndarray) -> np.ndarray:
    """累计实际路程（逐机），``positions`` 形状 ``(K+1,N,3)``。"""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape[0] < 2:
        return np.zeros(pos.shape[1], dtype=np.float64)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=2)
    return np.sum(steps, axis=0)


def segment_velocities_ok(
    velocities: np.ndarray, v_max: float, *, tol: float = geometry.VEL_EPS
) -> bool:
    v = np.asarray(velocities, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return False
    return bool(np.all(np.sqrt(geometry.sq_norm(v)) <= float(v_max) + tol))


def all_finite(*arrays: np.ndarray) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(a)))) for a in arrays)
