"""规划器接口契约（冻结版，见 CONTRACTS.md §6）。

规划器只读快照；不得修改位置、阶段、累计指标或时间。所有命令速度必须有限且
``||v|| <= v_max + VEL_EPS``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .. import geometry
from ..config import Config

__all__ = [
    "PlanState",
    "Diagnostics",
    "PlanResult",
    "Planner",
    "empty_diagnostics",
    "validate_commands",
]


@dataclass(frozen=True)
class PlanState:
    """单步全体快照（同步更新：所有规划器看到同一快照）。"""

    t: float
    dt: float
    positions: np.ndarray  # (N,3)
    velocities: np.ndarray  # (N,3)
    stages: np.ndarray  # (N,) int8
    frozen: np.ndarray  # (N,) bool
    active_ids: np.ndarray  # (A,) int64 升序
    waypoints: np.ndarray  # (A,3)
    v_pref: np.ndarray  # (A,3)
    neighbor_ids: np.ndarray  # (A,M) int64，无效位 -1
    neighbor_mask: np.ndarray  # (A,M) bool

    # ---- 派生视图（规划器常用） -------------------------------------------
    @property
    def n_agents(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_active(self) -> int:
        return int(self.active_ids.shape[0])

    def current_velocities(self) -> np.ndarray:
        """活动机当前速度 ``(A,3)``。"""
        return self.velocities[self.active_ids]

    def neighbor_positions(self) -> np.ndarray:
        """邻机位置 ``(A,M,3)``；无效位为 0（配合 ``neighbor_mask``）。"""
        safe_ids = np.where(self.neighbor_mask, self.neighbor_ids, 0)
        out = self.positions[safe_ids]
        out = np.where(self.neighbor_mask[..., None], out, 0.0)
        return out

    def neighbor_velocities(self) -> np.ndarray:
        safe_ids = np.where(self.neighbor_mask, self.neighbor_ids, 0)
        out = self.velocities[safe_ids]
        return np.where(self.neighbor_mask[..., None], out, 0.0)

    def neighbor_frozen(self) -> np.ndarray:
        safe_ids = np.where(self.neighbor_mask, self.neighbor_ids, 0)
        return np.where(self.neighbor_mask, self.frozen[safe_ids], False)

    def pair_offsets(self) -> np.ndarray:
        """``p_ij = p_j - p_i``，形状 ``(A,M,3)``（无效位为 0）。"""
        occ = self.positions[self.active_ids][:, None, :]
        return self.neighbor_positions() - occ

    def neighbor_distances(self) -> np.ndarray:
        """``||p_ij||``，无效位为 +inf。"""
        d = np.sqrt(geometry.sq_norm(self.pair_offsets()))
        return np.where(self.neighbor_mask, d, np.inf)


@dataclass
class Diagnostics:
    """每活动机每步诊断（形状均为 ``(A,)``）。"""

    sample_exhausted: np.ndarray
    used_fallback: np.ndarray
    solver_infeasible: np.ndarray
    constraint_violation_max: np.ndarray
    candidate_count_tested: np.ndarray
    initial_overlap_pairs: np.ndarray
    numerical_error: np.ndarray
    extra: dict[str, np.ndarray] | None = field(default=None)

    def __post_init__(self) -> None:
        n = int(np.asarray(self.sample_exhausted).shape[0])
        expected = {
            "sample_exhausted": (n,),
            "used_fallback": (n,),
            "solver_infeasible": (n,),
            "constraint_violation_max": (n,),
            "candidate_count_tested": (n,),
            "initial_overlap_pairs": (n,),
            "numerical_error": (n,),
        }
        for name, shape in expected.items():
            arr = np.asarray(getattr(self, name))
            if arr.shape != shape:
                raise ValueError(f"diagnostic {name} has shape {arr.shape}, expected {shape}")
        if self.extra is not None:
            for name, arr in self.extra.items():
                if np.asarray(arr).shape != (n,):
                    raise ValueError(f"extra diagnostic {name} must have shape ({n},)")


def empty_diagnostics(n_active: int) -> Diagnostics:
    """全部取零/False 的诊断（Direct 用）。"""
    n = int(n_active)
    return Diagnostics(
        sample_exhausted=np.zeros(n, dtype=bool),
        used_fallback=np.zeros(n, dtype=bool),
        solver_infeasible=np.zeros(n, dtype=bool),
        constraint_violation_max=np.zeros(n, dtype=np.float64),
        candidate_count_tested=np.zeros(n, dtype=np.int32),
        initial_overlap_pairs=np.zeros(n, dtype=np.int32),
        numerical_error=np.zeros(n, dtype=bool),
        extra=None,
    )


@dataclass
class PlanResult:
    velocities: np.ndarray  # (A,3)
    diagnostics: Diagnostics


@runtime_checkable
class Planner(Protocol):
    name: str

    def plan(self, state: PlanState) -> PlanResult:  # pragma: no cover - 协议
        ...


def validate_commands(
    velocities: np.ndarray, n_active: int, v_max: float, *, tol: float = geometry.VEL_EPS
) -> None:
    """校验规划输出：有限、形状正确、速度上限（含容差）。"""
    v = np.asarray(velocities, dtype=np.float64)
    if v.shape != (int(n_active), 3):
        raise ValueError(f"velocities shape {v.shape} != ({n_active}, 3)")
    if not np.all(np.isfinite(v)):
        raise FloatingPointError("planner returned non-finite velocity")
    norms = np.sqrt(geometry.sq_norm(v))
    if np.any(norms > v_max + tol):
        raise ValueError(
            f"planner exceeded v_max: max={float(np.max(norms)):.6f} > {v_max}+{tol:g}"
        )
