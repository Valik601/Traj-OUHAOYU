"""M2 / 3D-RVO：有限时间三维互惠速度障碍 + 确定性候选搜索（handoff §5.2）。

定义（闭合预测窗 ``t in [0, tau]``，CONTRACTS §7）::

    p_ij = p_j - p_i
    RVO 危险 ⇔ 存在 t in [0, tau] 使 ||p_ij - t*(2c - v_i - v_j)|| < d_plan
    冻结邻机（等待/到达）: w = c（与 VO 完全一致，不做互惠推理）

注意：RVO 不是“只把 VO 顶点平移”的近似。经典 2D 直觉里的顶点平移等价于
检查 ``c - v_i - v_j``（或 ``c - v_j`` 的某种平移）是否落在 VO 内，而本实现严格
按 ``2c - v_i - v_j`` 判定整个几何（截锥）。实现上使用等价变换
``2c - v_i - v_j = 2*(c - (v_i+v_j)/2)``，令 ``u = 2t`` 得

    min_{t in [0,tau]} ||p_ij - t*(2c-v_i-v_j)||
      = min_{u in [0,2tau]} ||p_ij - u*(c - (v_i+v_j)/2)||,

测试逐候选核对原始定义，并构造出“仅移顶点”实现会给出不同结论的用例。

* 候选集合与 M1 完全共享（同一 :class:`CandidateSet` 对象、同一搜索/回退规则）；
* 诊断：``solver_infeasible`` 恒 False（RVO 是判定而非变分求解器），
  ``constraint_violation_max`` 为返回速度对 RVO 约束的最大残差。
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .base import Diagnostics, PlanResult, PlanState, validate_commands
from .candidates import CandidateSet, search_velocities

__all__ = ["RVO3DPlanner"]


class RVO3DPlanner:
    """M2：3D 有限时间 RVO（``2c-v_i-v_j`` 定义）+ 共享候选搜索。"""

    name = "rvo"

    def __init__(self, cfg: Config, candidate_set: CandidateSet | None = None) -> None:
        self.cfg = cfg
        self.v_max = float(cfg.simulation.v_max)
        self.d_plan = float(cfg.simulation.d_plan)
        self.tau = float(cfg.planners.rvo.tau)
        if candidate_set is None:
            candidate_set = CandidateSet.from_config(cfg)
        self.candidate_set = candidate_set

    def plan(self, state: PlanState) -> PlanResult:
        n_active = state.n_active
        if n_active == 0:
            zeros_bool = np.zeros(0, dtype=bool)
            diagnostics = Diagnostics(
                sample_exhausted=zeros_bool.copy(),
                used_fallback=zeros_bool.copy(),
                solver_infeasible=zeros_bool.copy(),
                constraint_violation_max=np.zeros(0, dtype=np.float64),
                candidate_count_tested=np.zeros(0, dtype=np.int32),
                initial_overlap_pairs=np.zeros(0, dtype=np.int32),
                numerical_error=zeros_bool.copy(),
                extra={
                    "preferred_speed": np.zeros(0, dtype=np.float64),
                    "fallback_ttc": np.zeros(0, dtype=np.float64),
                },
            )
            return PlanResult(
                velocities=np.zeros((0, 3), dtype=np.float64),
                diagnostics=diagnostics,
            )
        active_ids = state.active_ids
        result = search_velocities(
            self.candidate_set,
            model="rvo",
            v_max=self.v_max,
            tau=self.tau,
            d_plan=self.d_plan,
            v_current=np.asarray(state.velocities[active_ids], dtype=np.float64),
            v_pref=np.asarray(state.v_pref, dtype=np.float64),
            p_ij=state.pair_offsets(),
            v_j=state.neighbor_velocities(),
            frozen_j=state.neighbor_frozen(),
            neighbor_mask=state.neighbor_mask,
        )
        zeros_bool = np.zeros(n_active, dtype=bool)
        diagnostics = Diagnostics(
            sample_exhausted=result.sample_exhausted,
            used_fallback=result.used_fallback,
            solver_infeasible=zeros_bool.copy(),  # RVO 不是求解器，无“不可行”
            constraint_violation_max=result.constraint_violation_max,
            candidate_count_tested=result.candidate_count_tested,
            initial_overlap_pairs=result.initial_overlap_pairs,
            numerical_error=zeros_bool.copy(),
            extra={
                "preferred_speed": result.preferred_speed,
                "fallback_ttc": result.fallback_min_ttc,
            },
        )
        validate_commands(result.velocities, n_active, self.v_max)
        return PlanResult(velocities=result.velocities, diagnostics=diagnostics)
