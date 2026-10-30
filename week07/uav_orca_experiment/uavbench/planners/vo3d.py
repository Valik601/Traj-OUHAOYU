"""M1 / 3D-VO：有限时间三维速度障碍 + 确定性候选搜索（handoff §5.1/§5.2）。

定义（闭合预测窗 ``t in [0, tau]``，CONTRACTS §7）::

    p_ij = p_j - p_i
    VO  危险 ⇔ 存在 t in [0, tau] 使 ||p_ij - t*(c - v_j)|| < d_plan
    冻结邻机（等待/到达）: w = c（与 RVO 完全一致，不做互惠推理）

* 候选集合由 :class:`~uavbench.planners.candidates.CandidateSet` 提供，M1/M2
  必须共享同一场景级对象；
* 选择规则：可接受候选中最小化 ``||c-v_pref||^2`` → ``||c-v_i||^2`` → 固定索引；
* 无可接受候选 → ``sample_exhausted``/``used_fallback``，按物理预测 TTC 排序；
* 诊断字段见 CONTRACTS §6：``solver_infeasible`` 恒 False（VO 是判定而非求解器），
  ``constraint_violation_max`` 为返回速度对本模型全部邻机约束的最大残差。

本规划器只读快照、不修改状态、无随机性；同一输入两次调用输出逐位一致。
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .base import Diagnostics, PlanResult, PlanState, validate_commands
from .candidates import CandidateSet, search_velocities

__all__ = ["VO3DPlanner"]


class VO3DPlanner:
    """M1：3D 有限时间 VO + 共享候选搜索（非互惠，冻结邻机按静止体处理）。"""

    name = "vo"

    def __init__(self, cfg: Config, candidate_set: CandidateSet | None = None) -> None:
        self.cfg = cfg
        self.v_max = float(cfg.simulation.v_max)
        self.d_plan = float(cfg.simulation.d_plan)
        self.tau = float(cfg.planners.vo.tau)
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
            model="vo",
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
            solver_infeasible=zeros_bool.copy(),  # VO 是判定而非求解器，无“不可行”
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
