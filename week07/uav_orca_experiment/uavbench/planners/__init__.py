"""规划器注册与构造。

``build_planner`` 按基础模型名构造规划器；释放类模型（orca_release/direct_release）
复用相同规划器，放行由仿真器调度层处理（CONTRACTS.md §9、§10）。
"""

from __future__ import annotations

from ..config import MODEL_BASE, Config
from .base import Diagnostics, PlanResult, PlanState, Planner, empty_diagnostics
from .direct import DirectPlanner

__all__ = [
    "DirectPlanner",
    "Diagnostics",
    "PlanResult",
    "PlanState",
    "Planner",
    "build_planner",
    "empty_diagnostics",
]


def build_planner(model: str, cfg: Config, candidate_set=None) -> Planner:
    """按模型名构造规划器；``model`` 可为协议模型名或基础模型名。"""
    base = MODEL_BASE.get(model)
    if base is None:
        raise ValueError(f"unknown model: {model}")
    if base == "direct":
        return DirectPlanner(cfg, candidate_set)
    if base == "vo":
        from .vo3d import VO3DPlanner

        return VO3DPlanner(cfg, candidate_set)
    if base == "rvo":
        from .rvo3d import RVO3DPlanner

        return RVO3DPlanner(cfg, candidate_set)
    if base == "orca":
        from .orca3d import ORCA3DPlanner

        return ORCA3DPlanner(cfg, candidate_set)
    raise ValueError(f"unknown base planner for model {model!r}")
