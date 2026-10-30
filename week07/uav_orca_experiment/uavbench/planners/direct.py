"""M0 / Direct：直接跟踪期望速度（解析校验与不避碰下限对照）。"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .base import Diagnostics, PlanResult, PlanState, empty_diagnostics


class DirectPlanner:
    """命令速度 = 期望速度；不做任何避碰。"""

    name = "direct"

    def __init__(self, cfg: Config, candidate_set=None):  # noqa: ARG002 - 统一构造签名
        self.cfg = cfg

    def plan(self, state: PlanState) -> PlanResult:
        a = state.n_active
        velocities = np.array(state.v_pref, dtype=np.float64, copy=True)
        return PlanResult(velocities=velocities, diagnostics=empty_diagnostics(a))
