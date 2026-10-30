"""分时放行（handoff §5.5）：ORCA + Staggered Release 与 Direct + Release 共用同一张表。

- 用 ``schedule_rng`` 对固定 ID 做随机排列（主实验随机流 0）；C1 完全复用 M4 的表。
- 计划放行时刻按绝对时刻量化到控制网格：``release_tick = ceil(planned/dt)``，
  用 ``Decimal`` 精确运算避免浮点边界误差；``release_actual = tick*dt``。
- Delta=0.75 s、dt=0.1 s 时实际间隔为 0.7 或 0.8 s；最后一个计划 74.25 s → 实际 74.3 s。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, localcontext

import numpy as np

from .config import Config
from .initialization import Scene
from .rng import STREAM_SCHEDULE_0, STREAM_SCHEDULE_1, make_rng

ORDER_FIXED_ID = "fixed_id"
ORDER_RANDOM_0 = "random_stream_0"
ORDER_RANDOM_1 = "random_stream_1"
VALID_ORDERS = (ORDER_FIXED_ID, ORDER_RANDOM_0, ORDER_RANDOM_1)


@dataclass(frozen=True)
class ReleaseTable:
    """每机放行计划与实际时刻（秒，从共同 t=0 起算）。"""

    ranks: np.ndarray  # (N,) int64，0 基
    planned: np.ndarray  # (N,) float64
    release_tick: np.ndarray  # (N,) int64
    actual: np.ndarray  # (N,) float64
    order: str
    delta: float

    @property
    def applied(self) -> bool:
        return self.order != ORDER_FIXED_ID or bool(np.any(self.actual > 0.0))


def _ceil_tick(planned: Decimal, dt: Decimal) -> int:
    with localcontext() as ctx:
        ctx.prec = 50
        q = planned / dt
        return int(q.to_integral_value(rounding=ROUND_CEILING))


def build_release_table(cfg: Config, scene: Scene) -> ReleaseTable:
    """构造放行表；同一场景（相同 ID 序列）M4/C1 得到完全相同的表。"""
    sim = cfg.simulation
    n = scene.n_agents
    order = cfg.scheduling.order
    if order not in VALID_ORDERS:
        raise ValueError(f"unknown release order: {order}")
    ids = np.asarray(scene.agent_ids, dtype=np.int64)
    if order == ORDER_FIXED_ID:
        ranks = np.empty(n, dtype=np.int64)
        ranks[ids] = np.arange(n, dtype=np.int64)
    else:
        stream = STREAM_SCHEDULE_0 if order == ORDER_RANDOM_0 else STREAM_SCHEDULE_1
        rng = make_rng(int(scene.seed), stream)
        perm = rng.permutation(n).astype(np.int64)  # perm[k] = 第 k 个放行的机
        ranks = np.empty(n, dtype=np.int64)
        ranks[perm] = np.arange(n, dtype=np.int64)

    delta_dec = Decimal(str(float(cfg.scheduling.delta)))
    dt_dec = Decimal(str(float(cfg.experiment.dt)))
    planned = np.asarray([float(ranks[i]) * float(cfg.scheduling.delta) for i in range(n)])
    ticks = np.empty(n, dtype=np.int64)
    for i in range(n):
        ticks[i] = _ceil_tick(Decimal(int(ranks[i])) * delta_dec, dt_dec)
    actual = np.asarray([float(Decimal(int(ticks[i])) * dt_dec) for i in range(n)])
    return ReleaseTable(
        ranks=ranks,
        planned=planned,
        release_tick=ticks,
        actual=actual,
        order=order,
        delta=float(cfg.scheduling.delta),
    )


def release_mask(table: ReleaseTable, t: float, *, eps: float = 1e-12) -> np.ndarray:
    """时刻 ``t`` 已放行的机（``actual <= t``）。"""
    return np.asarray(table.actual <= float(t) + eps, dtype=bool)
