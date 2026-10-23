"""Agent 状态与感知 / 几何接口。

设计要求：行人与车辆使用同一个 Agent 表征，只是参数不同：
    - 尺寸（length/width）、期望速度、加减速能力、松弛时间
    - 横向机动力 alpha ∈ {0, 1}：行人为 1（可侧移），车辆为 0（锁死横向）
    - 角色只影响运动模型中的参数与决策阈值，不影响状态更新方程本身
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .common import lateral_clearance, longitudinal_body_gap


@dataclass
class Agent:
    """统一的移动主体状态。"""

    name: str
    role: str                 # "ped" | "car"
    x: float
    y: float
    vx: float
    vy: float
    length: float
    width: float
    v0: float                 # 期望速度（沿自身行驶方向）
    # 运动能力（由 config 注入）
    tau: float
    a_max: float
    b_comfort: float
    b_max: float
    alpha_lat: float          # 1 = 可侧移, 0 = 不可侧移
    v_lat_max: float = 0.0
    a_lat_max: float = 0.0
    tau_lat: float = 0.3
    # 决策状态
    side_sign: float = 1.0    # 初始位置所在一侧（+1 / -1）
    mode: str = "FREE"
    v_target: float = 0.0
    y_target: float = 0.0
    has_stepped: bool = False
    # 本步结果
    a_long: float = 0.0
    a_world: tuple[float, float] = (0.0, 0.0)
    # 统计
    stopped_time: float = 0.0
    finish_time: float | None = None
    started_x: float = 0.0

    # ---- 派生量 -----------------------------------------------------------
    @property
    def heading(self) -> float:
        """行驶方向单位向量在 x 上的分量。"""
        return 1.0 if self.role == "ped" else -1.0

    @property
    def long_speed(self) -> float:
        """沿自身行驶方向的速度（>=0 表示前进）。"""
        return self.heading * self.vx

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def radius(self) -> float:
        """外接圆半径（行人 0.25 m；车辆的 SFM 近似另用三个圆）。"""
        return 0.5 * self.width

    # ---- 与其它 Agent 的几何 ---------------------------------------------
    def gap_to(self, other: "Agent") -> float:
        """沿自身行驶方向到 other 的身体净距（正值 = other 在前方）。"""
        return longitudinal_body_gap(self.x, other.x, self.length, other.length, self.heading)

    def clearance_to(self, other: "Agent") -> float:
        """横向身体间隙（负值 = 横向投影重叠）。"""
        return lateral_clearance(self.y, other.y, self.width, other.width)

    def closing_speed_to(self, other: "Agent") -> float:
        """沿自身行驶方向的接近速度（正值 = 正在靠近）。"""
        return self.heading * (self.vx - other.vx)


@dataclass
class World:
    """一次仿真的世界状态。"""

    width: float
    x_min: float
    x_max: float
    car: Agent
    peds: list[Agent] = field(default_factory=list)
    t: float = 0.0

    @property
    def agents(self) -> list[Agent]:
        return [self.car, *self.peds]

    @property
    def y_limit(self) -> float:
        """行人圆心允许的最大 |y|（身体不越出路缘）。"""
        return 0.5 * self.width - 0.5 * self.peds[0].width

    def in_path(self, agent: Agent, other: Agent, buffer_m: float = 0.0) -> bool:
        """other 是否位于 agent 的纵向行驶路径内（横向身体重叠）。

        buffer_m > 0 时要求更宽的横向重叠，用于安全投影。
        """
        return agent.clearance_to(other) < -buffer_m

    def nearest_leader(self, agent: Agent) -> Agent | None:
        """行驶路径内最近的纵向障碍（用于 Wiedemann 模型）。"""
        best: Agent | None = None
        best_gap = math.inf
        for other in self.agents:
            if other is agent:
                continue
            if agent.clearance_to(other) > 0.0:      # 横向已分离，可平行通过
                continue
            gap = agent.gap_to(other)
            if gap < -1.0:                            # 已在身后
                continue
            if gap < best_gap:
                best, best_gap = other, gap
        return best
