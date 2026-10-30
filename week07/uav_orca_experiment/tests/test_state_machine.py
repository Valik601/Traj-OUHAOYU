# OWNER: lead
"""状态机与时间推进验收（handoff §4、§8 状态机/同步性测试）。"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench import navigation
from uavbench.config import apply_overrides, load_config
from uavbench.initialization import Scene, build_base_fibonacci_scene
from uavbench.planners.base import PlanResult, empty_diagnostics
from uavbench.simulation import run_simulation


class StubPlanner:
    """测试用规划器：按固定模式返回预设速度（不参与正式实验）。"""

    def __init__(self, mode: str = "waypoint_constant"):
        self.mode = mode
        self.name = "stub"

    def plan(self, state):
        a = state.n_active
        if self.mode == "zero":
            v = np.zeros((a, 3))
        elif self.mode == "nan":
            v = np.full((a, 3), np.nan)
        elif self.mode == "overspeed":
            v = np.full((a, 3), 3.0)
        elif self.mode == "waypoint_constant":
            v = np.zeros((a, 3))
            for k in range(a):
                d = state.waypoints[k] - state.positions[state.active_ids[k]]
                n = float(np.linalg.norm(d))
                if n > 0:
                    v[k] = 2.0 * d / n
        else:  # pragma: no cover
            raise ValueError(self.mode)
        return PlanResult(velocities=v, diagnostics=empty_diagnostics(a))


def _two_agent_cfg(overrides=None):
    base = {
        "simulation.n_agents": 2,
        "experiment.observation_time": 2.0,
    }
    if overrides:
        base.update(overrides)
    return apply_overrides(load_config(), base)


def _two_agent_scene(starts, goals):
    return Scene(
        scene_id="unit2",
        seed=1000,
        starts=np.asarray(starts, dtype=np.float64),
        goals=np.asarray(goals, dtype=np.float64),
        agent_ids=np.arange(2, dtype=np.int64),
        meta={"mode": "unit_test"},
    )


def test_midstep_center_crossing_is_recorded_once():
    """步内扫过中心容差球必须记录真实事件时刻，并在下一步切换控制阶段。"""
    cfg = _two_agent_cfg()
    scene = _two_agent_scene(
        [[0.15, 0.0, 0.0], [100.0, 0.0, 0.0]],
        [[-10.0, 0.0, 0.0], [-100.0, 0.0, 0.0]],
    )
    res = run_simulation(cfg, scene, "stub", "B", planner=StubPlanner("waypoint_constant"))
    assert res.valid
    # 起点 0.15 m，恒定 2 m/s 走 0.2 m → 落在 -0.05 m；穿过 0.1 m 球面于 0.25*dt = 0.025 s
    assert res.center_entry_times[0] == pytest.approx(0.025, abs=1e-12)
    assert res.stages[0, 0] == navigation.STAGE_TO_CENTER
    assert res.stages[1, 0] == navigation.STAGE_TO_GOAL
    assert res.center_entry_steps[0] == 0
    # 中心子任务一旦完成不得重置
    assert np.all(res.stages[1:, 0] != navigation.STAGE_TO_CENTER)


def test_b_cannot_arrive_without_center_subtask():
    stages_next = np.array([navigation.STAGE_TO_GOAL, navigation.STAGE_TO_GOAL])
    goals = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    positions = goals.copy()
    center_done = np.array([False, True])
    mask_b = navigation.arrival_mask(stages_next, "B", center_done, positions, goals, 0.3)
    assert not mask_b[0]
    assert mask_b[1]
    mask_a = navigation.arrival_mask(stages_next, "A", center_done, positions, goals, 0.3)
    assert np.all(mask_a)


def test_arrival_requires_step_end_inside_goal_sphere():
    """步内扫过目标球但步末在球外不算到达。"""
    cfg = _two_agent_cfg({"simulation.goal_tolerance": 1e-8, "experiment.observation_time": 4.0})
    # A 场景：从 10 m 处以恒定 2 m/s 飞向位于 0.05 m 的目标，网格步长 0.2 m，
    # 永远不会在网格末落在 1e-8 容差球内（步内会扫过目标点）。
    scene = _two_agent_scene(
        [[10.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        [[0.05, 0.0, 0.0], [40.0, 0.0, 0.0]],
    )
    res = run_simulation(cfg, scene, "stub", "A", planner=StubPlanner("waypoint_constant"))
    assert res.valid
    assert np.all(np.isnan(res.arrival_times))
    assert np.all(res.stages[-1] == navigation.STAGE_TO_GOAL)


def test_goal_tolerance_boundary():
    goals = np.array([[0.0, 0.0, 0.0]])
    stages = np.array([navigation.STAGE_TO_GOAL])
    center = np.array([True])
    inside = np.array([[0.29, 0.0, 0.0]])
    outside = np.array([[0.31, 0.0, 0.0]])
    assert navigation.arrival_mask(stages, "A", center, inside, goals, 0.3)[0]
    assert not navigation.arrival_mask(stages, "A", center, outside, goals, 0.3)[0]
    wrong_stage = np.array([navigation.STAGE_TO_CENTER])
    assert not navigation.arrival_mask(wrong_stage, "A", center, inside, goals, 0.3)[0]


def test_waiting_agents_do_not_move_and_stay_in_safety_set():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    res = run_simulation(cfg, scene, "direct_release", "B")
    assert res.valid
    # 每个等待者在其实际放行时刻之前位置不变、冻结、阶段为 WAITING
    for i in range(cfg.simulation.n_agents):
        release = res.release_actual[i]
        before = int(round(release / cfg.experiment.dt))
        if before > 0:
            assert res.stages[0, i] == navigation.STAGE_WAITING
            assert res.frozen[0, i]
            assert np.all(res.positions[: before + 1, i] == res.positions[0, i])
            assert np.all(res.stages[:before, i] == navigation.STAGE_WAITING)
        assert res.stages[before, i] == navigation.STAGE_TO_CENTER
        if before == 0:
            assert not res.frozen[0, i]
    # 全体最终到达（Direct + Release 在中心瓶颈中会有大量侵犯，但任务完成性由几何决定）
    assert np.all(np.isfinite(res.arrival_times))


def test_single_agent_zero_velocity_never_arrives():
    cfg = _two_agent_cfg()
    scene = _two_agent_scene(
        [[5.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        [[-5.0, 0.0, 0.0], [-50.0, 0.0, 0.0]],
    )
    res = run_simulation(cfg, scene, "stub", "B", planner=StubPlanner("zero"))
    assert res.valid
    assert np.all(np.isnan(res.arrival_times))
    assert np.all(np.isnan(res.center_entry_times))
    assert np.all(res.stages[-1] == navigation.STAGE_TO_CENTER)
