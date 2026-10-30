# OWNER: lead
"""同步更新契约与数值错误处理（handoff §4.4、§2.2、§8 同步性测试）。"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench import dynamics, navigation
from uavbench.config import apply_overrides, load_config
from uavbench.initialization import Scene, build_base_fibonacci_scene, build_scene
from uavbench.planners.base import PlanResult, empty_diagnostics
from uavbench.simulation import build_neighbor_lists, run_simulation


class CapturePlanner:
    """记录调用时快照的规划器，返回期望速度。"""

    def __init__(self):
        self.states = []
        self.name = "capture"

    def plan(self, state):
        self.states.append(
            {
                "t": state.t,
                "positions": state.positions,
                "velocities": state.velocities,
                "active_ids": state.active_ids.copy(),
                "frozen": state.frozen.copy(),
                "neighbor_ids": state.neighbor_ids,
                "neighbor_mask": state.neighbor_mask,
                "v_pref": state.v_pref.copy(),
            }
        )
        return PlanResult(velocities=state.v_pref.copy(), diagnostics=empty_diagnostics(state.n_active))


def test_constant_velocity_segments_and_frozen_zero():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    res = run_simulation(cfg, scene, "direct", "A")
    expected = res.velocities * cfg.experiment.dt
    assert np.allclose(np.diff(res.positions, axis=0), expected, rtol=0.0, atol=1e-14)
    # 冻结者的速度恒为零，且未被规划
    assert np.all(res.velocities[res.frozen[:-1]] == 0.0)
    assert np.all(np.linalg.norm(res.velocities[res.active], axis=1) > 0.0)


def test_initial_snapshot_has_zero_velocity_and_correct_state():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    cap = CapturePlanner()
    run_simulation(cfg, scene, "capture", "A", planner=cap)
    first = cap.states[0]
    assert first["t"] == 0.0
    assert np.all(first["velocities"] == 0.0)
    assert np.array_equal(first["positions"], scene.starts)
    assert np.array_equal(first["active_ids"], np.arange(100))
    # 快照位置必须等于上一网格位置（同步更新）
    second = cap.states[1]
    assert np.allclose(second["positions"], scene.starts + first["v_pref"] * cfg.experiment.dt)


def test_snapshot_is_read_only_and_shared():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    cap = CapturePlanner()
    run_simulation(cfg, scene, "capture", "B", planner=cap)
    for snap in (cap.states[0], cap.states[1]):
        assert not snap["positions"].flags.writeable
        assert not snap["velocities"].flags.writeable
        assert np.all(snap["neighbor_ids"] >= -1)


def test_neighbor_list_all_mode_includes_frozen_and_sorted():
    cfg = load_config()
    scene = build_scene(cfg, 1000)
    positions = scene.starts.copy()
    active = np.array([3, 50, 99], dtype=np.int64)
    ids, mask = build_neighbor_lists(cfg, positions, active)
    assert ids.shape == (3, 99)
    assert np.all(mask)
    for row, a in zip(ids, active):
        assert np.array_equal(np.sort(row), np.setdiff1d(np.arange(100), [a]))
        assert np.all(np.diff(row) > 0)


def test_neighbor_range_mode_masks_by_distance():
    cfg = apply_overrides(
        load_config(),
        {
            "simulation.neighbor_mode": "range",
            "simulation.neighbor_range": 22.0,
        },
    )
    scene = build_scene(cfg, 1000)
    active = np.arange(100, dtype=np.int64)
    ids, mask = build_neighbor_lists(cfg, scene.starts, active)
    for i in range(100):
        valid = ids[i][mask[i]]
        d = np.linalg.norm(scene.starts[valid] - scene.starts[i], axis=1)
        assert np.all(d <= 22.0 + 1e-12)
        # 掩码外不包含 22 m 内的邻居（主实验不截断）
        assert len(valid) == int(np.sum(np.linalg.norm(scene.starts - scene.starts[i], axis=1)[
            np.arange(100) != i
        ] <= 22.0))


def test_release_models_only_in_scenario_b():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    from uavbench.simulation import SimulationError

    with pytest.raises(SimulationError):
        run_simulation(cfg, scene, "orca_release", "A")


@pytest.mark.parametrize("mode", ["nan", "overspeed"])
def test_invalid_planner_output_marks_run_invalid(mode):
    class BadPlanner:
        name = "bad"

        def plan(self, state):
            n = state.n_active
            if mode == "nan":
                v = np.full((n, 3), np.nan)
            else:
                v = np.full((n, 3), 3.0)
            return PlanResult(velocities=v, diagnostics=empty_diagnostics(n))

    cfg = apply_overrides(load_config(), {"experiment.observation_time": 2.0})
    scene = build_base_fibonacci_scene(cfg)
    res = run_simulation(cfg, scene, "bad", "A", planner=BadPlanner())
    assert not res.valid
    assert res.numerical_error
    assert res.error_message is not None
    assert res.n_steps_executed == 0
    # 轨迹数组形状完整，但状态被标记为无效（不得伪装成超时）
    assert res.positions.shape == (cfg.n_steps + 1, 100, 3)
    assert np.all(res.velocities == 0.0)


def test_reproducibility_of_arrays():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    a = run_simulation(cfg, scene, "direct", "B")
    b = run_simulation(cfg, scene, "direct", "B")
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.velocities, b.velocities)
    assert np.array_equal(a.stages, b.stages)
    assert np.array_equal(a.arrival_times, b.arrival_times, equal_nan=True)


def test_path_length_matches_displacement():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    res = run_simulation(cfg, scene, "direct", "B")
    paths = dynamics.path_lengths(res.positions)
    assert np.allclose(paths, 39.8, rtol=0.0, atol=1e-9)
