# OWNER: metrics-dev
"""连续安全检测单元测试（handoff §6.1 / §8、CONTRACTS §11.1）。

覆盖：端点安全但步内侵犯（handoff §8 构造例）、零相对速度、相切接触不构成侵犯、
起始重叠、跨步区间合并、``d_eval`` 与 ``d_safe`` 两侧一致性、曲线与 ``geometry``
标量原语一致、重复侵犯区间不重复计数、大 dt 纯几何穿越、中心进入几何一致性。

测试不依赖 ``simulation.py``：全部轨迹在测试内直接构造。
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench import geometry
from uavbench.config import Config, load_config
from uavbench.metrics import compute_metrics
from uavbench.navigation import (
    STAGE_ARRIVED,
    STAGE_TO_GOAL,
    STAGE_WAITING,
)

DT = 0.1
OBS = 120.0


# =============================================================================
# 测试内构造工具（与另两个 metrics 测试文件同构，避免跨测试文件 import 依赖）
# =============================================================================
class DuckScene:
    """鸭子类型场景：指标只读取 goals/agent_ids/scene_id/seed。"""

    def __init__(self, starts, goals, agent_ids=None, scene_id="unit", seed=0):
        self.starts = np.asarray(starts, dtype=np.float64)
        self.goals = np.asarray(goals, dtype=np.float64)
        n = self.starts.shape[0]
        self.agent_ids = (
            np.arange(n, dtype=np.int64)
            if agent_ids is None
            else np.asarray(agent_ids, dtype=np.int64)
        )
        self.scene_id = scene_id
        self.seed = seed


def make_cfg(dt: float = DT, observation_time: float = OBS, **overrides) -> Config:
    data = load_config().to_json_dict()
    data["experiment"]["dt"] = float(dt)
    data["experiment"]["observation_time"] = float(observation_time)
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return Config.from_json_dict(data)


def build_traj(positions, velocities, stages, *, dt: float = DT, observation_time: float = OBS, **extras):
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    stages = np.asarray(stages, dtype=np.int8)
    n_steps = positions.shape[0] - 1
    n_agents = positions.shape[1]
    t = np.arange(n_steps + 1, dtype=np.float64) * float(dt)
    traj = {
        "t": t,
        "positions": positions,
        "velocities": velocities,
        "stages": stages,
        "frozen": np.isin(stages, (STAGE_WAITING, STAGE_ARRIVED)),
        "active": ~np.isin(stages[:-1], (STAGE_WAITING, STAGE_ARRIVED)),
        "dt": float(dt),
        "n_agents": int(n_agents),
        "observation_time": float(observation_time),
    }
    traj.update(extras)
    return traj


def compute(traj, scene, *, scenario="A", cfg=None, center_times=None, **kwargs):
    cfg = make_cfg() if cfg is None else cfg
    n_agents = traj["positions"].shape[1]
    if center_times is None:
        center_times = np.full(n_agents, np.nan)
    return compute_metrics(
        traj,
        scene,
        cfg,
        model="direct",
        scenario=scenario,
        seed=0,
        scene_hash="scene-hash",
        config_hash="config-hash",
        center_entry_times=center_times,
        **kwargs,
    )


def two_agent_pair(rel0, rel_vel, n_steps: int = 1, dt: float = DT, goals=None):
    """两机相对轨迹：``r(t) = rel0 + t*rel_vel``，i 取 +w/2、j 取 -w/2。"""
    rel0 = np.asarray(rel0, dtype=np.float64)
    w = np.asarray(rel_vel, dtype=np.float64)
    rel = rel0[None, :] + np.arange(n_steps + 1)[:, None] * (dt * w)[None, :]
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 0, :] = 0.5 * rel
    positions[:, 1, :] = -0.5 * rel
    velocities = np.broadcast_to(
        np.stack([0.5 * w, -0.5 * w])[None, :, :], (n_steps, 2, 3)
    ).copy()
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    if goals is None:
        goals = np.array([[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]])
    return build_traj(positions, velocities, stages, dt=dt), DuckScene(positions[0], goals)


# =============================================================================
# 1. handoff §8 的端点安全、步内侵犯构造例
# =============================================================================
def test_handoff_midstep_violation_with_safe_endpoints():
    """dt=0.1 s，相对轨迹 (-0.2,0.99,0) -> (0.2,0.99,0)：端点 1.01 m，步内最小 0.99 m。"""
    traj, scene = two_agent_pair((-0.2, 0.99, 0.0), (4.0, 0.0, 0.0), n_steps=1)
    cfg = make_cfg()
    res = compute(traj, scene, cfg=cfg)

    # 端点安全：两个网格端点距离都 > d_safe
    endpoints = np.linalg.norm(
        traj["positions"][:, 0, :] - traj["positions"][:, 1, :], axis=-1
    )
    assert np.all(endpoints > cfg.simulation.d_safe)
    assert endpoints[0] == pytest.approx(1.01, rel=1e-12)
    assert endpoints[1] == pytest.approx(1.01, rel=1e-12)

    # 恰好一个合并事件，进入/退出时刻由 d_eval 求根解析给出
    assert res.metrics["violation_event_count"] == 1
    assert res.metrics["violating_pair_count"] == 1
    d_eval = cfg.d_eval
    assert d_eval == pytest.approx(1.0 - 1e-8, rel=0, abs=0)
    half = float(np.sqrt(d_eval**2 - 0.99**2))
    expected_start = (0.2 - half) / 4.0
    expected_end = (0.2 + half) / 4.0
    ev = res.violation_events.iloc[0]
    assert ev["start_time"] == pytest.approx(expected_start, abs=1e-9)
    assert ev["end_time"] == pytest.approx(expected_end, abs=1e-9)
    assert ev["duration"] == pytest.approx(expected_end - expected_start, abs=1e-9)
    assert ev["min_distance"] == pytest.approx(0.99, rel=1e-12)
    assert ev["min_distance_time"] == pytest.approx(0.05, abs=1e-9)
    assert ev["agent_i"] == 0 and ev["agent_j"] == 1

    # 全局指标：首侵犯时间 = 区间下确界；全程最小距离 = 0.99
    assert res.metrics["first_violation_time"] == pytest.approx(expected_start, abs=1e-9)
    assert res.metrics["min_separation"] == pytest.approx(0.99, rel=1e-12)
    assert res.metrics["min_separation_time"] == pytest.approx(0.05, abs=1e-9)
    assert res.metrics["violation_pair_seconds"] == pytest.approx(
        expected_end - expected_start, abs=1e-9
    )

    # 区间最小距离曲线同样捕捉步内最小值（不是端点距离）
    row = res.min_distance_curve.iloc[0]
    assert row["min_distance"] == pytest.approx(0.99, rel=1e-12)
    assert row["violating_pair_count"] == 1
    assert row["boundary_pair_count"] == 0
    assert res.boundary_contacts.empty
    assert res.agents["violated"].tolist() == [True, True]


# =============================================================================
# 2. 零相对速度
# =============================================================================
def test_zero_relative_velocity_overlap_merges_across_steps():
    """静止且中心距 0.5 m < d_eval：整段侵犯，跨 4 步合并为 1 个事件。"""
    n_steps = 4
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = 0.5
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[20.0, 0, 0], [-20.0, 0, 0]]))
    res = compute(traj, scene)

    assert res.metrics["violation_event_count"] == 1
    ev = res.violation_events.iloc[0]
    assert ev["start_time"] == pytest.approx(0.0, abs=1e-12)
    assert ev["end_time"] == pytest.approx(0.4, abs=1e-12)
    assert ev["duration"] == pytest.approx(0.4, abs=1e-12)
    assert ev["min_distance"] == pytest.approx(0.5, rel=1e-12)
    assert ev["min_distance_time"] == pytest.approx(0.0, abs=1e-12)
    assert res.metrics["first_violation_time"] == pytest.approx(0.0, abs=1e-12)
    assert res.metrics["violation_pair_seconds"] == pytest.approx(0.4, abs=1e-12)
    assert res.min_distance_curve["violating_pair_count"].tolist() == [1, 1, 1, 1]


def test_zero_relative_velocity_safe_gap():
    """静止且中心距 1.5 m：无侵犯、无边界接触，曲线恒为 1.5。"""
    n_steps = 3
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = 1.5
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[20.0, 0, 0], [-20.0, 0, 0]]))
    res = compute(traj, scene)

    assert res.metrics["violation_event_count"] == 0
    assert res.metrics["violating_pair_count"] == 0
    assert res.metrics["first_violation_time"] is None
    assert res.metrics["min_separation"] == pytest.approx(1.5, rel=1e-12)
    assert res.boundary_contacts.empty
    assert np.allclose(res.min_distance_curve["min_distance"].to_numpy(), 1.5)


# =============================================================================
# 3. 相切是接触、不是严格侵犯
# =============================================================================
def test_tangency_static_contact_not_violation():
    """静止中心距恰好 1.0 m：全程最小距离在边界带内，但不是严格侵犯。"""
    cfg = make_cfg()
    assert cfg.d_eval < cfg.simulation.d_safe
    n_steps = 2
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = 1.0
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_ARRIVED, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[20.0, 0, 0], [-20.0, 0, 0]]))
    res = compute(traj, scene, cfg=cfg)

    assert res.metrics["violating_pair_count"] == 0
    assert res.metrics["violation_event_count"] == 0
    assert res.metrics["min_separation"] == pytest.approx(1.0, rel=1e-12)
    # 边界接触记录（不进入 violation_events）
    assert len(res.boundary_contacts) == 1
    contact = res.boundary_contacts.iloc[0]
    assert contact["min_distance"] == pytest.approx(1.0, rel=1e-12)
    assert contact["min_distance_time"] == pytest.approx(0.0, abs=1e-12)
    # 全程都在带内（距离恒 1.0）：带内时长 = 总时长 0.2 s
    assert contact["min_time_inside_band"] == pytest.approx(0.2, abs=1e-9)
    # 已完成且从未参与侵犯 → 安全完成不受“接触”影响
    assert res.agents["safe_completed"].tolist() == [True, True]


def test_tangency_moving_contact_not_violation():
    """相对轨迹在步内与 d_safe 相切一次：最小距离恰为 1.0 m，仍不算侵犯。"""
    # r(t) = (-0.2 + 4t, 1.0, 0)，在 t=0.05 处 |r| = 1.0（相切）
    traj, scene = two_agent_pair((-0.2, 1.0, 0.0), (4.0, 0.0, 0.0), n_steps=1)
    res = compute(traj, scene)

    assert res.metrics["violating_pair_count"] == 0
    assert res.metrics["min_separation"] == pytest.approx(1.0, rel=1e-12)
    assert res.metrics["min_separation_time"] == pytest.approx(0.05, abs=1e-9)
    assert len(res.boundary_contacts) == 1
    # 相切时在边界带内的时间是极短的一小段，而不是整步
    inside = float(res.boundary_contacts.iloc[0]["min_time_inside_band"])
    assert 0.0 < inside < 1.0e-3


# =============================================================================
# 4. 起始重叠
# =============================================================================
def test_start_overlap_all_pairs_violate_from_zero():
    """3 机起始完全重合：3 个机对全部从 t=0 起侵犯，跨步合并，各 1 个事件。"""
    n_agents = 3
    n_steps = 5
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, :, 0] = 0.1  # 全部重合于同一点
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.full((n_agents, 3), 20.0))
    res = compute(traj, scene)

    assert res.metrics["violating_pair_count"] == 3  # C(3,2)
    assert res.metrics["violation_event_count"] == 3
    assert res.metrics["first_violation_time"] == pytest.approx(0.0, abs=1e-12)
    assert res.metrics["min_separation"] == pytest.approx(0.0, abs=1e-12)
    assert res.metrics["violation_pair_seconds"] == pytest.approx(3 * 0.5, abs=1e-9)
    for _, ev in res.violation_events.iterrows():
        assert ev["start_time"] == pytest.approx(0.0, abs=1e-12)
        assert ev["end_time"] == pytest.approx(0.5, abs=1e-12)
        assert ev["min_distance"] == pytest.approx(0.0, abs=1e-12)
    assert res.agents["violated"].tolist() == [True, True, True]


# =============================================================================
# 5. 跨步区间合并
# =============================================================================
def test_cross_step_violation_interval_merged():
    """相对轨迹 r(t) = -1.5 + 4t，7 步：区间 [0.125, 0.625] 跨 6 步但只算 1 个事件。"""
    n_steps = 7
    dt = DT
    t = np.arange(n_steps + 1) * dt
    rel = -1.5 + 4.0 * t
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = 0.5 * rel
    positions[:, 1, 0] = -0.5 * rel
    velocities = np.broadcast_to(
        np.array([2.0, 0.0, 0.0]), (n_steps, 1, 3)
    ) * np.array([1.0, -1.0])[None, :, None]
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[20.0, 0, 0], [-20.0, 0, 0]]))
    res = compute(traj, scene)

    assert res.metrics["violation_event_count"] == 1  # 6 个违规步 -> 1 个事件
    assert res.metrics["violating_pair_count"] == 1
    ev = res.violation_events.iloc[0]
    assert ev["start_time"] == pytest.approx(0.125, abs=1e-7)
    assert ev["end_time"] == pytest.approx(0.625, abs=1e-7)
    assert ev["duration"] == pytest.approx(0.5, abs=1e-7)
    assert ev["min_distance"] == pytest.approx(0.0, abs=1e-12)
    assert ev["min_distance_time"] == pytest.approx(0.375, abs=1e-9)
    counts = res.min_distance_curve["violating_pair_count"].tolist()
    assert counts == [0, 1, 1, 1, 1, 1, 1]


# =============================================================================
# 6. d_eval 与 d_safe 两侧分类一致性
# =============================================================================
@pytest.mark.parametrize(
    "distance, expect_violation",
    [
        (1.0 - 5e-9, False),  # 在 (d_eval, d_safe) 之间：边界接触，不算侵犯
        (1.0 - 2e-8, True),   # 小于 d_eval：严格侵犯
    ],
)
def test_d_eval_vs_d_safe_classification(distance, expect_violation):
    cfg = make_cfg()
    assert cfg.d_eval < distance < cfg.simulation.d_safe if not expect_violation else distance < cfg.d_eval
    n_steps = 2
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = distance
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_ARRIVED, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[20.0, 0, 0], [-20.0, 0, 0]]))
    res = compute(traj, scene, cfg=cfg)

    assert res.metrics["min_separation"] == pytest.approx(distance, rel=1e-12)
    if expect_violation:
        assert res.metrics["violating_pair_count"] == 1
        assert res.metrics["violation_event_count"] == 1
        assert res.metrics["first_violation_time"] == pytest.approx(0.0, abs=1e-12)
        assert res.boundary_contacts.empty  # 严格侵犯者不进入边界接触表
        # 已完成但参与侵犯 -> 安全完成被取消
        assert res.metrics["safe_completion_count"] == 0
        assert res.agents["safe_completed"].tolist() == [False, False]
    else:
        assert res.metrics["violating_pair_count"] == 0
        assert res.metrics["violation_event_count"] == 0
        assert len(res.boundary_contacts) == 1
        contact = res.boundary_contacts.iloc[0]
        assert contact["min_distance"] == pytest.approx(distance, rel=1e-12)
        # 仅边界接触不算侵犯 -> 已完成者保持安全完成
        assert res.metrics["safe_completion_count"] == 2
        assert res.agents["safe_completed"].tolist() == [True, True]


# =============================================================================
# 7. 连续曲线与 geometry.py 标量原语一致
# =============================================================================
def test_min_distance_curve_matches_scalar_geometry_primitive():
    rng = np.random.default_rng(20261030)
    n_steps = 5
    n_agents = 3
    starts = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.5, 0.0]])
    velocities = rng.uniform(-2.0, 2.0, size=(n_steps, n_agents, 3))
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[0] = starts
    for k in range(n_steps):
        positions[k + 1] = positions[k] + velocities[k] * DT
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    goals = np.array([[10.0, 0, 0], [-10.0, 0, 0], [0, -10.0, 0]])
    res = compute(traj, DuckScene(starts, goals))

    for k in range(n_steps):
        expected_sq = np.inf
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                r = positions[k, i] - positions[k, j]
                dp = positions[k + 1, i] - positions[k + 1, j]
                expected_sq = min(
                    expected_sq, float(geometry.segment_point_min_dist_sq(r, dp))
                )
        got = float(res.min_distance_curve.iloc[k]["min_distance"])
        assert got == pytest.approx(float(np.sqrt(expected_sq)), rel=1e-12)
    assert res.metrics["min_separation"] == pytest.approx(
        float(res.min_distance_curve["min_distance"].min()), rel=1e-12
    )


# =============================================================================
# 8. 重复侵犯：机对只计一次，事件分两段
# =============================================================================
def test_repeated_violations_pair_counted_once_events_twice():
    """相对位置先在步 0.5-1.5 s 进入，离开后再于 2.5-3.5 s 进入。"""
    dt = DT
    n_steps = 40
    t = np.arange(n_steps + 1) * dt

    def rel_x(times):
        times = np.asarray(times, dtype=np.float64)
        first = -2.0 + 2.0 * times
        second = 2.0 - 2.0 * (times - 2.0)
        return np.where(times <= 2.0, first, second)

    rel = rel_x(t)
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = 0.5 * rel
    positions[:, 1, 0] = -0.5 * rel
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    w = np.where(t[:-1] < 2.0, 2.0, -2.0)
    velocities[:, 0, 0] = 0.5 * w
    velocities[:, 1, 0] = -0.5 * w
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[30.0, 0, 0], [-30.0, 0, 0]]))
    res = compute(traj, scene)

    assert res.metrics["violating_pair_count"] == 1  # 同一机对只计一次
    assert res.metrics["violation_event_count"] == 2  # 两段不连续区间
    starts = res.violation_events["start_time"].to_numpy()
    ends = res.violation_events["end_time"].to_numpy()
    assert starts[0] == pytest.approx(0.5, abs=1e-2)
    assert ends[0] == pytest.approx(1.5, abs=1e-2)
    assert starts[1] == pytest.approx(2.5, abs=1e-2)
    assert ends[1] == pytest.approx(3.5, abs=1e-2)
    assert res.metrics["violation_pair_seconds"] == pytest.approx(2.0, abs=2e-2)
    assert res.metrics["first_violation_time"] == pytest.approx(0.5, abs=1e-2)
    assert res.metrics["min_separation"] == pytest.approx(0.0, abs=1e-12)
    assert res.agents["violated"].tolist() == [True, True]


# =============================================================================
# 9. 大 dt 纯几何完全穿越（明确不属于主实验参数）
# =============================================================================
def test_large_dt_full_crossing_merges_across_step_boundary():
    """dt=1.0 s（非主实验参数）：相对轨迹 -2.5 -> 3.5 m，穿越区间跨两个大步骤合并。"""
    dt = 1.0
    n_steps = 2
    rel0 = -2.5
    rel_vel = 3.0
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    rel = rel0 + rel_vel * np.arange(n_steps + 1) * dt
    positions[:, 0, 0] = 0.5 * rel
    positions[:, 1, 0] = -0.5 * rel
    velocities = np.broadcast_to(
        np.array([1.5, 0.0, 0.0]), (n_steps, 1, 3)
    ) * np.array([1.0, -1.0])[None, :, None]
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages, dt=dt, observation_time=2.0)
    scene = DuckScene(positions[0], np.array([[20.0, 0, 0], [-20.0, 0, 0]]))
    cfg = make_cfg(dt=dt, observation_time=2.0)
    res = compute(traj, scene, cfg=cfg)

    assert res.metrics["violation_event_count"] == 1
    d_eval = cfg.d_eval
    expected_start = (2.5 - d_eval) / 3.0
    expected_end = (2.5 + d_eval) / 3.0
    ev = res.violation_events.iloc[0]
    assert ev["start_time"] == pytest.approx(expected_start, abs=1e-6)
    assert ev["end_time"] == pytest.approx(expected_end, abs=1e-6)
    assert ev["duration"] == pytest.approx(expected_end - expected_start, abs=1e-6)
    assert ev["min_distance"] == pytest.approx(0.0, abs=1e-12)
    assert ev["min_distance_time"] == pytest.approx(2.5 / 3.0, abs=1e-6)
    assert res.min_distance_curve["violating_pair_count"].tolist() == [1, 1]


# =============================================================================
# 10. 中心首次进入几何：与 geometry 标量原语一致，且能校验仿真提供值
# =============================================================================
def test_center_entry_geometry_consistent_with_scalar_primitive():
    dt = DT
    n_steps = 30
    t = np.arange(n_steps + 1) * dt
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = 2.0 - 2.0 * t  # 2 m/s 沿 -x，t=0.95 时首次进入 0.1 m 中心球
    positions[:, 1, 0] = 50.0
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    velocities[:, 0, 0] = -2.0
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[-20.0, 0, 0], [60.0, 0, 0]]))

    # 第 9 步（[0.9, 1.0]）内从 0.2 m 到 0.0 m：进入参数 s=0.5，即 t=0.95
    s = geometry.segment_sphere_enter_time(
        positions[9, 0], positions[10, 0] - positions[9, 0], 0.1, closed=True
    )
    assert s == pytest.approx(0.5, rel=1e-12)

    res = compute(traj, scene, center_times=np.array([0.95, np.nan]))
    consistency = res.metrics["consistency"]
    assert res.metrics["center_crossing_count"] == 1
    assert res.metrics["center_crossing_rate"] is None  # 场景 A 不适用
    assert consistency["center_entry_checked"] == 1
    assert consistency["center_entry_max_abs_diff_s"] == pytest.approx(0.0, abs=1e-9)
    assert consistency["center_entry_mismatch_count"] == 0
    assert consistency["center_entry_position_violations"] == 0
    # 中心进入方程与标量原语互相验证：t = 0.9 + s*dt
    assert 0.9 + s * dt == pytest.approx(0.95, rel=1e-12)
