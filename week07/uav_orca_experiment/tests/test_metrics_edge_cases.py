# OWNER: metrics-dev
"""指标模块边界情况与健壮性测试（CONTRACTS §11、handoff §6.4/§8）。

覆盖：零步长轨迹、全部在 t=0 到达、多机对起始重叠、空事件表保留表头、
JSON null 语义、非有限轨迹数值错误标记、中心进入步索引校验、``(K,N)`` / ``(K,)``
两种兼容诊断布局、截断网格告警、非均匀网格拒绝、配置驱动的阈值（eps_distance /
goal_tolerance / observation_time）、场景 A 放行列缺失语义、曲线边界带计数。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from uavbench.config import Config, load_config
from uavbench.metrics import (
    AGENT_COLUMNS,
    BOUNDARY_CONTACT_COLUMNS,
    MIN_DISTANCE_CURVE_COLUMNS,
    VIOLATION_EVENT_COLUMNS,
    compute_metrics,
    write_metrics,
)
from uavbench.navigation import STAGE_ARRIVED, STAGE_TO_CENTER, STAGE_TO_GOAL, STAGE_WAITING

DT = 0.1
OBS = 120.0


# =============================================================================
# 测试内构造工具
# =============================================================================
class DuckScene:
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


def static_run(n_agents, n_steps, gaps=None, stages=None):
    gaps = 5.0 * np.arange(n_agents) if gaps is None else np.asarray(gaps, dtype=np.float64)
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, :, 1] = gaps[None, :]
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    if stages is None:
        stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    goals = positions[-1].copy()
    return build_traj(positions, velocities, stages), DuckScene(positions[0], goals)


# =============================================================================
# 1. 最小规模与零步长
# =============================================================================
def test_minimum_two_agents():
    n_steps = 10
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, :] = (0.0, 2.0, 0.0)  # 静止，与 0 号机保持 2 m 间距
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    positions[:, 0, 0] = 2.0 * np.arange(n_steps + 1) * DT  # 沿 +x 运动
    velocities[:, 0, 0] = 2.0
    goals = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    traj = build_traj(positions, velocities, stages)
    res = compute(traj, DuckScene(positions[0], goals))
    assert res.metrics["n_agents"] == 2
    assert res.metrics["violating_pair_count"] == 0
    assert res.metrics["min_separation"] == pytest.approx(2.0, rel=1e-6)
    assert json.dumps(res.metrics, allow_nan=False)  # 全部为可序列化的有限值/None


def test_zero_step_trajectory_handles_degenerate_grid():
    """K=0（仅初始网格点）不崩溃；无区间因而无事件，曲线为空表但保留表头。"""
    positions = np.zeros((1, 2, 3), dtype=np.float64)
    positions[0, 1, 0] = 0.5
    stages = np.full((1, 2), STAGE_ARRIVED, dtype=np.int8)
    traj = {
        "t": np.array([0.0]),
        "positions": positions,
        "velocities": np.zeros((0, 2, 3)),
        "stages": stages,
        "dt": DT,
        "n_agents": 2,
        "observation_time": OBS,
    }
    res = compute(traj, DuckScene(positions[0], np.array([[1.0, 0, 0], [-1.0, 0, 0]])))
    assert res.metrics["arrival_count"] == 2
    assert res.metrics["all_arrival_time"] == pytest.approx(0.0, abs=1e-12)
    assert res.metrics["min_separation"] == pytest.approx(0.5, rel=1e-12)
    assert res.min_distance_curve.empty
    assert list(res.min_distance_curve.columns) == list(MIN_DISTANCE_CURVE_COLUMNS)
    assert res.violation_events.empty
    assert res.arrival_curve.shape == (1, 5)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        write_metrics(out, res)
        header = (out / "min_distance_curve.csv").read_text(encoding="utf-8")
        assert header.strip() == ",".join(MIN_DISTANCE_CURVE_COLUMNS)


def test_all_agents_arrived_at_time_zero():
    n_agents = 4
    n_steps = 20
    traj, scene = static_run(
        n_agents,
        n_steps,
        stages=np.full((n_steps + 1, n_agents), STAGE_ARRIVED, dtype=np.int8),
    )
    res = compute(traj, scene)
    metrics = res.metrics
    assert metrics["arrival_count"] == 4
    assert metrics["all_arrival_time"] == pytest.approx(0.0, abs=1e-12)
    # N=4 时 T90 需要 ceil(0.9*4)=4 架完成 -> 0 s
    assert metrics["t90"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["mean_completion_time_limited"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["safe_completion_count"] == 4
    assert metrics["team_safe_success"] is True
    assert res.agents["arrival_time"].to_numpy() == pytest.approx(np.zeros(4), abs=1e-12)
    assert res.agents["path_length"].to_numpy() == pytest.approx(np.zeros(4), abs=1e-12)


# =============================================================================
# 2. 多机对起始重叠 / 空事件表
# =============================================================================
def test_multiple_pairs_start_overlap_counts():
    n_agents = 5
    n_steps = 4
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, 0, :] = (0.0, 0.0, 0.0)
    positions[:, 1, :] = (0.0, 0.0, 0.0)
    positions[:, 2, :] = (0.0, 0.0, 0.0)  # 三机完全重合 -> C(3,2)=3 个侵犯机对
    positions[:, 3, :] = (5.0, 0.0, 0.0)
    positions[:, 4, :] = (10.0, 0.0, 0.0)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    goals = np.full((n_agents, 3), 100.0)
    res = compute(build_traj(positions, velocities, stages), DuckScene(positions[0], goals))
    assert res.metrics["violating_pair_count"] == 3
    assert res.metrics["violation_event_count"] == 3
    assert res.metrics["violation_pair_seconds"] == pytest.approx(3 * 0.4, abs=1e-9)
    assert res.agents["violated"].tolist() == [True, True, True, False, False]


def test_empty_event_tables_keep_header(tmp_path):
    n_agents = 3
    n_steps = 20
    traj, scene = static_run(n_agents, n_steps)
    res = compute(traj, scene)
    assert res.violation_events.empty
    assert res.boundary_contacts.empty
    out = tmp_path / "run"
    write_metrics(out, res)
    events_lines = (out / "violation_events.csv").read_text(encoding="utf-8").splitlines()
    contacts_lines = (out / "boundary_contacts.csv").read_text(encoding="utf-8").splitlines()
    assert len(events_lines) == 1
    assert events_lines[0] == ",".join(VIOLATION_EVENT_COLUMNS)
    assert len(contacts_lines) == 1
    assert contacts_lines[0] == ",".join(BOUNDARY_CONTACT_COLUMNS)
    # agents.csv 行数 = N + 表头
    assert len((out / "agents.csv").read_text(encoding="utf-8").splitlines()) == n_agents + 1
    header = (out / "agents.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(AGENT_COLUMNS)


# =============================================================================
# 3. JSON null 语义 / 非有限轨迹
# =============================================================================
def test_json_null_semantics_for_missing_values(tmp_path):
    traj, scene = static_run(2, 20)
    res = compute(traj, scene)
    metrics = res.metrics
    # 无完成者 / 无事件 / A 场景中心率 -> null
    assert metrics["all_arrival_time"] is None
    assert metrics["t90"] is None
    assert metrics["first_violation_time"] is None
    assert metrics["center_crossing_rate"] is None
    assert metrics["mean_path_efficiency_excess"] is None
    assert metrics["planning"]["mean_ms"] is None
    assert metrics["planning"]["p95_ms"] is None
    out = tmp_path / "run"
    write_metrics(out, res)
    parsed = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert parsed["all_arrival_time"] is None
    assert parsed["t90"] is None
    assert parsed["center_crossing_rate"] is None
    assert parsed["planning"]["mean_ms"] is None
    assert parsed["mean_path_efficiency_excess"] is None


def test_nan_trajectory_flagged_as_numerical_error(tmp_path):
    n_agents = 2
    n_steps = 10
    traj, scene = static_run(n_agents, n_steps)
    traj["positions"] = traj["positions"].copy()
    traj["positions"][5, 0, 0] = np.nan
    res = compute(traj, scene)
    assert res.metrics["status"]["numerical_error"] is True
    assert res.metrics["status"]["valid"] is False
    assert res.metrics["consistency"]["non_finite_data"] is True
    out = tmp_path / "run"
    write_metrics(out, res)
    text = (out / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)
    assert parsed["status"]["valid"] is False


# =============================================================================
# 4. 中心进入步索引 / 截断网格 / 非均匀网格
# =============================================================================
def test_center_entry_step_index_mismatch_detected():
    n_steps = 30
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = 2.0 - 2.0 * t  # 首次进入在 t=0.95（第 9 步内）
    positions[:, 1, 0] = 50.0
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    velocities[:, 0, 0] = -2.0
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[-20.0, 0, 0], [60.0, 0, 0]]))
    res = compute(
        traj,
        scene,
        center_times=np.array([0.95, np.nan]),
        center_entry_steps=np.array([8, -1]),  # 正确步索引应为 9
    )
    consistency = res.metrics["consistency"]
    assert consistency["center_entry_step_checked"] == 1
    assert consistency["center_entry_step_mismatch_count"] == 1
    assert consistency["center_entry_mismatch_count"] == 0  # 时刻本身一致


def test_truncated_grid_flagged_in_consistency():
    n_agents = 2
    n_steps = 20  # 2 s，而主实验观察窗 120 s
    traj, scene = static_run(n_agents, n_steps)
    res = compute(traj, scene)
    consistency = res.metrics["consistency"]
    assert consistency["grid_steps"] == 20
    assert consistency["expected_grid_steps"] == 1200
    assert consistency["grid_end_time"] == pytest.approx(2.0, abs=1e-9)
    assert consistency["grid_complete"] is False
    # 未完成者仍按 120 s 观察窗计入限制平均完成时间
    assert res.metrics["mean_completion_time_limited"] == pytest.approx(120.0)


def test_non_uniform_time_grid_rejected():
    n_steps = 2
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = 2.0
    traj = {
        "t": np.array([0.0, 0.1, 0.25]),  # 非均匀
        "positions": positions,
        "velocities": np.zeros((n_steps, 2, 3)),
        "stages": np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8),
        "dt": 0.1,
        "n_agents": 2,
        "observation_time": OBS,
    }
    with pytest.raises(ValueError, match="uniform"):
        compute(traj, DuckScene(positions[0], np.zeros((2, 3))))


# =============================================================================
# 5. 配置驱动阈值
# =============================================================================
@pytest.mark.parametrize(
    "distance, expect_violation",
    [
        (1.0 - 5e-7, False),  # 落在 d_safe 与 d_eval 之间：边界接触
        (1.0 - 2e-6, True),   # 小于 d_eval：严格侵犯
    ],
)
def test_config_eps_distance_threshold(distance, expect_violation):
    cfg = make_cfg(**{"metrics.eps_distance": 1e-6})
    assert cfg.d_eval == pytest.approx(1.0 - 1e-6)
    n_steps = 2
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = distance
    stages = np.full((n_steps + 1, 2), STAGE_ARRIVED, dtype=np.int8)
    traj = build_traj(positions, np.zeros((n_steps, 2, 3)), stages)
    res = compute(traj, DuckScene(positions[0], np.array([[1.0, 0, 0], [-1.0, 0, 0]])), cfg=cfg)
    if expect_violation:
        assert res.metrics["violating_pair_count"] == 1
        assert res.boundary_contacts.empty
    else:
        assert res.metrics["violating_pair_count"] == 0
        assert len(res.boundary_contacts) == 1


def test_config_goal_tolerance_changes_path_efficiency_lower_bound():
    cfg = make_cfg(**{"simulation.goal_tolerance": 0.5})
    n_agents = 2
    n_steps = 40
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, 1, :] = (0.0, 20.0, 0.0)
    positions[:, 0, 0] = 2.0 * np.minimum(t, 2.0)  # 路径 4 m，2 s 到达
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    velocities[t[:-1] < 2.0 - 1e-12, 0, 0] = 2.0
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    stages[t >= 2.0 - 1e-12, 0] = STAGE_ARRIVED
    goals = np.array([[4.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    res = compute(build_traj(positions, velocities, stages), DuckScene(positions[0], goals), cfg=cfg)
    lower = 2.0 * 20.0 - 0.5  # 39.5
    assert res.metrics["completed_count_for_path_metrics"] == 1
    assert res.metrics["mean_path_efficiency_excess"] == pytest.approx(4.0 / lower - 1.0, rel=1e-12)


def test_config_observation_time_used_for_limited_mean():
    cfg = make_cfg(observation_time=10.0)
    n_agents = 2
    n_steps = 100  # 10 s，与配置观察窗一致
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, 1, :] = (0.0, 20.0, 0.0)
    positions[:, 0, 0] = 2.0 * np.minimum(t, 2.0)  # 2 s 到达
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    velocities[t[:-1] < 2.0 - 1e-12, 0, 0] = 2.0
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    stages[t >= 2.0 - 1e-12, 0] = STAGE_ARRIVED
    goals = np.array([[4.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    res = compute(build_traj(positions, velocities, stages), DuckScene(positions[0], goals), cfg=cfg)
    assert res.metrics["consistency"]["grid_complete"] is True
    # 完成者 2 s，未完成者按观察窗 10 s：mean = 6 s
    assert res.metrics["mean_completion_time_limited"] == pytest.approx(6.0, abs=1e-9)


# =============================================================================
# 6. 场景 A 放行列缺失语义
# =============================================================================
def test_scenario_a_release_columns_missing_and_wait_zero():
    n_agents = 2
    n_steps = 30
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, 1, :] = (0.0, 20.0, 0.0)
    finishes = [1.0, 2.0]
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    for i, dur in enumerate(finishes):
        positions[:, i, 0] = 2.0 * np.minimum(t, dur)
        velocities[t[:-1] < dur - 1e-12, i, 0] = 2.0
        stages[t >= dur - 1e-12, i] = STAGE_ARRIVED
    goals = np.array([[2.0, 0.0, 0.0], [4.0, 20.0, 0.0]])
    res = compute(build_traj(positions, velocities, stages), DuckScene(positions[0], goals))
    metrics = res.metrics
    # A：放行列为 NaN，但 wait_duration 恒为 0，等待统计为 0
    assert np.isnan(res.agents["release_planned"].to_numpy()).all()
    assert np.isnan(res.agents["release_actual"].to_numpy()).all()
    assert res.agents["wait_duration"].to_numpy() == pytest.approx([0.0, 0.0])
    assert metrics["waiting"]["mean_wait"] == pytest.approx(0.0)
    assert metrics["waiting"]["total_wait"] == pytest.approx(0.0)
    # A 的“放行后行程时间” = 到达时刻（虚拟放行时刻 0）
    assert res.agents["post_release_travel_time"].to_numpy() == pytest.approx([1.0, 2.0], abs=1e-9)
    assert metrics["waiting"]["mean_post_release_travel"] == pytest.approx(1.5, abs=1e-9)


# =============================================================================
# 7. 兼容诊断布局（(K,N) 掩码 / (K,) 逐步聚合）
# =============================================================================
def test_diag_2d_mask_layout_aggregation():
    n_agents = 3
    n_steps = 4
    traj, scene = static_run(n_agents, n_steps)
    mask = np.zeros((n_steps, n_agents), dtype=bool)
    mask[:, 0] = True
    mask[:2, 1] = True
    traj["active"] = mask.copy()
    traj["diag_active_ids"] = np.where(mask, np.arange(n_agents)[None, :], -1)
    sample = np.zeros((n_steps, n_agents), dtype=bool)
    sample[0, 0] = True
    sample[3, 0] = True
    sample[1, 2] = True  # 非活动位：不应被计数
    traj["diag_sample_exhausted"] = sample
    fallback = np.zeros((n_steps, n_agents), dtype=bool)
    fallback[1, 1] = True
    traj["diag_used_fallback"] = fallback
    constraint = np.zeros((n_steps, n_agents), dtype=np.float64)
    constraint[2, 0] = 0.4
    constraint[0, 2] = 99.0  # 非活动位：不应计入最大值
    traj["diag_constraint_violation_max"] = constraint
    overlap = np.zeros((n_steps, n_agents), dtype=np.int32)
    overlap[0, 1] = 2
    overlap[3, 0] = 1
    traj["diag_initial_overlap_pairs"] = overlap

    res = compute(traj, scene)
    diag = res.metrics["planner_diagnostics"]
    assert diag["sample_exhausted_count"] == 2  # (0,0),(3,0)
    assert diag["solver_fallback_count"] == 1
    assert diag["constraint_violation_max"] == pytest.approx(0.4)
    assert diag["initial_overlap_pairs_total"] == 3
    assert diag["active_planning_calls"] == int(mask.sum()) == 6


def test_diag_per_step_1d_layout_aggregation():
    n_agents = 2
    n_steps = 4
    traj, scene = static_run(n_agents, n_steps)
    traj["diag_sample_exhausted"] = np.array([1, 0, 2, 0])  # 每步计数
    traj["diag_used_fallback"] = np.array([0, 1, 0, 0])
    traj["diag_solver_infeasible"] = np.array([0, 0, 0, 1])
    traj["diag_constraint_violation_max"] = np.array([0.1, 0.2, 0.0, 0.3])
    traj["diag_initial_overlap_pairs"] = np.array([1, 0, 2, 0])
    res = compute(traj, scene)
    diag = res.metrics["planner_diagnostics"]
    assert diag["sample_exhausted_count"] == 3
    assert diag["solver_fallback_count"] == 1
    assert diag["solver_infeasible_count"] == 1
    assert diag["constraint_violation_max"] == pytest.approx(0.3)
    assert diag["initial_overlap_pairs_total"] == 3
    assert diag["active_planning_calls"] == int(traj["active"].sum())


# =============================================================================
# 8. 曲线边界带计数
# =============================================================================
def test_min_distance_curve_boundary_and_violation_counts():
    n_agents = 3
    n_steps = 3
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, 0, :] = (0.0, 0.0, 0.0)
    positions[:, 1, :] = (0.5, 0.0, 0.0)  # 与 0 号机侵犯
    positions[:, 2, :] = (0.0, 1.0, 0.0)  # 与 0 号机恰好 1.0 m：边界接触
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_ARRIVED, dtype=np.int8)
    goals = np.full((n_agents, 3), 100.0)
    res = compute(build_traj(positions, velocities, stages), DuckScene(positions[0], goals))

    curve = res.min_distance_curve
    assert curve["violating_pair_count"].tolist() == [1, 1, 1]
    assert curve["boundary_pair_count"].tolist() == [1, 1, 1]
    assert curve["min_distance"].to_numpy() == pytest.approx([0.5, 0.5, 0.5])
    # 边界接触表只含恰好 1 m 的机对（0-2）
    assert len(res.boundary_contacts) == 1
    contact = res.boundary_contacts.iloc[0]
    assert (int(contact["agent_i"]), int(contact["agent_j"])) == (0, 2)
    assert contact["min_time_inside_band"] == pytest.approx(n_steps * DT, abs=1e-9)


# =============================================================================
# 9. 状态默认值与到达率一致性
# =============================================================================
def test_status_defaults_from_arrival_state():
    n_agents = 2
    n_steps = 40
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, 1, :] = (0.0, 20.0, 0.0)
    positions[:, 0, 0] = 2.0 * np.minimum(t, 2.0)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    velocities[t[:-1] < 2.0 - 1e-12, 0, 0] = 2.0
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    stages[t >= 2.0 - 1e-12, 0] = STAGE_ARRIVED
    goals = np.array([[4.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    res = compute(build_traj(positions, velocities, stages), DuckScene(positions[0], goals))
    assert res.metrics["status"]["completed"] is False
    assert res.metrics["status"]["timeout"] is True
    assert res.metrics["status"]["valid"] is True
    assert res.metrics["arrival_rate"] == pytest.approx(0.5)
    assert res.metrics["safe_completion_rate"] == pytest.approx(0.5)


def test_planning_timing_p95_matches_numpy():
    n_agents = 2
    n_steps = 6
    traj, scene = static_run(n_agents, n_steps)
    times = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    traj["planning_step_seconds"] = times
    res = compute(traj, scene)
    planning = res.metrics["planning"]
    assert planning["mean_ms"] == pytest.approx(float(times.mean()) * 1000.0)
    assert planning["p95_ms"] == pytest.approx(float(np.percentile(times, 95.0)) * 1000.0)
    assert planning["max_ms"] == pytest.approx(float(times.max()) * 1000.0)
    assert planning["total_s"] == pytest.approx(float(times.sum()))
    calls = int(traj["active"].sum())
    assert planning["per_active_call_ms"] == pytest.approx(float(times.sum()) * 1000.0 / calls)
