# OWNER: metrics-dev
"""指标模块行为测试（CONTRACTS §11.2/§11.3、handoff §6/§8）。

覆盖：metrics.json 全部键、五类课程指标、安全完成（含到达后失去资格）、120 s 静止尾段、
到达曲线、T90、全部到达时间、限制平均完成时间、路程/效率、等待与阶段时间、
未完成分类、停滞规则、规划诊断与墙钟聚合、状态透传、CSV 表头、JSON 无 NaN/Infinity、
离线可复现性、场景稳定 ID 映射。

全部轨迹在测试内构造，不依赖仿真器。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from uavbench.config import Config, load_config
from uavbench.metrics import (
    AGENT_COLUMNS,
    ARRIVAL_CURVE_COLUMNS,
    BOUNDARY_CONTACT_COLUMNS,
    MIN_DISTANCE_CURVE_COLUMNS,
    VIOLATION_EVENT_COLUMNS,
    compute_metrics,
    write_metrics,
)
from uavbench.navigation import (
    STAGE_ARRIVED,
    STAGE_TO_CENTER,
    STAGE_TO_GOAL,
    STAGE_WAITING,
)

DT = 0.1
OBS = 120.0

#: CONTRACTS §11.2 冻结键（顺序即规范顺序）。
METRICS_TOP_KEYS = (
    "experiment_id",
    "model",
    "scenario",
    "scene_id",
    "seed",
    "scene_hash",
    "config_hash",
    "status",
    "n_agents",
    "arrival_count",
    "arrival_rate",
    "completed_all",
    "safe_completion_count",
    "safe_completion_rate",
    "center_crossing_count",
    "center_crossing_rate",
    "team_safe_success",
    "min_separation",
    "min_separation_time",
    "violating_pair_count",
    "violation_event_count",
    "violation_pair_seconds",
    "first_violation_time",
    "all_arrival_time",
    "t90",
    "mean_completion_time_limited",
    "average_path_length",
    "mean_path_efficiency_excess",
    "mean_path_over_40_minus_1",
    "completed_count_for_path_metrics",
    "waiting",
    "planning",
    "planner_diagnostics",
    "stall",
    "uncompleted",
)


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


def static_case(n_agents: int, n_steps: int, goals=None, *, scenario="A", gap=2.0):
    """静止、互不侵犯的简单轨迹（可选择全部到达/全部超时）。"""
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, :, 1] = gap * np.arange(n_agents)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    if goals is None:
        goals = np.zeros((n_agents, 3))
        goals[:, 1] = gap * np.arange(n_agents)
    traj = build_traj(positions, velocities, stages)
    return traj, DuckScene(positions[0], goals)


def line_arrival_case(n_agents, n_steps, finish_times, *, offsets=None, goals=None):
    """每机沿 +x 以 2 m/s 运动并在各自 ``finish_times`` 网格时刻到达后冻结。"""
    finish = np.asarray(finish_times, dtype=np.float64)
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    y = 10.0 + 3.0 * np.arange(n_agents) if offsets is None else np.asarray(offsets, dtype=np.float64)
    positions[:, :, 1] = y[None, :]
    for i in range(n_agents):
        dur = float(finish[i])
        positions[:, i, 0] = 2.0 * np.minimum(t, dur)
        velocities[t[:-1] < dur - 1e-12, i, 0] = 2.0
        stages[t >= dur - 1e-12, i] = STAGE_ARRIVED
    if goals is None:
        goals = np.zeros((n_agents, 3))
        goals[:, 0] = 2.0 * finish
        goals[:, 1] = y
    return build_traj(positions, velocities, stages), DuckScene(positions[0], goals)


# =============================================================================
# 1. metrics.json 契约键与 CSV 表头
# =============================================================================
def test_metrics_dict_has_all_contract_keys_and_subkeys():
    traj, scene = line_arrival_case(4, 60, [0.5, 1.0, 1.5, 2.0])
    res = compute(traj, scene)
    metrics = res.metrics

    for key in METRICS_TOP_KEYS:
        assert key in metrics, f"missing metrics key {key}"
    assert set(metrics["status"]) == {"completed", "timeout", "numerical_error", "valid"}
    assert set(metrics["waiting"]) == {"mean_wait", "total_wait", "mean_post_release_travel"}
    assert set(metrics["planning"]) == {
        "mean_ms",
        "p95_ms",
        "max_ms",
        "total_s",
        "calls",
        "per_active_call_ms",
    }
    assert set(metrics["planner_diagnostics"]) >= {
        "sample_exhausted_count",
        "solver_fallback_count",
        "solver_infeasible_count",
        "constraint_violation_max",
        "initial_overlap_pairs_total",
        "numerical_error_count",
        "active_planning_calls",
    }
    assert set(metrics["stall"]) == {"stalled_agent_count", "low_progress_mobile_count"}
    assert set(metrics["uncompleted"]) == {"not_released", "not_center", "center_done_not_arrived"}

    # 基本数值一致性
    assert metrics["model"] == "direct"
    assert metrics["scenario"] == "A"
    assert metrics["scene_id"] == "unit"
    assert metrics["seed"] == 0
    assert metrics["n_agents"] == 4
    assert metrics["arrival_count"] == 4
    assert metrics["arrival_rate"] == pytest.approx(1.0)
    assert metrics["completed_all"] is True
    assert metrics["center_crossing_rate"] is None  # 场景 A 必须为 null
    assert metrics["all_arrival_time"] == pytest.approx(2.0, abs=1e-9)
    assert metrics["team_safe_success"] is True


def test_agents_dataframe_and_csv_headers_exact():
    traj, scene = line_arrival_case(4, 60, [0.5, 1.0, 1.5, 2.0])
    res = compute(traj, scene)
    assert list(res.agents.columns) == list(AGENT_COLUMNS)


def test_all_csv_headers_exact(tmp_path):
    traj, scene = static_case(3, 20)
    res = compute(traj, scene)
    out_dir = tmp_path / "run"
    write_metrics(out_dir, res)

    expected = {
        "agents.csv": AGENT_COLUMNS,
        "violation_events.csv": VIOLATION_EVENT_COLUMNS,
        "boundary_contacts.csv": BOUNDARY_CONTACT_COLUMNS,
        "min_distance_curve.csv": MIN_DISTANCE_CURVE_COLUMNS,
        "arrival_curve.csv": ARRIVAL_CURVE_COLUMNS,
    }
    for name, columns in expected.items():
        text = (out_dir / name).read_text(encoding="utf-8")
        header = text.splitlines()[0]
        assert header == ",".join(columns), f"{name} header mismatch: {header}"


def test_write_metrics_creates_files_and_nested_dirs(tmp_path):
    traj, scene = static_case(2, 10)
    res = compute(traj, scene)
    out_dir = tmp_path / "nested" / "run"
    assert not out_dir.exists()
    write_metrics(out_dir, res)
    assert out_dir.is_dir()
    for name in (
        "metrics.json",
        "agents.csv",
        "violation_events.csv",
        "boundary_contacts.csv",
        "min_distance_curve.csv",
        "arrival_curve.csv",
    ):
        assert (out_dir / name).is_file(), name


def test_metrics_json_has_no_nan_or_infinity(tmp_path):
    # 全部超时 + 场景 A：all_arrival_time / t90 / center_crossing_rate 等为 null
    traj, scene = static_case(4, 20)
    res = compute(traj, scene)
    assert res.metrics["all_arrival_time"] is None
    assert res.metrics["t90"] is None
    assert res.metrics["center_crossing_rate"] is None
    out_dir = tmp_path / "run"
    write_metrics(out_dir, res)
    text = (out_dir / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    parsed = json.loads(text)  # 严格 JSON 可解析
    assert parsed["all_arrival_time"] is None
    assert parsed["t90"] is None
    assert parsed["status"]["timeout"] is True
    assert parsed["status"]["completed"] is False


# =============================================================================
# 2. 到达率、T90、全部到达时间、限制平均完成时间、到达曲线
# =============================================================================
def test_t90_all_arrival_time_and_limited_mean_completion():
    n_agents = 100
    finish = np.arange(1, n_agents + 1, dtype=np.float64) * DT  # 0.1 .. 10.0 s
    traj, scene = line_arrival_case(n_agents, 120, finish)
    res = compute(traj, scene)
    metrics = res.metrics

    assert metrics["arrival_count"] == 100
    assert metrics["arrival_rate"] == pytest.approx(1.0)
    assert metrics["completed_all"] is True
    assert metrics["all_arrival_time"] == pytest.approx(10.0, abs=1e-9)
    # T90：第 90 架完成的最早网格时刻 = 9.0 s
    assert metrics["t90"] == pytest.approx(9.0, abs=1e-9)
    # 限制平均完成时间 = mean(min(T_i,120)) = mean(finish)
    assert metrics["mean_completion_time_limited"] == pytest.approx(5.05, abs=1e-9)
    # 安全完成：平行航线间距 3 m，无侵犯
    assert metrics["violating_pair_count"] == 0
    assert metrics["safe_completion_count"] == 100
    assert metrics["team_safe_success"] is True
    assert metrics["min_separation"] == pytest.approx(3.0, rel=1e-12)

    curve = res.arrival_curve
    assert list(curve.columns) == list(ARRIVAL_CURVE_COLUMNS)
    assert len(curve) == traj["t"].size
    assert curve["cumulative_arrivals"].to_numpy()[-1] == 100
    assert (np.diff(curve["cumulative_arrivals"].to_numpy()) >= 0).all()
    # t=1.0 s 时已完成 10 架
    row = curve.iloc[int(np.argmin(np.abs(curve["t"].to_numpy() - 1.0)))]
    assert row["t"] == pytest.approx(1.0, rel=1e-12)
    assert row["cumulative_arrivals"] == 10
    assert row["cumulative_center_visits"] == 0
    assert row["cumulative_safe_completions_known_later"] == 10


def test_all_arrival_time_null_when_timeout():
    traj, scene = static_case(4, 20)
    res = compute(traj, scene)
    assert res.metrics["arrival_count"] == 0
    assert res.metrics["all_arrival_time"] is None
    assert res.metrics["t90"] is None
    assert res.metrics["completed_all"] is False
    assert res.metrics["mean_completion_time_limited"] == pytest.approx(120.0)
    assert res.metrics["status"] == {
        "completed": False,
        "timeout": True,
        "numerical_error": False,
        "valid": True,
    }


def test_mean_completion_time_limited_counts_uncompleted_as_window():
    # 4 机：2 架在 10 s / 20 s 到达，其余永不完成 -> mean([10, 20, 120, 120]) = 67.5
    n_agents = 4
    n_steps = 200
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    positions[:, :, 1] = 5.0 * np.arange(n_agents)
    finish = [10.0, 20.0, np.inf, np.inf]
    for i, dur in enumerate(finish):
        if np.isfinite(dur):
            positions[:, i, 0] = 2.0 * np.minimum(t, dur)
            velocities[t[:-1] < dur - 1e-12, i, 0] = 2.0
            stages[t >= dur - 1e-12, i] = STAGE_ARRIVED
    goals = np.zeros((n_agents, 3))
    goals[:, 1] = 5.0 * np.arange(n_agents)
    goals[0, 0] = 20.0
    goals[1, 0] = 40.0
    goals[2, 0] = 100.0
    goals[3, 0] = 100.0
    traj = build_traj(positions, velocities, stages)
    res = compute(traj, DuckScene(positions[0], goals))

    assert res.metrics["arrival_count"] == 2
    assert res.metrics["completed_all"] is False
    assert res.metrics["all_arrival_time"] is None
    assert res.metrics["mean_completion_time_limited"] == pytest.approx(67.5, abs=1e-9)
    # N=4 时 T90 需要 ceil(0.9*4)=4 架完成 -> 缺失
    assert res.metrics["t90"] is None


# =============================================================================
# 3. 路程与效率（只对完成者计算效率；平均路程包含未完成者）
# =============================================================================
def test_path_length_and_efficiency_metrics():
    n_agents = 4
    n_steps = 60
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    # agent0: (-10,0,0) -> (0,0,0)，5 s 到达，路径 10 m
    positions[:, 0, 0] = -10.0 + 2.0 * np.minimum(t, 5.0)
    velocities[t[:-1] < 5.0 - 1e-12, 0, 0] = 2.0
    stages[t >= 5.0 - 1e-12, 0] = STAGE_ARRIVED
    # agent1: (-5,5,0) -> (0,5,0)，2.5 s 到达，路径 5 m
    positions[:, 1, 0] = -5.0 + 2.0 * np.minimum(t, 2.5)
    positions[:, 1, 1] = 5.0
    velocities[t[:-1] < 2.5 - 1e-12, 1, 0] = 2.0
    stages[t >= 2.5 - 1e-12, 1] = STAGE_ARRIVED
    # agent2/3: 静止，永不完成，路径 0
    positions[:, 2, :] = (20.0, 20.0, 0.0)
    positions[:, 3, :] = (20.0, 40.0, 0.0)
    goals = np.array(
        [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [100.0, 20.0, 0.0], [100.0, 40.0, 0.0]]
    )
    traj = build_traj(positions, velocities, stages)
    res = compute(traj, DuckScene(positions[0], goals))
    metrics = res.metrics

    assert res.agents["path_length"].to_numpy() == pytest.approx([10.0, 5.0, 0.0, 0.0], abs=1e-9)
    assert metrics["average_path_length"] == pytest.approx(3.75, abs=1e-9)
    assert metrics["completed_count_for_path_metrics"] == 2
    lower = 2.0 * 20.0 - 0.3  # 2R - eps_goal = 39.7
    expected_eff = float(np.mean([10.0 / lower - 1.0, 5.0 / lower - 1.0]))
    assert metrics["mean_path_efficiency_excess"] == pytest.approx(expected_eff, rel=1e-12)
    assert metrics["mean_path_over_40_minus_1"] == pytest.approx(3.75 / 40.0 - 1.0, rel=1e-12)


# =============================================================================
# 4. 安全完成：到达后被撞失去资格；静止尾段必须计入
# =============================================================================
def test_safe_completion_cancelled_by_later_hit_after_arrival():
    """A 在 t=8 s 完成并冻结；B 在 t≈9.317 s 进入 1 m 内 -> A 失去安全完成资格。"""
    n_steps = 200  # 20 s
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)

    positions[:, 0, 0] = -16.0 + 2.0 * np.minimum(t, 8.0)  # goal (0,0,0)，t=8 到达
    velocities[t[:-1] < 8.0 - 1e-12, 0, 0] = 2.0
    stages[t >= 8.0 - 1e-12, 0] = STAGE_ARRIVED
    positions[:, 1, 0] = 0.5
    positions[:, 1, 1] = 19.5 - 2.0 * np.minimum(t, 9.6)  # goal (0.5,0.3,0)，t=9.6 到达
    velocities[:, 1, 1] = np.where(t[:-1] < 9.6 - 1e-12, -2.0, 0.0)
    stages[t >= 9.6 - 1e-12, 1] = STAGE_ARRIVED

    goals = np.array([[0.0, 0.0, 0.0], [0.5, 0.3, 0.0]])
    res = compute(traj=build_traj(positions, velocities, stages), scene=DuckScene(positions[0], goals))

    a_arrival = float(res.agents.loc[0, "arrival_time"])
    assert a_arrival == pytest.approx(8.0, abs=1e-9)
    assert res.metrics["violation_event_count"] == 1
    ev = res.violation_events.iloc[0]
    # 侵犯开始晚于 A 到达 -> A 是“到达后被撞”
    assert ev["start_time"] > a_arrival
    # 解析：0.25 + (19.5-2t)^2 = d_eval^2
    d_eval = make_cfg().d_eval
    expected_start = (19.5 - float(np.sqrt(d_eval**2 - 0.25))) / 2.0
    assert ev["start_time"] == pytest.approx(expected_start, abs=1e-6)
    assert res.agents.loc[0, "safe_completed"] == False  # noqa: E712
    assert res.agents.loc[1, "safe_completed"] == False  # noqa: E712
    assert res.metrics["safe_completion_count"] == 0
    assert res.metrics["team_safe_success"] is False


def test_static_tail_extended_to_observation_window():
    """A/B 早期到达；C 在 t=60 到达并停在 A 0.583 m 处 -> 尾段侵犯延伸到 120 s。"""
    n_steps = 1200
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    n_agents = 3
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)

    # A: (-8,0,0) -> (0,0,0)，4 s 到达并冻结
    positions[:, 0, 0] = -8.0 + 2.0 * np.minimum(t, 4.0)
    velocities[t[:-1] < 4.0 - 1e-12, 0, 0] = 2.0
    stages[t >= 4.0 - 1e-12, 0] = STAGE_ARRIVED
    # B: (18,0,0) -> (10,0,0)，4 s 到达，与 A 相距 10 m
    positions[:, 1, 0] = 18.0 - 2.0 * np.minimum(t, 4.0)
    velocities[t[:-1] < 4.0 - 1e-12, 1, 0] = -2.0
    stages[t >= 4.0 - 1e-12, 1] = STAGE_ARRIVED
    # C: (0.5,60.3,0) -> (0.5,0.3,0)，1 m/s，60 s 到达；末态距 A 0.583 m
    positions[:, 2, 0] = 0.5
    positions[:, 2, 1] = 60.3 - 1.0 * np.minimum(t, 60.0)
    velocities[t[:-1] < 60.0 - 1e-12, 2, 1] = -1.0
    stages[t >= 60.0 - 1e-12, 2] = STAGE_ARRIVED

    goals = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.5, 0.3, 0.0]])
    traj = build_traj(positions, velocities, stages)
    res = compute(traj, DuckScene(positions[0], goals))
    metrics = res.metrics

    assert metrics["arrival_count"] == 3
    assert metrics["all_arrival_time"] == pytest.approx(60.0, abs=1e-9)
    assert metrics["violation_event_count"] == 1
    ev = res.violation_events.iloc[0]
    # 事件在静止尾段被正确延长到观察窗结束
    assert ev["end_time"] == pytest.approx(float(traj["t"][-1]), abs=1e-12)
    d_eval = make_cfg().d_eval
    expected_start = 60.3 - float(np.sqrt(d_eval**2 - 0.25))
    assert ev["start_time"] == pytest.approx(expected_start, abs=1e-6)
    assert ev["duration"] == pytest.approx(float(traj["t"][-1]) - expected_start, abs=1e-6)
    assert ev["min_distance"] == pytest.approx(float(np.hypot(0.5, 0.3)), rel=1e-9)
    assert metrics["min_separation"] == pytest.approx(float(np.hypot(0.5, 0.3)), rel=1e-9)
    # 只有 B 干净完成；A 到达后被撞，C 到达前参与侵犯
    assert res.agents["safe_completed"].tolist() == [False, True, False]
    assert metrics["safe_completion_count"] == 1
    assert metrics["team_safe_success"] is False
    # 尾段曲线仍报告侵犯
    assert res.min_distance_curve.iloc[-1]["violating_pair_count"] == 1
    assert res.min_distance_curve.iloc[-1]["min_distance"] == pytest.approx(
        float(np.hypot(0.5, 0.3)), rel=1e-9
    )


# =============================================================================
# 5. 场景 B：等待、放行后行程时间、阶段时间、未完成分类
# =============================================================================
def test_scenario_b_wait_release_and_phase_metrics():
    n_agents = 3
    n_steps = 40
    dt = DT
    t = np.arange(n_steps + 1, dtype=np.float64) * dt
    release = np.array([0.0, 1.0, 2.0])
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.zeros((n_steps + 1, n_agents), dtype=np.int8)
    for i in range(n_agents):
        rel = release[i]
        center_arr = rel + 1.0
        positions[:, i, 1] = 20.0 * i
        stages[t < rel - 1e-12, i] = STAGE_WAITING
        stages[(t >= rel - 1e-12) & (t < center_arr - 1e-12), i] = STAGE_TO_CENTER
        stages[(t >= center_arr - 1e-12) & (t < 3.9 - 1e-12), i] = STAGE_TO_GOAL
        stages[t >= 3.9 - 1e-12, i] = STAGE_ARRIVED
    goals = np.array([[20.0, 0.0, 0.0], [20.0, 20.0, 0.0], [20.0, 40.0, 0.0]])
    traj = build_traj(positions, velocities, stages)
    res = compute(
        traj,
        DuckScene(positions[0], goals),
        scenario="B",
        release_planned=np.array([0.0, 0.8, 1.6]),
        release_actual=release,
        center_times=np.array([1.0, 2.0, 3.0]),
    )
    metrics = res.metrics

    # wait_duration = release_actual；阶段等待时长与之一致
    assert res.agents["wait_duration"].to_numpy() == pytest.approx(release, abs=1e-12)
    assert res.agents["phase_wait_duration"].to_numpy() == pytest.approx(release, abs=1e-12)
    assert res.agents["phase_to_center_duration"].to_numpy() == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)
    assert metrics["waiting"]["mean_wait"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["waiting"]["total_wait"] == pytest.approx(3.0, abs=1e-12)
    assert metrics["waiting"]["mean_post_release_travel"] == pytest.approx(2.9, abs=1e-9)
    assert res.agents["release_rank"].tolist() == [0.0, 1.0, 2.0]
    assert res.agents["post_release_travel_time"].to_numpy() == pytest.approx(
        [3.9, 2.9, 1.9], abs=1e-9
    )
    # B 场景中心访问率有效
    assert metrics["center_crossing_count"] == 3
    assert metrics["center_crossing_rate"] == pytest.approx(1.0)
    assert metrics["uncompleted"] == {
        "not_released": 0,
        "not_center": 0,
        "center_done_not_arrived": 0,
    }


def test_uncompleted_classification_scenario_b():
    n_agents = 4
    n_steps = 20
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, :, 1] = 5.0 * np.arange(n_agents)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    stages[:, 0] = STAGE_TO_CENTER  # 已释放、未过中心
    stages[:, 1] = STAGE_TO_CENTER  # 已释放、未过中心
    # agent2 已过中心、未到达；agent3 未释放
    stages[:, 3] = STAGE_WAITING
    goals = np.full((n_agents, 3), 100.0)
    traj = build_traj(positions, velocities, stages)
    res = compute(
        traj,
        DuckScene(positions[0], goals),
        scenario="B",
        release_actual=np.array([0.0, 0.0, 0.0, 500.0]),
        release_planned=np.array([0.0, 0.0, 0.0, 500.0]),
        center_times=np.array([np.nan, np.nan, 1.0, np.nan]),
    )
    metrics = res.metrics
    assert metrics["arrival_count"] == 0
    assert metrics["uncompleted"] == {
        "not_released": 1,
        "not_center": 2,
        "center_done_not_arrived": 1,
    }


def test_center_crossing_rate_scenario_a_is_null():
    n_agents = 2
    n_steps = 10
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    positions[:, 1, 0] = 0.05  # 两机起始都在 0.1 m 中心球内
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
    goals = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    traj = build_traj(positions, velocities, stages)
    res_b = compute(
        traj,
        DuckScene(positions[0], goals),
        scenario="B",
        release_actual=np.zeros(n_agents),
        release_planned=np.zeros(n_agents),
        center_times=np.array([0.0, 0.0]),
    )
    assert res_b.metrics["center_crossing_count"] == 2
    assert res_b.metrics["center_crossing_rate"] == pytest.approx(1.0)
    res_a = compute(traj, DuckScene(positions[0], goals), scenario="A", center_times=np.array([0.0, 0.0]))
    assert res_a.metrics["center_crossing_rate"] is None
    assert res_a.metrics["center_crossing_count"] == 2


# =============================================================================
# 6. 中心进入一致性校验
# =============================================================================
def test_center_entry_consistency_check():
    n_steps = 30
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = 2.0 - 2.0 * t  # 首次进入 0.1 m 球于 t=0.95
    positions[:, 1, 0] = 50.0
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    velocities[:, 0, 0] = -2.0
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    scene = DuckScene(positions[0], np.array([[-20.0, 0, 0], [60.0, 0, 0]]))

    good = compute(traj, scene, center_times=np.array([0.95, np.nan]))
    assert good.metrics["consistency"]["center_entry_checked"] == 1
    assert good.metrics["consistency"]["center_entry_mismatch_count"] == 0
    assert good.metrics["consistency"]["center_entry_position_violations"] == 0

    bad = compute(traj, scene, center_times=np.array([0.5, np.nan]))
    consistency = bad.metrics["consistency"]
    assert consistency["center_entry_mismatch_count"] == 1
    assert consistency["center_entry_max_abs_diff_s"] == pytest.approx(0.45, abs=1e-6)
    assert consistency["center_entry_position_violations"] == 1  # ||p(0.5)|| = 1 m ≫ 0.1 m


# =============================================================================
# 7. 停滞规则
# =============================================================================
def test_stall_rule_and_low_progress_mobile():
    n_agents = 6
    n_steps = 120
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    positions = np.zeros((n_steps + 1, n_agents, 3), dtype=np.float64)
    velocities = np.zeros((n_steps, n_agents, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)

    # 0: 静止且永不完成 -> stalled
    positions[:, 0, :] = (0.0, 0.0, 0.0)
    # 1: 以 0.05 m/s 向远处 goal 前进：有路程、有进展
    positions[:, 1, 0] = 0.05 * t
    positions[:, 1, 1] = 20.0
    # 2: 绕圈徘徊：持续移动但 waypoint 距离无进展
    angle = 2.0 * np.pi * 0.1 * t
    positions[:, 2, 0] = 5.0 + 0.3 * np.cos(angle)
    positions[:, 2, 1] = 5.0 + 0.3 * np.sin(angle)
    # 3: 计划等待者（WAITING）静止 -> 排除
    positions[:, 3, :] = (30.0, 0.0, 0.0)
    stages[:, 3] = STAGE_WAITING
    # 4: 已到达者静止 -> 排除
    positions[:, 4, :] = (40.0, 0.0, 0.0)
    stages[:, 4] = STAGE_ARRIVED
    # 5: 静止但阶段每 3 s 切换 -> 5 s 窗口被重置，不算停滞
    positions[:, 5, :] = (50.0, 0.0, 0.0)
    block = (np.arange(n_steps + 1) // 30) % 2
    stages[:, 5] = np.where(block == 0, STAGE_TO_CENTER, STAGE_TO_GOAL)

    goals = np.array(
        [
            [100.0, 0.0, 0.0],
            [100.0, 20.0, 0.0],
            [5.0, 5.0, 0.0],
            [30.0, 100.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
        ]
    )
    traj = build_traj(positions, velocities, stages)
    res = compute(
        traj,
        DuckScene(positions[0], goals),
        scenario="B",
        release_actual=np.zeros(n_agents),
        release_planned=np.zeros(n_agents),
    )
    flags = res.agents[["stalled", "low_progress_mobile"]].to_numpy()
    assert flags.tolist() == [
        [True, False],   # 0 静止无进展
        [False, False],  # 1 有进展
        [False, True],   # 2 徘徊
        [False, False],  # 3 计划等待者被排除
        [False, False],  # 4 已到达被排除
        [False, False],  # 5 阶段切换重置窗口
    ]
    assert res.metrics["stall"] == {"stalled_agent_count": 1, "low_progress_mobile_count": 1}


# =============================================================================
# 8. 规划诊断与墙钟聚合
# =============================================================================
def test_planner_diagnostics_per_call_layout_and_planning_timing():
    n_agents = 3
    n_steps = 5
    traj, scene = static_case(n_agents, n_steps)
    active = np.zeros((n_steps, n_agents), dtype=bool)
    active[:, 0] = True
    active[:3, 1] = True
    traj["active"] = active
    calls = int(active.sum())  # 5 + 3 + 0 = 8
    traj["diag_step"] = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    traj["diag_agent_id"] = np.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=np.int64)
    traj["diag_sample_exhausted"] = np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
    traj["diag_used_fallback"] = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    traj["diag_solver_infeasible"] = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    traj["diag_constraint_violation_max"] = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0])
    traj["diag_initial_overlap_pairs"] = np.array([1, 0, 2, 0, 0, 0, 0, 0], dtype=np.int32)
    traj["diag_numerical_error"] = np.zeros(8, dtype=bool)
    traj["diag_candidate_count_tested"] = np.arange(8, dtype=np.int32)
    step_seconds = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    traj["planning_step_seconds"] = step_seconds

    res = compute(traj, scene)
    diag = res.metrics["planner_diagnostics"]
    assert diag["sample_exhausted_count"] == 3
    assert diag["solver_fallback_count"] == 2
    assert diag["solver_infeasible_count"] == 2
    assert diag["numerical_error_count"] == 0
    assert diag["constraint_violation_max"] == pytest.approx(0.3)
    assert diag["initial_overlap_pairs_total"] == 3
    assert diag["candidate_count_tested_total"] == 28
    assert diag["active_planning_calls"] == calls == 8
    assert res.metrics["consistency"]["diag_active_call_mismatch"] == 0

    planning = res.metrics["planning"]
    assert planning["calls"] == 8
    assert planning["mean_ms"] == pytest.approx(float(np.mean(step_seconds)) * 1000.0)
    assert planning["p95_ms"] == pytest.approx(float(np.percentile(step_seconds, 95.0)) * 1000.0)
    assert planning["max_ms"] == pytest.approx(5.0)
    assert planning["total_s"] == pytest.approx(float(np.sum(step_seconds)))
    assert planning["per_active_call_ms"] == pytest.approx(
        float(np.sum(step_seconds)) * 1000.0 / calls
    )


def test_planner_diagnostics_missing_defaults_to_zero():
    n_agents = 3
    n_steps = 4
    traj, scene = static_case(n_agents, n_steps)
    res = compute(traj, scene)
    diag = res.metrics["planner_diagnostics"]
    assert diag["sample_exhausted_count"] == 0
    assert diag["solver_fallback_count"] == 0
    assert diag["solver_infeasible_count"] == 0
    assert diag["numerical_error_count"] == 0
    assert diag["constraint_violation_max"] == 0.0
    assert diag["initial_overlap_pairs_total"] == 0
    assert diag["active_planning_calls"] == int(traj["active"].sum())
    assert res.metrics["consistency"]["diag_active_call_mismatch"] is None
    # 未提供墙钟：缺失值必须是 null，而不是 0
    planning = res.metrics["planning"]
    assert planning["mean_ms"] is None
    assert planning["p95_ms"] is None
    assert planning["max_ms"] is None
    assert planning["total_s"] is None
    assert planning["calls"] == int(traj["active"].sum())


# =============================================================================
# 9. 状态透传、合法/数值错误标志
# =============================================================================
def test_status_argument_passthrough(tmp_path):
    traj, scene = static_case(2, 10)
    res = compute(
        traj,
        scene,
        status={"completed": False, "timeout": False, "numerical_error": True, "valid": False},
    )
    assert res.metrics["status"] == {
        "completed": False,
        "timeout": False,
        "numerical_error": True,
        "valid": False,
    }
    out_dir = tmp_path / "run"
    write_metrics(out_dir, res)
    parsed = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert parsed["status"]["valid"] is False
    assert parsed["status"]["numerical_error"] is True


# =============================================================================
# 10. 离线可复现性
# =============================================================================
def test_offline_reproducibility_identical_metrics_and_files(tmp_path):
    traj, scene = line_arrival_case(5, 80, [0.4, 0.8, 1.2, 1.6, 2.0])
    first = compute(traj, scene)
    second = compute(traj, scene)
    assert first.metrics == second.metrics
    for name in ("agents", "violation_events", "boundary_contacts", "min_distance_curve", "arrival_curve"):
        left = getattr(first, name)
        right = getattr(second, name)
        assert left.equals(right), name

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_metrics(dir_a, first)
    write_metrics(dir_b, second)
    for name in (
        "metrics.json",
        "agents.csv",
        "violation_events.csv",
        "boundary_contacts.csv",
        "min_distance_curve.csv",
        "arrival_curve.csv",
    ):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes(), name


# =============================================================================
# 11. 场景稳定 ID 与到达时刻来源
# =============================================================================
def test_scene_stable_ids_used_in_outputs():
    n_steps = 5
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = 0.5  # 固定侵犯
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    traj = build_traj(positions, velocities, stages)
    goals = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
    res = compute(
        traj,
        DuckScene(positions[0], goals, agent_ids=np.array([10, 20]), scene_id="s-42"),
    )
    assert res.metrics["scene_id"] == "s-42"
    assert res.agents["agent_id"].tolist() == [10, 20]
    assert res.violation_events["agent_i"].tolist() == [10]
    assert res.violation_events["agent_j"].tolist() == [20]


def test_arrival_from_stages_and_provided_arrival_times():
    # 末位置已在 goal 容差球内（相距 0.1 m），但 stages 从未 ARRIVED -> 按规范不算到达
    n_steps = 20
    positions = np.zeros((n_steps + 1, 2, 3), dtype=np.float64)
    positions[:, :, 1] = 5.0 * np.arange(2)
    positions[:, 0, 0] = 0.1  # goal 在 (0,0,0)，末位置在 0.1 m 内
    velocities = np.zeros((n_steps, 2, 3), dtype=np.float64)
    stages = np.full((n_steps + 1, 2), STAGE_TO_GOAL, dtype=np.int8)
    goals = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    traj = build_traj(positions, velocities, stages)
    res = compute(traj, DuckScene(positions[0], goals))
    assert res.metrics["arrival_count"] == 0  # 不按最终位置推断
    assert res.metrics["consistency"]["arrival_checked"] == 0

    # 仿真提供 arrival_times（含 sub-step 记录）-> 作为权威值并交叉校验
    traj2 = build_traj(positions, velocities, stages, arrival_times=np.array([1.0, np.nan]))
    res2 = compute(traj2, DuckScene(positions[0], goals))
    assert res2.metrics["arrival_count"] == 1
    assert res2.agents.loc[0, "arrival_time"] == pytest.approx(1.0)
    assert res2.metrics["consistency"]["arrival_mismatch_count"] == 1  # stages 无 ARRIVED


# =============================================================================
# 12. 性能基准（默认跳过；`python -m pytest ... --bench` 运行）
# =============================================================================
def _bench_case(n_agents, n_steps, *, dense):
    """构造 100 机 × 1200 步基准轨迹：sparse=稀疏航迹，dense=全部机对汇聚并侵犯。"""
    rng = np.random.default_rng(20261030)
    t = np.arange(n_steps + 1, dtype=np.float64) * DT
    if dense:
        starts = np.zeros((n_agents, 3), dtype=np.float64)
        angle = 2.0 * np.pi * np.arange(n_agents) / n_agents
        starts[:, 0] = 20.0 * np.cos(angle)
        starts[:, 1] = 20.0 * np.sin(angle)
        goals = -starts
        travel = np.minimum(t, 10.0) / 10.0
        positions = (
            starts[None, :, :] * (1.0 - travel)[:, None, None]
            + goals[None, :, :] * travel[:, None, None]
        )
        velocities = np.broadcast_to(
            ((goals - starts) / 10.0)[None, :, :], (n_steps, n_agents, 3)
        ).copy()
        stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
        stages[t >= 10.0 - 1e-12] = STAGE_ARRIVED
    else:
        starts = rng.uniform(-30.0, 30.0, size=(n_agents, 3))
        goals = rng.uniform(-30.0, 30.0, size=(n_agents, 3))
        direction = goals - starts
        norm = np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
        direction = direction / norm
        speeds = rng.uniform(1.0, 2.0, size=n_agents)
        stop_time = np.linalg.norm(goals - starts, axis=-1) / speeds
        travel = np.minimum(t[:, None], stop_time[None, :])[:, :, None]
        positions = starts[None, :, :] + travel * speeds[None, :, None] * direction[None, :, :]
        velocities = np.broadcast_to(
            speeds[None, :, None] * direction[None, :, :], (n_steps, n_agents, 3)
        ).copy()
        stages = np.full((n_steps + 1, n_agents), STAGE_TO_GOAL, dtype=np.int8)
        stages[t[:, None] >= stop_time[None, :] - 1e-12] = STAGE_ARRIVED
    goals_arr = np.asarray(goals)
    traj = build_traj(positions, velocities, stages)
    return traj, DuckScene(starts, goals_arr)


@pytest.mark.bench
@pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense_all_pairs"])
def test_bench_recompute_100_agents_1200_steps(dense, capsys):
    """CONTRACTS §11.1 性能：100 机 × 1200 步离线重算 < 20 s（单线程）。"""
    import time

    traj, scene = _bench_case(100, 1200, dense=dense)
    center_times = np.full(100, np.nan)
    release = np.zeros(100)
    t0 = time.perf_counter()
    res = compute(
        traj,
        scene,
        scenario="B",
        center_times=center_times,
        release_actual=release,
        release_planned=release,
    )
    elapsed = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[bench] 100x1200 dense={dense}: recompute {elapsed:.3f} s")
    assert elapsed < 20.0
    assert res.metrics["n_agents"] == 100
    if dense:
        assert res.metrics["violating_pair_count"] == 4950
        assert res.metrics["violation_event_count"] == 4950
