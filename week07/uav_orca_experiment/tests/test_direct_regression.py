# OWNER: lead
"""Direct 解析回归（handoff §8）：精确端点 20 s / 40 m 与默认任务 19.9 s / 39.8 m。"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench import dynamics, navigation
from uavbench.config import apply_overrides, load_config
from uavbench.initialization import build_base_fibonacci_scene, build_scene
from uavbench.simulation import run_simulation


def _path_lengths(result):
    return dynamics.path_lengths(result.positions)


def _grid_min_separation(positions: np.ndarray) -> float:
    n = positions.shape[1]
    best = np.inf
    for k in range(positions.shape[0]):
        p = positions[k]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        best = min(best, float(np.min(d)))
    return best


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_direct_exact_endpoint_regression(scenario):
    cfg = apply_overrides(load_config(), {"simulation.goal_tolerance": 1e-8})
    scene = build_base_fibonacci_scene(cfg)
    res = run_simulation(cfg, scene, "direct", scenario)
    assert res.valid and not res.numerical_error
    arrivals = res.arrival_times
    assert np.all(np.isfinite(arrivals)), "exact endpoint regression: all agents must arrive"
    assert np.allclose(arrivals, 20.0, rtol=0.0, atol=1e-9)
    paths = _path_lengths(res)
    assert np.allclose(paths, 40.0, rtol=0.0, atol=1e-9)
    # 全体经过原点 → 网格最小机间距≈0
    assert _grid_min_separation(res.positions) < 1e-9
    if scenario == "B":
        assert np.all(np.isfinite(res.center_entry_times))
        assert np.allclose(res.center_entry_times, 9.95, rtol=0.0, atol=1e-9)
        # 控制阶段切换发生在 10.0 s 网格
        for i in range(cfg.simulation.n_agents):
            first_goal = int(np.argmax(res.stages[:, i] == navigation.STAGE_TO_GOAL))
            assert res.t[first_goal] == pytest.approx(10.0, abs=1e-12)
    else:
        assert np.all(np.isnan(res.center_entry_times))


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_direct_default_task_regression(scenario):
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    res = run_simulation(cfg, scene, "direct", scenario)
    assert res.valid
    arrivals = res.arrival_times
    assert np.all(np.isfinite(arrivals))
    assert np.allclose(arrivals, 19.9, rtol=0.0, atol=1e-9)
    assert np.allclose(_path_lengths(res), 39.8, rtol=0.0, atol=1e-9)
    if scenario == "B":
        assert np.allclose(res.center_entry_times, 9.95, rtol=0.0, atol=1e-9)
        for i in range(cfg.simulation.n_agents):
            first_goal = int(np.argmax(res.stages[:, i] == navigation.STAGE_TO_GOAL))
            assert res.t[first_goal] == pytest.approx(10.0, abs=1e-12)
    assert _grid_min_separation(res.positions) < 1e-6


def test_exact_and_default_differ_as_documented():
    """默认容差与精确端点必须得到不同数值（不能为测试偷改规则）。"""
    cfg_exact = apply_overrides(load_config(), {"simulation.goal_tolerance": 1e-8})
    scene = build_base_fibonacci_scene(cfg_exact)
    a_exact = run_simulation(cfg_exact, scene, "direct", "A").arrival_times
    a_default = run_simulation(load_config(), scene, "direct", "A").arrival_times
    assert abs(float(np.mean(a_exact)) - 20.0) < 1e-12
    assert abs(float(np.mean(a_default)) - 19.9) < 1e-12


def test_direct_perturbed_scene_arrival_symmetry(default_cfg):
    """正式扰动场景下 Direct 仍全部到达（每机航程 40 m 几何下界附近）。"""
    scene = build_scene(default_cfg, 1000)
    res = run_simulation(default_cfg, scene, "direct", "A")
    assert np.all(np.isfinite(res.arrival_times))
    paths = _path_lengths(res)
    assert np.allclose(paths, 39.8, rtol=0.0, atol=1e-9)
    assert np.allclose(res.arrival_times, 19.9, rtol=0.0, atol=1e-9)


def test_arrived_agents_are_frozen_and_stationary():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    res = run_simulation(cfg, scene, "direct", "A")
    arrival_steps = np.argmax(res.stages == navigation.STAGE_ARRIVED, axis=0)
    assert np.all(arrival_steps > 0)
    for i in range(cfg.simulation.n_agents):
        k = int(arrival_steps[i])
        assert res.frozen[k, i]
        assert np.all(res.positions[k:, i] == res.positions[k, i])


def _metrics_for(cfg, scene, scenario):
    from uavbench.metrics import compute_metrics

    res = run_simulation(cfg, scene, "direct", scenario)
    metrics = compute_metrics(
        res.to_npz_dict(),
        scene,
        cfg,
        model="direct",
        scenario=scenario,
        seed=int(scene.seed),
        scene_hash="unit",
        config_hash="unit",
        center_entry_times=res.center_entry_times,
        center_entry_steps=res.center_entry_steps,
        release_planned=res.release_planned,
        release_actual=res.release_actual,
        status={
            "completed": bool(np.all(np.isfinite(res.arrival_times))),
            "timeout": False,
            "numerical_error": False,
            "valid": True,
        },
    )
    return res, metrics


@pytest.mark.parametrize("scenario", ["A", "B"])
def test_direct_exact_endpoint_metrics_regression(scenario):
    """handoff §8：精确端点回归必须出现 d_min≈0 与全部 4950 个机对侵犯。"""
    cfg = apply_overrides(load_config(), {"simulation.goal_tolerance": 1e-8})
    scene = build_base_fibonacci_scene(cfg)
    res, metrics_result = _metrics_for(cfg, scene, scenario)
    m = metrics_result.metrics
    assert m["arrival_rate"] == pytest.approx(1.0)
    assert m["min_separation"] == pytest.approx(0.0, abs=1e-12)
    assert m["violating_pair_count"] == 4950
    assert m["violation_event_count"] == 4950
    assert m["safe_completion_rate"] == pytest.approx(0.0)
    assert m["team_safe_success"] is False
    # 首次侵犯早于“全体经过原点”时刻（不同机对斜穿路径提前接近；不做精确值断言）
    assert 0.0 < m["first_violation_time"] <= 10.0 + 1e-9
    assert m["all_arrival_time"] == pytest.approx(20.0, abs=1e-9)
    assert m["t90"] == pytest.approx(20.0, abs=1e-9)
    if scenario == "B":
        assert m["center_crossing_rate"] == pytest.approx(1.0)
    else:
        assert m["center_crossing_rate"] is None


def test_direct_default_metrics_regression():
    cfg = load_config()
    scene = build_base_fibonacci_scene(cfg)
    res, metrics_result = _metrics_for(cfg, scene, "B")
    m = metrics_result.metrics
    assert m["arrival_rate"] == pytest.approx(1.0)
    assert m["violating_pair_count"] == 4950
    assert m["min_separation"] == pytest.approx(0.0, abs=1e-12)
    assert m["all_arrival_time"] == pytest.approx(19.9, abs=1e-9)
    assert m["average_path_length"] == pytest.approx(39.8, abs=1e-9)
    assert m["center_crossing_rate"] == pytest.approx(1.0)
