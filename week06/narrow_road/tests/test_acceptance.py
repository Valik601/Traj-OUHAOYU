"""实验 2（速度相关接受模型）的单元测试。"""
from __future__ import annotations

import pytest

from narrowroad.acceptance import allowed_speed, predict_outcome, predicted_boundary
from narrowroad.common import (
    acceptance_clearance,
    clearance_speed_cap,
    draw_initial_conditions,
    load_config,
)
from narrowroad.decision import decide_pedestrian
from narrowroad.simulate import RunSpec, build_world, simulate


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def ics(cfg):
    return draw_initial_conditions(cfg)


# ---------------------------------------------------------------------------
# 接受关系
# ---------------------------------------------------------------------------
def test_acceptance_clearance(cfg):
    s = cfg["rules"]["S_mod"]
    assert acceptance_clearance(s, cfg, 0.0) == pytest.approx(0.02)
    assert acceptance_clearance(s, cfg, 1.0) == pytest.approx(0.05)
    assert acceptance_clearance(s, cfg, 9.0) == pytest.approx(0.02 + 0.03 * 9.0)
    # 固定阈值的旧规则与车速无关
    assert acceptance_clearance(cfg["rules"]["R1"], cfg, 0.0) == pytest.approx(0.30)
    assert acceptance_clearance(cfg["rules"]["R1"], cfg, 9.0) == pytest.approx(0.30)


def test_clearance_speed_cap_with_factor(cfg):
    v0 = cfg["car"]["v0_mps"]
    # R1 没有 speed_factor（默认 1.0）
    assert clearance_speed_cap(cfg["rules"]["R1"], 0.10, v0) == pytest.approx(1.0)
    # S_mod: 0.9 * (0.10 - 0.02) / 0.03 = 2.4
    assert clearance_speed_cap(cfg["rules"]["S_mod"], 0.10, v0) == pytest.approx(2.4)
    # S_cons: 0.9 * (0.05 - 0.05) / 0.05 = 0
    assert clearance_speed_cap(cfg["rules"]["S_cons"], 0.05, v0) == pytest.approx(0.0)


def test_predicted_boundaries(cfg):
    b = {r: predicted_boundary(cfg, r) for r in ("R1", "S_cons", "S_mod", "S_lib")}
    assert b["R1"]["l_crit_m"] == pytest.approx(2.90)
    assert b["S_cons"]["l_crit_m"] == pytest.approx(2.90)
    assert b["S_mod"]["l_crit_m"] == pytest.approx(2.84)
    assert b["S_lib"]["l_crit_m"] == pytest.approx(2.82)
    # R1 严格大于、S 规则大于等于
    assert predict_outcome(cfg, "R1", 2.90) is False
    assert predict_outcome(cfg, "S_cons", 2.90) is True
    assert predict_outcome(cfg, "S_mod", 2.84) is True
    assert predict_outcome(cfg, "S_mod", 2.82) is False
    assert predict_outcome(cfg, "S_lib", 2.82) is True
    assert predict_outcome(cfg, "S_lib", 2.80) is False
    assert allowed_speed(cfg, "S_mod", 3.00) == pytest.approx(2.4)


# ---------------------------------------------------------------------------
# 行人决策：接受 → 继续走；不接受 → 速度降到 0
# ---------------------------------------------------------------------------
def _world_with(cfg, width: float, ped_y: float, car_v_target: float, gap: float):
    ic = draw_initial_conditions(cfg)
    world = build_world(RunSpec("t", width, "S_mod", "sfm", ic, save_trace=False), cfg)
    world.rule = cfg["rules"]["S_mod"]
    car = world.car
    ped = world.peds[0]
    car.x, car.y = 200.0, 0.0
    car.vx = -car_v_target
    car.v_target = car_v_target
    ped.x = car.x - 2.5 - gap
    ped.y = ped_y
    ped.vx = ped.v0
    return world, ped, cfg["rules"]["S_mod"]


def test_ped_accepts_when_planned_speed_is_low(cfg):
    # L=3.0，行人贴边 c=0.10，车辆计划速度 2.4 m/s → 2.4 <= (0.10-0.02)/0.03 → 接受
    world, ped, rule = _world_with(cfg, 3.0, 1.25, 2.4, gap=20.0)
    d = decide_pedestrian(ped, world, cfg, rule)
    assert d.mode == "PASS"
    assert d.v_target == pytest.approx(ped.v0)


def test_ped_stops_when_planned_speed_is_high(cfg):
    # 同一几何，但车辆计划速度 9 m/s → 需要 0.29 m 余量 > 0.10 → 让行
    world, ped, rule = _world_with(cfg, 3.0, 1.25, 9.0, gap=20.0)
    d = decide_pedestrian(ped, world, cfg, rule)
    assert d.mode == "STOP"
    assert d.v_target == pytest.approx(0.0)          # 速度可以降到 0
    assert d.y_target == pytest.approx(ped.side_sign * world.y_limit)   # 同时继续贴向路缘


def test_ped_stops_when_floor_not_met(cfg):
    # L=2.82，c=0.01，即使车停住（0 m/s）也需要 0.02 m → 不接受，停车等待
    world, ped, rule = _world_with(cfg, 2.82, 1.16, 0.0, gap=20.0)
    d = decide_pedestrian(ped, world, cfg, rule)
    assert d.mode == "STOP"
    assert d.v_target == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 仿真层结果
# ---------------------------------------------------------------------------
CASES = [
    ("R1", 3.00, "wiedemann", "PASS"),
    ("S_mod", 3.00, "wiedemann", "PASS"),
    ("S_mod", 2.90, "wiedemann", "PASS"),      # R1 在同一宽度僵持
    ("R1", 2.90, "wiedemann", "DEADLOCK"),
    ("S_mod", 2.84, "wiedemann", "PASS"),
    ("S_mod", 2.82, "wiedemann", "DEADLOCK"),
    ("S_lib", 2.82, "wiedemann", "PASS"),
    ("S_lib", 2.80, "wiedemann", "DEADLOCK"),
]


@pytest.mark.parametrize("rule, width, model, expected", CASES)
def test_outcomes(cfg, ics, rule, width, model, expected):
    res, _, _ = simulate(
        RunSpec("t", width, rule, model, ics, save_trace=False, t_max_s=420.0), cfg
    )
    assert res.outcome == expected
    assert res.collision_events == 0
    assert res.min_body_distance_m > -1e-6


def test_new_model_is_cheaper_and_lets_pedestrians_walk(cfg, ics):
    r1, _, _ = simulate(RunSpec("a", 3.0, "R1", "wiedemann", ics, save_trace=False), cfg)
    sm, tr, _ = simulate(
        RunSpec("b", 3.0, "S_mod", "wiedemann", ics, save_trace=True), cfg
    )
    assert r1.outcome == sm.outcome == "PASS"
    assert sm.car_delay_s < r1.car_delay_s
    assert sm.ped_stop_time_total_s < r1.ped_stop_time_total_s
    # 行人的速度可以降到 0（让行），之后又继续走
    assert tr is not None
    min_speed = min(float(g["speed"].min()) for name, g in tr.groupby("agent") if name != "CAR")
    assert min_speed == pytest.approx(0.0, abs=1e-6)
    assert sm.min_body_distance_m >= 0.10 - 1e-6


def test_force_scaling_ablation_is_second_order(cfg, ics):
    """SFM 力缩放开关（S_mod vs S_mod_nf）不改变定性结论、定量差异很小。"""
    a, _, _ = simulate(RunSpec("a", 3.0, "S_mod", "sfm", ics, save_trace=False), cfg)
    b, _, _ = simulate(RunSpec("b", 3.0, "S_mod_nf", "sfm", ics, save_trace=False), cfg)
    assert a.outcome == b.outcome == "PASS"
    assert abs(a.car_delay_s - b.car_delay_s) < 0.5


def test_old_rules_unchanged(cfg, ics):
    """新模型加入后，实验 1 的 R0/R1 结论保持不变。"""
    r1 = simulate(RunSpec("r1", 3.0, "R1", "sfm", ics, save_trace=False), cfg)[0]
    r0 = simulate(RunSpec("r0", 3.0, "R0", "sfm", ics, save_trace=False), cfg)[0]
    assert r1.outcome == "PASS"
    assert r1.car_delay_s == pytest.approx(29.766667, abs=1e-3)
    assert r0.outcome == "DEADLOCK"

