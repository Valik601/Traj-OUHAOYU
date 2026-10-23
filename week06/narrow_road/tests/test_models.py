"""决策层 / 控制层（SFM、Wiedemann）的单元测试。"""
from __future__ import annotations

import math

import pytest

from narrowroad.common import draw_initial_conditions, load_config
from narrowroad.controllers import (
    _free_accel,
    control,
    sfm_force,
    wiedemann_longitudinal,
)
from narrowroad.decision import decide_car, decide_pedestrian
from narrowroad.simulate import RunSpec, build_world


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_world(cfg, width=3.0):
    ic = draw_initial_conditions(cfg)
    spec = RunSpec("unit", width, "R0", "sfm", ic, save_trace=False)
    world = build_world(spec, cfg)
    # 把行人放到其它地方，避免默认位置干扰
    for i, p in enumerate(world.peds):
        p.x, p.y, p.vx, p.vy = 40.0 + 10.0 * i, 0.0, p.v0, 0.0
    world.car.x, world.car.y = 300.0, 0.0
    world.car.vx = -cfg["car"]["v0_mps"]
    return world


# ---------------------------------------------------------------------------
# 基础控制律
# ---------------------------------------------------------------------------
def test_free_accel_regimes():
    a_up = _free_accel(1.0, 9.0, 2.0, 2.5, 0.2, 1.5)
    a_down = _free_accel(10.0, 9.0, 2.0, 2.5, 0.2, 1.5)
    a_stop_target = _free_accel(1.0, 0.0, 2.0, 2.5, 0.2, 1.5)
    a_stopped = _free_accel(0.0, 0.0, 2.0, 2.5, 0.2, 1.5)
    assert a_up > 0.0
    assert a_down < 0.0
    assert a_stop_target == pytest.approx(-2.5)   # 期望停车时必须真的刹车
    assert a_stopped == 0.0


def test_sfm_pushes_pedestrian_away_from_vehicle(cfg):
    world = make_world(cfg)
    ped = world.peds[0]
    world.car.x, world.car.y = 100.0, 0.0
    ped.x, ped.y, ped.vx, ped.vy = 100.0, -1.25, 0.0, 0.0
    f = sfm_force(ped, world, cfg)
    # 行人位于车辆左侧：斥力应把它继续往 -y 推（远离车身）
    assert f[1] < 0.0
    assert abs(f[0]) < 2.0


def test_sfm_vehicle_brakes_for_pedestrian_in_path(cfg):
    world = make_world(cfg)
    world.car.x, world.car.y = 100.0, 0.0
    world.peds[0].x, world.peds[0].y = 97.5, 0.0
    f = sfm_force(world.car, world, cfg)
    # 车辆朝 -x 行驶；行人在正前方 -> 世界坐标斥力应指向 +x（减速）
    assert f[0] > 0.0
    assert f[1] == pytest.approx(0.0)


def test_wiedemann_free_driving_and_braking(cfg):
    world = make_world(cfg)
    car = world.car
    ped = world.peds[0]
    # 前方无人：自由行驶 -> 加速
    ped.x, ped.y = 20.0, 0.0
    car.x, car.y, car.vx = 100.0, 0.0, -5.0
    from narrowroad.decision import Decision

    a = wiedemann_longitudinal(car, world, Decision("FREE", 9.0, 0.0), cfg)
    assert a > 0.0
    # 行人就在路径内、很近：应明显减速
    ped.x, ped.y = car.x - 3.0, 0.0
    a_brake = wiedemann_longitudinal(car, world, Decision("YIELD", 0.0, 0.0), cfg)
    assert a_brake < 0.0
    # 已停住且贴近：保持不动
    car.vx = 0.0
    a_hold = wiedemann_longitudinal(car, world, Decision("YIELD", 0.0, 0.0), cfg)
    assert a_hold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 决策层
# ---------------------------------------------------------------------------
def _ped_decision(world, cfg, rule_name, car_gap, ped_y, ped_x=None):
    ped = world.peds[0]
    world.car.x, world.car.y = 200.0, 0.0
    ped.y = ped_y
    ped.x = world.car.x - 2.25 - 0.25 - car_gap  # 使 ped.gap_to(car) == car_gap
    if ped_x is not None:
        ped.x = ped_x
    return decide_pedestrian(ped, world, cfg, cfg["rules"][rule_name])


def test_pedestrian_modes(cfg):
    world = make_world(cfg)
    # 车辆很远 -> WALK
    d = _ped_decision(world, cfg, "R0", car_gap=100.0, ped_y=-1.25)
    assert d.mode == "WALK"
    # 车辆进入感知范围但未贴近，且横向余量不足 -> SIDESTEP
    d = _ped_decision(world, cfg, "R0", car_gap=15.0, ped_y=-1.25)
    assert d.mode == "SIDESTEP"
    assert d.v_target < world.peds[0].v0
    assert d.y_target == pytest.approx(-1.25)
    # 车辆贴近 -> WAIT；R0 冻住侧向，R1 继续移向路缘
    world = make_world(cfg)
    d_r0 = _ped_decision(world, cfg, "R0", car_gap=3.0, ped_y=-0.6)
    assert d_r0.mode == "WAIT"
    assert d_r0.v_target == 0.0
    assert d_r0.y_target == pytest.approx(-0.6)
    world = make_world(cfg)
    d_r1 = _ped_decision(world, cfg, "R1", car_gap=3.0, ped_y=-0.6)
    assert d_r1.mode == "WAIT"
    assert d_r1.y_target == pytest.approx(-1.25)


def test_car_clearance_rules(cfg):
    world = make_world(cfg)
    car = world.car
    ped = world.peds[0]
    car.x, car.y, car.vx = 100.0, 0.0, 0.0
    ped.vx = 0.0
    # 行人贴在路缘：3.0 m 路 -> c = 0.10；把车放到停车线（净距 3.0 m）处
    ped.y = -1.25
    ped.x = car.x - 2.25 - 0.5 - 3.0 + 0.25  # 令 car.gap_to(ped) = 3.0
    d_r0 = decide_car(car, world, cfg, cfg["rules"]["R0"])
    d_r1 = decide_car(car, world, cfg, cfg["rules"]["R1"])
    assert d_r0.v_target == pytest.approx(0.0)           # R0: 0.10 < 0.30 -> 必须停
    assert d_r1.v_target == pytest.approx(1.0, abs=0.2)  # R1: 允许约 1 m/s 蹭行
    # 行人离得很远时不施加限制
    ped.x = car.x + 60.0
    d = decide_car(car, world, cfg, cfg["rules"]["R0"])
    assert d.v_target == pytest.approx(cfg["car"]["v0_mps"])


def test_control_dispatch(cfg):
    world = make_world(cfg)
    from narrowroad.decision import Decision

    ped = world.peds[0]
    world.car.x = 100.0
    ped.x, ped.y = 100.0, -1.25
    ax, ay, a_long = control(ped, world, Decision("WALK", ped.v0, -1.25), "sfm", cfg)
    assert math.isfinite(ax) and math.isfinite(ay)
    with pytest.raises(ValueError):
        control(ped, world, Decision("WALK", ped.v0, -1.25), "unknown", cfg)
