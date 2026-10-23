"""几何 / 规则阈值 / 解析判据的单元测试。"""
from __future__ import annotations

import math

import pytest

from narrowroad.common import (
    analytic_predictions,
    clearance_available,
    clearance_speed_cap,
    draw_initial_conditions,
    lateral_clearance,
    longitudinal_body_gap,
    load_config,
    required_clearance,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_lateral_clearance_car_vs_ped(cfg):
    car_w = cfg["car"]["width_m"]
    ped_w = cfg["pedestrians"]["width_m"]
    # 行人贴到 3.0 m 路的路缘：|y| = 1.25，车在 y = 0
    c = lateral_clearance(-1.25, 0.0, ped_w, car_w)
    assert c == pytest.approx(0.10, abs=1e-12)
    # 行人还在路中线
    c0 = lateral_clearance(0.0, 0.0, ped_w, car_w)
    assert c0 == pytest.approx(-1.15, abs=1e-12)


def test_longitudinal_body_gap():
    # 行人朝 +x 走：a 在后(x=90)、b 在前(x=100)
    g = longitudinal_body_gap(x_a=90.0, x_b=100.0, length_a=0.5, length_b=0.5, heading_a=1.0)
    assert g == pytest.approx(9.5)
    # 车辆朝 -x 驶来：a 在 x=100，行人 b 在 x=90（在车辆前方）
    g_car = longitudinal_body_gap(x_a=100.0, x_b=90.0, length_a=4.5, length_b=0.5, heading_a=-1.0)
    assert g_car == pytest.approx(7.5)


@pytest.mark.parametrize(
    "width, expected",
    [(4.0, 0.60), (3.5, 0.35), (3.2, 0.20), (3.0, 0.10), (2.9, 0.05), (2.8, 0.00)],
)
def test_clearance_available(cfg, width, expected):
    c = clearance_available(width, cfg["car"]["width_m"], cfg["pedestrians"]["width_m"])
    assert c == pytest.approx(expected, abs=1e-12)


def test_required_clearance(cfg):
    r0 = cfg["rules"]["R0"]
    r1 = cfg["rules"]["R1"]
    assert required_clearance(r0, 0.0) == pytest.approx(0.30)
    assert required_clearance(r1, 0.0) == pytest.approx(0.05)
    assert required_clearance(r1, 2.0) == pytest.approx(0.05 + 0.05 * 2.0)


def test_clearance_speed_cap(cfg):
    r0 = cfg["rules"]["R0"]
    r1 = cfg["rules"]["R1"]
    v0 = cfg["car"]["v0_mps"]
    # R0：固定阈值，低于阈值一律 0
    assert clearance_speed_cap(r0, 0.29, v0) == 0.0
    assert clearance_speed_cap(r0, 0.30, v0) == v0
    # R1：余量 → 允许速度，线性
    assert clearance_speed_cap(r1, 0.05, v0) == pytest.approx(0.0)
    assert clearance_speed_cap(r1, 0.10, v0) == pytest.approx(1.0)
    assert clearance_speed_cap(r1, 0.20, v0) == pytest.approx(3.0)
    assert clearance_speed_cap(r1, 0.60, v0) == pytest.approx(11.0)


def test_analytic_prediction_thresholds(cfg):
    widths = [4.0, 3.5, 3.2, 3.0, 2.9, 2.8]
    preds = {row["width_m"]: row for row in analytic_predictions(cfg, widths)}
    # R0 的临界宽度 = 2*(0.30+0.90+0.50) = 3.40 m
    assert preds[4.0]["R0_feasible"] is True
    assert preds[3.5]["R0_feasible"] is True
    assert preds[3.2]["R0_feasible"] is False
    assert preds[3.0]["R0_feasible"] is False
    # R1 的临界宽度 = 2*(0.05+0.90+0.50) = 2.90 m
    assert preds[3.0]["R1_feasible"] is True
    assert preds[2.9]["R1_feasible"] is False
    assert preds[2.8]["R1_feasible"] is False
    # 允许速度随宽度单调增
    speeds = [preds[w]["R1_v_allow_mps"] for w in [2.8, 3.0, 3.2, 3.5, 4.0]]
    assert speeds == sorted(speeds)


def test_initial_conditions_deterministic(cfg):
    a = draw_initial_conditions(cfg)
    b = draw_initial_conditions(cfg)
    assert a == b
    assert len(a) == cfg["pedestrians"]["count"]
    xs = sorted(row["x_m"] for row in a)
    for lo, hi in zip(xs, xs[1:]):
        assert hi - lo >= cfg["pedestrians"]["min_spacing_m"]
    # P1 是离车辆最近（x 最大）的行人
    assert a[0]["x_m"] == max(row["x_m"] for row in a)
    assert all(0.9 <= row["v0_mps"] <= 1.5 for row in a)
