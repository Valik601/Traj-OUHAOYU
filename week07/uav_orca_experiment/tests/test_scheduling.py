# OWNER: lead
"""分时放行验收（handoff §5.5、§8 调度测试）。"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

import numpy as np
import pytest

from uavbench.config import apply_overrides, load_config
from uavbench.initialization import build_scene
from uavbench.scheduling import (
    ORDER_FIXED_ID,
    ORDER_RANDOM_0,
    ORDER_RANDOM_1,
    build_release_table,
    release_mask,
)


@pytest.fixture(scope="module")
def scene1000():
    cfg = load_config()
    return cfg, build_scene(cfg, 1000)


def _ceil_tick(planned: float, dt: float) -> int:
    q = Decimal(str(planned)) / Decimal(str(dt))
    return int(q.to_integral_value(rounding=ROUND_CEILING))


def test_quantization_070_080_and_last_74_3(scene1000):
    cfg, scene = scene1000
    tab = build_release_table(cfg, scene)
    dt = cfg.experiment.dt
    assert tab.planned.max() == pytest.approx(74.25, abs=1e-12)
    assert tab.actual.max() == pytest.approx(74.3, abs=1e-12)
    assert np.all(tab.release_tick == np.asarray([_ceil_tick(p, dt) for p in tab.planned]))
    assert np.allclose(tab.actual, tab.release_tick * dt, rtol=0.0, atol=1e-12)
    ordered = np.sort(tab.actual)
    gaps = np.diff(ordered)
    unique = set(np.round(gaps, 10).tolist())
    assert unique <= {0.7, 0.8}
    assert len(unique) == 2
    # 逐个从绝对计划时刻向上量化，而不是累加四舍五入后的间隔
    assert np.all(gaps >= 0.7 - 1e-12)
    assert np.all(gaps <= 0.8 + 1e-12)


def test_order_streams_rank_and_mask(scene1000):
    cfg, scene = scene1000
    tab0 = build_release_table(cfg, scene)
    assert tab0.order == ORDER_RANDOM_0
    assert sorted(tab0.ranks.tolist()) == list(range(scene.n_agents))
    assert len(set(tab0.ranks.tolist())) == scene.n_agents

    cfg_fixed = apply_overrides(cfg, {"scheduling.order": ORDER_FIXED_ID})
    tab_fixed = build_release_table(cfg_fixed, scene)
    assert np.array_equal(tab_fixed.ranks, np.arange(scene.n_agents))
    assert np.array_equal(np.sort(tab_fixed.actual), np.sort(tab0.actual))  # 时间值集合相同
    assert not np.array_equal(tab_fixed.ranks, tab0.ranks)  # 但分配不同

    cfg_1 = apply_overrides(cfg, {"scheduling.order": ORDER_RANDOM_1})
    tab1 = build_release_table(cfg_1, scene)
    assert tab1.order == ORDER_RANDOM_1
    assert not np.array_equal(tab1.ranks, tab0.ranks)
    assert np.array_equal(np.sort(tab1.actual), np.sort(tab0.actual))

    assert np.array_equal(release_mask(tab0, 0.0), tab0.actual <= 1e-12)
    assert np.all(release_mask(tab0, 120.0))


def test_zero_delta_degenerates_to_simultaneous(scene1000):
    cfg, scene = scene1000
    cfg0 = apply_overrides(cfg, {"scheduling.delta": 0.0})
    tab = build_release_table(cfg0, scene)
    assert np.all(tab.actual == 0.0)
    assert np.all(release_mask(tab, 0.0))


@pytest.mark.parametrize(
    "overrides,expected_last",
    [
        ({"scheduling.delta": 0.5}, 49.5),
        ({"scheduling.delta": 1.0}, 99.0),
        ({"scheduling.delta": 0.75, "experiment.dt": 0.05}, 74.25),
        ({"scheduling.delta": 0.75, "experiment.dt": 0.2}, 74.4),
    ],
)
def test_delta_and_dt_variants_requantize(scene1000, overrides, expected_last):
    cfg, scene = scene1000
    cfg2 = apply_overrides(cfg, overrides)
    tab = build_release_table(cfg2, scene)
    assert tab.planned.max() == pytest.approx(99 * float(cfg2.scheduling.delta), abs=1e-12)
    assert tab.actual.max() == pytest.approx(expected_last, abs=1e-12)
    if cfg2.experiment.dt == 0.05:
        assert np.allclose(tab.actual, tab.planned, atol=1e-12)


def test_table_independent_of_planner_and_repeatable(scene1000):
    """C1/M4 必须复用完全相同的放行表：同一配置同一场景构造两次逐位相等。"""
    cfg, scene = scene1000
    a = build_release_table(cfg, scene)
    b = build_release_table(cfg, scene)
    assert np.array_equal(a.ranks, b.ranks)
    assert np.array_equal(a.planned, b.planned)
    assert np.array_equal(a.release_tick, b.release_tick)
    assert np.array_equal(a.actual, b.actual)


def test_release_table_differs_by_scene_seed(default_cfg):
    _, sc1 = default_cfg, build_scene(default_cfg, 1000)
    _, sc2 = default_cfg, build_scene(default_cfg, 1001)
    t1 = build_release_table(default_cfg, sc1)
    t2 = build_release_table(default_cfg, sc2)
    assert not np.array_equal(t1.ranks, t2.ranks)
