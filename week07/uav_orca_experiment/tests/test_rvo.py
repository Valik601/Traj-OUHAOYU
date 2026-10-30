# OWNER: vo-rvo-dev
"""M2 / 3D-RVO 规划器测试（handoff §5.2、CONTRACTS §7）。

覆盖：
* 逐候选核对 ``w = 2c - v_i - v_j`` 的有限时间定义（随机对拍 + 解析用例）；
* 冻结邻机退化为 ``w = c``（与 VO 判定/规划逐位一致）；
* **“只平移 VO 顶点”的错误 RVO 实现与正确定义不等价**（构造出两判定相反、且
  规划器输出不同的用例）；
* ``v_i`` 进入互惠定义（忽略当前速度的近似会误判）；
* 起始重叠回退诊断、只读快照、确定性、速度上限、空活动集；
* 与全量暴力选择逐位一致；性能基准（--bench）。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from uavbench.planners.base import PlanState
from uavbench.planners.candidates import (
    CandidateSet,
    build_candidate_set,
    danger_flags,
    fallback_select,
    min_closed_segment_distance,
    min_relative_distances,
    relative_velocities,
    select_exhaustive,
)
from uavbench.planners.rvo3d import RVO3DPlanner
from uavbench.planners.vo3d import VO3DPlanner

V_MAX = 2.0
D_PLAN = 1.0
TAU = 5.0
N_AGENTS = 100


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _make_state(
    positions,
    velocities,
    v_pref,
    active_ids,
    *,
    frozen=None,
    stages=None,
    t=0.0,
    dt=0.1,
) -> PlanState:
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    vp = np.asarray(v_pref, dtype=np.float64)
    n = int(pos.shape[0])
    active = np.asarray(active_ids, dtype=np.int64)
    a = int(active.size)
    if frozen is None:
        frozen = np.zeros(n, dtype=bool)
    if stages is None:
        stages = np.full(n, 2, dtype=np.int8)
    if a:
        ids = np.broadcast_to(np.arange(n, dtype=np.int64), (a, n)).copy()
        keep = ids != active[:, None]
        neighbor_ids = ids[keep].reshape(a, n - 1).astype(np.int64)
    else:
        neighbor_ids = np.zeros((0, 0), dtype=np.int64)
    return PlanState(
        t=float(t),
        dt=float(dt),
        positions=pos,
        velocities=vel,
        stages=np.asarray(stages, dtype=np.int8),
        frozen=np.asarray(frozen, dtype=bool),
        active_ids=active,
        waypoints=np.zeros((a, 3), dtype=np.float64),
        v_pref=vp,
        neighbor_ids=neighbor_ids,
        neighbor_mask=np.ones(neighbor_ids.shape, dtype=bool),
    )


def _two_agent_state(neighbor_position, neighbor_velocity, v_pref, *, ego_velocity=None, frozen_neighbor=False):
    positions = np.zeros((2, 3), dtype=np.float64)
    positions[1] = np.asarray(neighbor_position, dtype=np.float64)
    velocities = np.zeros((2, 3), dtype=np.float64)
    velocities[1] = np.asarray(neighbor_velocity, dtype=np.float64)
    if ego_velocity is not None:
        velocities[0] = np.asarray(ego_velocity, dtype=np.float64)
    vp = np.asarray(v_pref, dtype=np.float64).reshape(1, 3)
    return _make_state(
        positions,
        velocities,
        vp,
        active_ids=np.array([0], dtype=np.int64),
        frozen=np.array([False, bool(frozen_neighbor)]),
    )


def _candidate_set(default_cfg, seed: int = 1000):
    return build_candidate_set(default_cfg, seed)


def _rvo_danger_reference(velocity, p_ij, *, v_i, v_j, frozen_j, tau=TAU, d_plan=D_PLAN) -> bool:
    """按定义直接计算：``w = 2c-v_i-v_j``（冻结邻机 ``w=c``）。"""
    flag = danger_flags(
        np.asarray(velocity, dtype=np.float64).reshape(1, 3),
        model="rvo",
        v_i=np.asarray(v_i, dtype=np.float64),
        v_j=np.asarray(v_j, dtype=np.float64).reshape(-1, 3),
        frozen_j=np.asarray(frozen_j, dtype=bool).reshape(-1),
        p_ij=np.asarray(p_ij, dtype=np.float64).reshape(-1, 3),
        tau=tau,
        d_plan=d_plan,
    )
    return bool(flag[0])


def _wrong_apex_shift_danger(candidates, p_ij, *, v_i, v_j, tau=TAU, d_plan=D_PLAN) -> np.ndarray:
    """错误实现：只把 VO 顶点平移到 ``c - v_i - v_j``，不做 ``2c`` 变换。

    ``candidates`` ``(C,3)``，``p_ij`` ``(M,3)``；返回 ``(C,)`` 危险标志。
    """
    cand = np.asarray(candidates, dtype=np.float64)
    p = np.asarray(p_ij, dtype=np.float64)
    vj = np.asarray(v_j, dtype=np.float64)
    if vj.ndim == 1:
        vj = vj[None, :]
    w = (
        cand[:, None, :]
        - np.asarray(v_i, dtype=np.float64)[None, None, :]
        - vj[None, :, :]
    )  # (C,M,3)
    md = min_closed_segment_distance(p[None, :, :], w, tau)  # (C,M)
    return np.any(md < float(d_plan), axis=1)


# ---------------------------------------------------------------------------
# 定义层：逐候选核对 2c - vi - vj
# ---------------------------------------------------------------------------
def test_rvo_relative_velocity_definition():
    cand = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vi = np.array([0.5, -0.5, 0.0])
    vj = np.array([[0.25, 0.0, 0.0], [-0.25, 0.5, 0.0]])
    frozen = np.array([False, True])
    w = relative_velocities(cand, model="rvo", v_i=vi, v_j=vj, frozen_j=frozen)
    assert np.array_equal(w[:, 0, :], 2.0 * cand - (vi + vj[0])[None, :])
    # 冻结邻机：不参与互惠推理，w = c（与 VO 相同）
    assert np.array_equal(w[:, 1, :], cand)


def test_rvo_danger_matches_definition_per_candidate_random():
    rng = np.random.default_rng(123)
    n_cand, m = 17, 9
    cand = rng.uniform(-2.0, 2.0, size=(n_cand, 3))
    vi = rng.uniform(-1.5, 1.5, size=3)
    vj = rng.uniform(-1.5, 1.5, size=(m, 3))
    frozen = rng.random(m) < 0.3
    p = rng.uniform(-15.0, 15.0, size=(m, 3))
    flags = danger_flags(
        cand, model="rvo", v_i=vi, v_j=vj, frozen_j=frozen, p_ij=p, tau=TAU, d_plan=D_PLAN
    )
    # 逐候选逐邻机核对原始定义：活动 w=2c-vi-vj；冻结 w=c
    w_all = relative_velocities(cand, model="rvo", v_i=vi, v_j=vj, frozen_j=frozen)
    for i in range(n_cand):
        expected = False
        for j in range(m):
            w = cand[i] if frozen[j] else 2.0 * cand[i] - (vi + vj[j])
            assert np.array_equal(w, w_all[i, j])
            if float(min_closed_segment_distance(p[j], w, TAU)) < D_PLAN:
                expected = True
                break
        assert bool(flags[i]) == expected
    # 与 min_relative_distances 的逐对最近距离一致
    mrd = min_relative_distances(
        cand, model="rvo", v_i=vi, v_j=vj, frozen_j=frozen, p_ij=p, tau=TAU
    )
    assert np.array_equal(flags, np.any(mrd < D_PLAN, axis=1))


def test_rvo_transformed_horizon_identity():
    """实现等价性：min_{t in [0,tau]}||p-t(2c-vi-vj)|| == min_{u in [0,2tau]}||p-u(c-(vi+vj)/2)||。"""
    rng = np.random.default_rng(77)
    for _ in range(20):
        c = rng.uniform(-2.0, 2.0, size=3)
        vi = rng.uniform(-1.5, 1.5, size=3)
        vj = rng.uniform(-1.5, 1.5, size=3)
        p = rng.uniform(-12.0, 12.0, size=3)
        w_rvo = 2.0 * c - vi - vj
        w_half = c - 0.5 * (vi + vj)
        md_direct = float(min_closed_segment_distance(p, w_rvo, TAU))
        md_transformed = float(min_closed_segment_distance(p, w_half, 2.0 * TAU))
        assert md_direct == pytest.approx(md_transformed, rel=0.0, abs=1e-12)


def test_rvo_known_hit_and_outside_window():
    # 精确命中：p=(-10,0,0), vi=vj=0, c=(-2,0,0) → w=2c=(-4,0,0)，t=2.25 穿过
    assert _rvo_danger_reference(
        np.array([-2.0, 0.0, 0.0]),
        np.array([-10.0, 0.0, 0.0]),
        v_i=np.zeros(3),
        v_j=np.zeros((1, 3)),
        frozen_j=np.zeros(1, bool),
    )
    # 预测窗外：p=(-30,0,0) → 2c=(-4,0,0)，接触 t=(30-1)/4=7.25 > 5
    assert not _rvo_danger_reference(
        np.array([-2.0, 0.0, 0.0]),
        np.array([-30.0, 0.0, 0.0]),
        v_i=np.zeros(3),
        v_j=np.zeros((1, 3)),
        frozen_j=np.zeros(1, bool),
    )
    # 冻结邻机：w=c，接触 t=(30-1)/2=14.5 → 仍安全
    assert not _rvo_danger_reference(
        np.array([-2.0, 0.0, 0.0]),
        np.array([-30.0, 0.0, 0.0]),
        v_i=np.zeros(3),
        v_j=np.zeros((1, 3)),
        frozen_j=np.ones(1, bool),
    )
    # 冻结邻机 p=(-10,0,0)：w=c=(-2,0,0) → 接触 t=4.5 → 危险
    assert _rvo_danger_reference(
        np.array([-2.0, 0.0, 0.0]),
        np.array([-10.0, 0.0, 0.0]),
        v_i=np.zeros(3),
        v_j=np.zeros((1, 3)),
        frozen_j=np.ones(1, bool),
    )


def test_rvo_uses_current_velocity_vi():
    # p=(-6,0,0), vi=(2,0,0), vj=0, c=(0.3,0,0)：正确 w=2c-vi=(-1.4,0,0) → 3.57 s 命中；
    # 忽略 vi 的错误近似 w=2c=(0.6,0,0) → 8.33 s，窗外 → 会漏判。
    p = np.array([[-6.0, 0.0, 0.0]])
    assert _rvo_danger_reference(
        np.array([0.3, 0.0, 0.0]),
        p[0],
        v_i=np.array([2.0, 0.0, 0.0]),
        v_j=np.zeros((1, 3)),
        frozen_j=np.zeros(1, bool),
    )
    md_naive = float(min_closed_segment_distance(p, np.array([[0.6, 0.0, 0.0]]), TAU)[0])
    assert md_naive >= D_PLAN


def test_rvo_3d_safe_but_2d_projection_would_be_unsafe(default_cfg):
    # 冻结邻机 p=(-5,0,1.5)，w=c=(-2,0,0)：三维最近距离 1.5 > 1，二维投影 0 < 1
    state = _two_agent_state((-5.0, 0.0, 1.5), (0.0, 0.0, 0.0), (-2.0, 0.0, 0.0), frozen_neighbor=True)
    p_3d = np.array([[-5.0, 0.0, 1.5]])
    w = np.array([[-2.0, 0.0, 0.0]])
    md_3d = float(min_closed_segment_distance(p_3d, w, TAU)[0])
    p_2d = np.array([[-5.0, 0.0, 0.0]])
    md_2d = float(min_closed_segment_distance(p_2d, w, TAU)[0])
    assert md_3d == pytest.approx(1.5, rel=0.0, abs=1e-15)
    assert md_2d < D_PLAN
    result = RVO3DPlanner(default_cfg, _candidate_set(default_cfg)).plan(state)
    assert np.array_equal(result.velocities[0], np.array([-2.0, 0.0, 0.0]))
    assert int(result.diagnostics.candidate_count_tested[0]) == 1


# ---------------------------------------------------------------------------
# 错误“只平移 VO 顶点”实现必须与正确 RVO 定义不同
# ---------------------------------------------------------------------------
def test_wrong_apex_shift_danger_differs_from_correct_rvo():
    # p=(-10,0,0)，vi=vj=0，t=2.5 s 的几何：正确 RVO（w=2c）在 c=(-1.8,0,0) 时
    # 3.6 m/s 迎面 → 2.5 s 穿过；错误“顶点平移”（w=c）在 5 s 时恰好距 1 m（相切）→ 安全。
    p = np.array([[-10.0, 0.0, 0.0]])
    c = np.array([[-1.8, 0.0, 0.0]])
    vi = np.zeros(3)
    vj = np.zeros((1, 3))
    frozen = np.zeros(1, bool)
    assert _rvo_danger_reference(c[0], p[0], v_i=vi, v_j=vj, frozen_j=frozen) is True
    wrong = _wrong_apex_shift_danger(c, p, v_i=vi, v_j=vj)
    assert bool(wrong[0]) is False
    # 正确定义下窗口内最近距离 < 1；错误实现下最近距离恰为 1（不在严格 < 内）
    correct_md = float(min_closed_segment_distance(p, 2.0 * c - vi - vj, TAU)[0])
    wrong_md = float(min_closed_segment_distance(p, c - vi - vj, TAU)[0])
    assert correct_md < D_PLAN <= wrong_md + 1e-15


def test_wrong_apex_shift_planner_differs_from_correct_rvo(default_cfg):
    """规划器级：同一输入下正确 RVO 与“仅移顶点”实现选择不同速度。"""
    cset = CandidateSet(
        dirs=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        speeds=(1.8,),
    )
    # 单活动 ego + 1 个静止活动邻机；v_pref=(-1.8,0,0)
    positions = np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    v_pref = np.array([[-1.8, 0.0, 0.0]])
    state = _make_state(positions, velocities, v_pref, np.array([0], dtype=np.int64))
    result = RVO3DPlanner(default_cfg, cset).plan(state)
    # 正确 RVO：v_pref 危险（2c-vi-vj=-3.6 m/s 迎面），零速度安全 → 返回 0
    p = np.array([[-10.0, 0.0, 0.0]])
    assert _rvo_danger_reference(
        np.array([-1.8, 0.0, 0.0]), p[0], v_i=np.zeros(3), v_j=np.zeros((1, 3)),
        frozen_j=np.zeros(1, bool),
    )
    assert np.array_equal(result.velocities[0], np.zeros(3))
    assert not bool(result.diagnostics.sample_exhausted[0])
    # 错误“仅移顶点”实现：v_pref 被判安全 → 选择 v_pref
    cand = cset.build(V_MAX, np.zeros(3), np.array([-1.8, 0.0, 0.0]))
    wrong_danger = _wrong_apex_shift_danger(
        cand, p, v_i=np.zeros(3), v_j=np.zeros((1, 3))
    )
    wrong_index = select_exhaustive(cand, wrong_danger, np.zeros(3), np.array([-1.8, 0.0, 0.0]))
    assert wrong_index >= 0
    assert np.array_equal(cand[wrong_index], np.array([-1.8, 0.0, 0.0]))
    assert not np.array_equal(result.velocities[0], cand[wrong_index])


# ---------------------------------------------------------------------------
# 冻结体语义：RVO == VO
# ---------------------------------------------------------------------------
def test_rvo_frozen_neighbor_uses_vo_test(default_cfg):
    p = np.array([[-3.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    vj = np.zeros((2, 3))
    frozen = np.ones(2, dtype=bool)
    cand = np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    vi = np.array([0.5, -0.25, 0.0])
    flags_vo = danger_flags(cand, model="vo", v_i=vi, v_j=vj, frozen_j=frozen, p_ij=p, tau=TAU, d_plan=D_PLAN)
    flags_rvo = danger_flags(cand, model="rvo", v_i=vi, v_j=vj, frozen_j=frozen, p_ij=p, tau=TAU, d_plan=D_PLAN)
    assert np.array_equal(flags_vo, flags_rvo)
    w_vo = relative_velocities(cand, model="vo", v_i=vi, v_j=vj, frozen_j=frozen)
    w_rvo = relative_velocities(cand, model="rvo", v_i=vi, v_j=vj, frozen_j=frozen)
    assert np.array_equal(w_vo, w_rvo)
    assert np.array_equal(w_vo, np.broadcast_to(cand[:, None, :], w_vo.shape))


def test_rvo_frozen_planner_equals_vo_planner(default_cfg):
    positions = np.array(
        [[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64
    )
    velocities = np.zeros((3, 3))
    v_pref = np.array([[-2.0, 0.0, 0.0]])
    state = _make_state(
        positions,
        velocities,
        v_pref,
        active_ids=np.array([0], dtype=np.int64),
        frozen=np.array([False, True, True]),
        stages=np.array([2, 0, 3], dtype=np.int8),
    )
    cset = _candidate_set(default_cfg)
    vo_result = VO3DPlanner(default_cfg, cset).plan(state)
    rvo_result = RVO3DPlanner(default_cfg, cset).plan(state)
    assert np.array_equal(vo_result.velocities, rvo_result.velocities)
    assert np.array_equal(
        vo_result.diagnostics.candidate_count_tested,
        rvo_result.diagnostics.candidate_count_tested,
    )
    assert np.array_equal(
        vo_result.diagnostics.constraint_violation_max,
        rvo_result.diagnostics.constraint_violation_max,
    )


# ---------------------------------------------------------------------------
# 起始重叠 / 回退 / 诊断 / 确定性
# ---------------------------------------------------------------------------
def test_rvo_initial_overlap_fallback_diagnostics(default_cfg):
    state = _two_agent_state((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    result = RVO3DPlanner(default_cfg, _candidate_set(default_cfg)).plan(state)
    diag = result.diagnostics
    assert bool(diag.sample_exhausted[0]) is True
    assert bool(diag.used_fallback[0]) is True
    assert int(diag.initial_overlap_pairs[0]) == 1
    assert not bool(diag.solver_infeasible[0])
    assert not bool(diag.numerical_error[0])
    # 回退速度朝重叠邻机运动 → 窗口内最近距离 0 → 残差 = d_plan
    assert float(diag.constraint_violation_max[0]) == pytest.approx(
        1.0, rel=0.0, abs=1e-12
    )
    n_cand = _candidate_set(default_cfg).build(V_MAX, np.zeros(3), np.array([2.0, 0.0, 0.0])).shape[0]
    assert int(diag.candidate_count_tested[0]) == n_cand
    assert np.all(np.isfinite(result.velocities[0]))


def test_rvo_selection_matches_bruteforce(default_cfg):
    rng = np.random.default_rng(4242)
    cset = _candidate_set(default_cfg)
    planner = RVO3DPlanner(default_cfg, cset)
    n = 7
    for _ in range(4):
        positions = rng.uniform(-6.0, 6.0, size=(n, 3))
        velocities = rng.uniform(-1.2, 1.2, size=(n, 3))
        v_pref = rng.uniform(-2.0, 2.0, size=(n, 3))
        norms = np.linalg.norm(v_pref, axis=1, keepdims=True)
        v_pref = v_pref / np.maximum(norms, 1e-12) * np.minimum(norms, 2.0)
        frozen = rng.random(n) < 0.2
        stages = np.where(frozen, 0, 2).astype(np.int8)
        active = np.flatnonzero(~frozen).astype(np.int64)
        state = _make_state(
            positions, velocities, v_pref[active], active, frozen=frozen, stages=stages
        )
        result = planner.plan(state)
        for row, agent in enumerate(active):
            solver_ids = state.neighbor_ids[row]
            p = state.positions[solver_ids] - state.positions[agent]
            vj = state.velocities[solver_ids]
            fz = state.frozen[solver_ids]
            cand = cset.build(V_MAX, state.velocities[agent], state.v_pref[row])
            danger = danger_flags(
                cand,
                model="rvo",
                v_i=state.velocities[agent],
                v_j=vj,
                frozen_j=fz,
                p_ij=p,
                tau=TAU,
                d_plan=D_PLAN,
            )
            index = select_exhaustive(cand, danger, state.velocities[agent], state.v_pref[row])
            if index >= 0:
                expected = cand[index]
                expected_exhausted = False
            else:
                index, _ = fallback_select(cand, vj, p, D_PLAN, state.v_pref[row])
                expected = cand[index]
                expected_exhausted = True
            assert np.array_equal(result.velocities[row], expected)
            assert bool(result.diagnostics.sample_exhausted[row]) == expected_exhausted


def test_rvo_plan_accepts_readonly_snapshot_and_is_deterministic(default_cfg):
    rng = np.random.default_rng(9)
    n = 10
    positions = rng.uniform(-5.0, 5.0, size=(n, 3))
    velocities = rng.uniform(-1.0, 1.0, size=(n, 3))
    v_pref = rng.uniform(-2.0, 2.0, size=(n, 3))
    state = _make_state(positions, velocities, v_pref, np.arange(n, dtype=np.int64))
    for arr in (
        state.positions,
        state.velocities,
        state.stages,
        state.frozen,
        state.active_ids,
        state.waypoints,
        state.v_pref,
        state.neighbor_ids,
        state.neighbor_mask,
    ):
        arr.setflags(write=False)
    positions_before = np.array(state.positions, copy=True)
    velocities_before = np.array(state.velocities, copy=True)
    planner = RVO3DPlanner(default_cfg, _candidate_set(default_cfg))
    first = planner.plan(state)
    second = planner.plan(state)
    assert np.array_equal(first.velocities, second.velocities)
    assert np.array_equal(state.positions, positions_before)
    assert np.array_equal(state.velocities, velocities_before)
    assert np.all(np.linalg.norm(first.velocities, axis=1) <= V_MAX + 1e-8)


def test_rvo_diagnostics_shapes_and_extra(default_cfg):
    rng = np.random.default_rng(31)
    n = 12
    positions = rng.uniform(-8.0, 8.0, size=(n, 3))
    velocities = rng.uniform(-1.5, 1.5, size=(n, 3))
    v_pref = rng.uniform(-2.0, 2.0, size=(n, 3))
    state = _make_state(positions, velocities, v_pref, np.arange(n, dtype=np.int64))
    result = RVO3DPlanner(default_cfg, _candidate_set(default_cfg)).plan(state)
    diag = result.diagnostics
    assert result.velocities.shape == (n, 3)
    assert result.velocities.dtype == np.float64
    assert np.all(np.isfinite(result.velocities))
    for name in (
        "sample_exhausted",
        "used_fallback",
        "solver_infeasible",
        "constraint_violation_max",
        "candidate_count_tested",
        "initial_overlap_pairs",
        "numerical_error",
    ):
        assert np.asarray(getattr(diag, name)).shape == (n,), name
    assert not np.any(diag.solver_infeasible)
    assert not np.any(diag.numerical_error)
    assert np.array_equal(diag.sample_exhausted, diag.used_fallback)
    assert diag.extra is not None
    for key in ("preferred_speed", "fallback_ttc"):
        assert diag.extra[key].shape == (n,)
    # constraint_violation_max 与显式定义一致（允许 1e-9 数值差）
    for a in range(n):
        ids = state.neighbor_ids[a]
        p = state.positions[ids] - state.positions[a]
        md = min_relative_distances(
            result.velocities[a][None, :],
            model="rvo",
            v_i=state.velocities[a],
            v_j=state.velocities[ids],
            frozen_j=state.frozen[ids],
            p_ij=p,
            tau=TAU,
        )[0]
        expected = float(np.max(np.maximum(D_PLAN - md, 0.0)))
        assert float(diag.constraint_violation_max[a]) == pytest.approx(
            expected, rel=0.0, abs=1e-9
        )


def test_rvo_empty_active_state(default_cfg):
    state = _make_state(
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros((0, 3)),
        active_ids=np.zeros(0, dtype=np.int64),
        frozen=np.ones(3, dtype=bool),
        stages=np.zeros(3, dtype=np.int8),
    )
    result = RVO3DPlanner(default_cfg, _candidate_set(default_cfg)).plan(state)
    assert result.velocities.shape == (0, 3)
    assert result.diagnostics.sample_exhausted.shape == (0,)
    assert result.diagnostics.extra is not None
    assert result.diagnostics.extra["preferred_speed"].shape == (0,)


# ---------------------------------------------------------------------------
# 性能基准（默认跳过，--bench 启用）
# ---------------------------------------------------------------------------
def _bench_states() -> dict[str, PlanState]:
    n = N_AGENTS
    rng = np.random.default_rng(0)

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    states: dict[str, PlanState] = {}
    d = np.random.default_rng(1).normal(size=(n, 3))
    states["antipodal_start"] = _make_state(
        unit(d) * 20.0, np.zeros((n, 3)), -unit(d) * V_MAX, np.arange(n, dtype=np.int64)
    )
    d2 = rng.normal(size=(n, 3))
    states["dense_congested"] = _make_state(
        unit(d2) * rng.uniform(0.1, 1.2, size=(n, 1)),
        unit(rng.normal(size=(n, 3))) * V_MAX,
        unit(rng.normal(size=(n, 3))) * V_MAX,
        np.arange(n, dtype=np.int64),
    )
    return states


@pytest.mark.bench
def test_rvo_planning_bench_1200_steps(default_cfg):
    """N=100、M=99、单线程：外推 1200 步规划时间必须 < 180 s（CONTRACTS §7）。"""
    planner = RVO3DPlanner(default_cfg, _candidate_set(default_cfg))
    worst = 0.0
    lines = []
    for name, state in _bench_states().items():
        planner.plan(state)  # warmup
        n_iter = 20
        t0 = time.perf_counter()
        for _ in range(n_iter):
            result = planner.plan(state)
        per_step = (time.perf_counter() - t0) / n_iter
        estimated = per_step * 1200.0
        worst = max(worst, estimated)
        lines.append(
            f"RVO {name}: {per_step * 1000.0:.1f} ms/step, "
            f"est 1200 steps = {estimated:.1f} s, "
            f"exhausted={int(result.diagnostics.sample_exhausted.sum())}/{N_AGENTS}"
        )
    print("\n" + "\n".join(lines))
    assert worst < 180.0, f"estimated 1200-step RVO planning time {worst:.1f}s >= 180 s"
