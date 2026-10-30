# OWNER: vo-rvo-dev
"""M1 / 3D-VO 规划器测试（handoff §5.2/§5.3、CONTRACTS §6/§7）。

覆盖：
* 已知命中 / 远离 / 擦边相切 / 预测窗外交付的判定与规划输出；
* **真三维安全但二维投影会误判为危险**的显式构造用例；
* 活动邻机用 ``w = c - v_j``、冻结邻机用 ``w = c``；
* 起始重叠 → 全候选危险 → ``sample_exhausted``/``used_fallback``/诊断；
* 只读快照、确定性、速度上限、诊断字段形状与取值；
* 与全量暴力选择逐位一致（规划器级集成）；性能基准（--bench）。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from uavbench.config import apply_overrides
from uavbench.planners.base import PlanState
from uavbench.planners.candidates import (
    build_candidate_set,
    danger_flags,
    fallback_select,
    min_closed_segment_distance,
    min_relative_distances,
    select_exhaustive,
)
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
    neighbor_mask = np.ones(neighbor_ids.shape, dtype=bool)
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
        neighbor_mask=neighbor_mask,
    )


def _two_agent_state(neighbor_position, neighbor_velocity, v_pref, *, frozen_neighbor=False):
    """ego 在原点、单个邻机位于 ``neighbor_position`` 的最小 PlanState。"""
    positions = np.zeros((2, 3), dtype=np.float64)
    positions[1] = np.asarray(neighbor_position, dtype=np.float64)
    velocities = np.zeros((2, 3), dtype=np.float64)
    velocities[1] = np.asarray(neighbor_velocity, dtype=np.float64)
    vp = np.asarray(v_pref, dtype=np.float64).reshape(1, 3)
    return _make_state(
        positions,
        velocities,
        vp,
        active_ids=np.array([0], dtype=np.int64),
        frozen=np.array([False, bool(frozen_neighbor)]),
    )


def _candidate_set(default_cfg):
    return build_candidate_set(default_cfg, 1000)


def _planner(default_cfg, **kwargs):
    return VO3DPlanner(default_cfg, _candidate_set(default_cfg))


def _reference_danger(
    velocity, p_ij, *, v_j, frozen_j, tau=TAU, d_plan=D_PLAN, v_i=None
) -> bool:
    flag = danger_flags(
        np.asarray(velocity, dtype=np.float64).reshape(1, 3),
        model="vo",
        v_i=np.zeros(3) if v_i is None else np.asarray(v_i, dtype=np.float64),
        v_j=np.asarray(v_j, dtype=np.float64).reshape(-1, 3),
        frozen_j=np.asarray(frozen_j, dtype=bool).reshape(-1),
        p_ij=np.asarray(p_ij, dtype=np.float64).reshape(-1, 3),
        tau=tau,
        d_plan=d_plan,
    )
    return bool(flag[0])


# ---------------------------------------------------------------------------
# 几何判定与规划输出
# ---------------------------------------------------------------------------
def test_vo_known_hit_is_avoided(default_cfg):
    # 邻机在 -10 m x，ego 期望速度 (-2,0,0) 迎面 → 4.5 s 后进入 1 m
    state = _two_agent_state((-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-2.0, 0.0, 0.0))
    planner = _planner(default_cfg)
    result = planner.plan(state)
    velocity = result.velocities[0]
    assert not np.array_equal(velocity, np.array([-2.0, 0.0, 0.0]))
    p_ij = np.array([[-10.0, 0.0, 0.0]])
    assert not _reference_danger(velocity, p_ij, v_j=np.zeros((1, 3)), frozen_j=np.zeros(1, bool))
    assert not bool(result.diagnostics.sample_exhausted[0])
    assert int(result.diagnostics.candidate_count_tested[0]) > 1


def test_vo_known_away_keeps_pref(default_cfg):
    state = _two_agent_state((-10.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    result = _planner(default_cfg).plan(state)
    assert np.array_equal(result.velocities[0], np.array([2.0, 0.0, 0.0]))
    assert int(result.diagnostics.candidate_count_tested[0]) == 1
    assert not bool(result.diagnostics.sample_exhausted[0])


def test_vo_outside_prediction_window_is_safe(default_cfg):
    # 20 m 外迎面：首次接触约 9.5 s > tau=5 → 安全
    state = _two_agent_state((-20.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-2.0, 0.0, 0.0))
    result = _planner(default_cfg).plan(state)
    assert np.array_equal(result.velocities[0], np.array([-2.0, 0.0, 0.0]))
    assert int(result.diagnostics.candidate_count_tested[0]) == 1
    # 同一几何把 tau 放大到 10 s → 命中 → 不再选择 v_pref
    long_tau_cfg = apply_overrides(default_cfg, {"planners.vo.tau": 10.0})
    result2 = VO3DPlanner(long_tau_cfg, _candidate_set(default_cfg)).plan(state)
    assert not np.array_equal(result2.velocities[0], np.array([-2.0, 0.0, 0.0]))
    assert not bool(result2.diagnostics.sample_exhausted[0])


def test_vo_grazing_tangent_is_safe(default_cfg):
    # 邻机 (1,1,0)，w=(1,0,0)：最近距离恰为 d_plan=1（相切）→ 严格 < 不成立 → 安全
    state = _two_agent_state((1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    p = np.array([[1.0, 1.0, 0.0]])
    w = np.array([[[1.0, 0.0, 0.0]]])
    md = float(min_closed_segment_distance(p, w, TAU)[0, 0])
    assert md == pytest.approx(1.0, rel=0.0, abs=1e-15)
    result = _planner(default_cfg).plan(state)
    assert np.array_equal(result.velocities[0], np.array([1.0, 0.0, 0.0]))
    assert int(result.diagnostics.candidate_count_tested[0]) == 1


def test_vo_exact_hit_is_dangerous(default_cfg):
    # 5 m 外迎面：t=2 s 精确穿过原点 → 危险
    state = _two_agent_state((-5.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-2.0, 0.0, 0.0))
    p = np.array([[-5.0, 0.0, 0.0]])
    assert _reference_danger(
        np.array([-2.0, 0.0, 0.0]), p, v_j=np.zeros((1, 3)), frozen_j=np.zeros(1, bool)
    )
    result = _planner(default_cfg).plan(state)
    assert not np.array_equal(result.velocities[0], np.array([-2.0, 0.0, 0.0]))
    assert not _reference_danger(
        result.velocities[0], p, v_j=np.zeros((1, 3)), frozen_j=np.zeros(1, bool)
    )


def test_vo_3d_safe_but_2d_projection_would_be_unsafe(default_cfg):
    """真三维安全、但忽略 z 的二维投影会误判为危险的显式构造。

    邻机相对位置 p_ij = (-5, 0, 1.5)，冻结邻机（w = c = (-2,0,0)）：
    三维轨迹 r(t)=(-5+2t, 0, 1.5)，t=2.5 s 时最近距离 = 1.5 m > d_plan=1 → 安全；
    二维投影 r_xy(t)=(-5+2t, 0) 在 t=2.5 s 穿过原点（距离 0 < 1）→ 会误判危险。
    （因为三维距离 >= 二维投影距离，反方向的不一致不可能出现。）
    """
    state = _two_agent_state(
        (-5.0, 0.0, 1.5), (0.0, 0.0, 0.0), (-2.0, 0.0, 0.0), frozen_neighbor=True
    )
    p_3d = np.array([[(-5.0, 0.0, 1.5)]])
    w = np.array([[[-2.0, 0.0, 0.0]]])
    md_3d = float(min_closed_segment_distance(p_3d, w, TAU)[0, 0])
    # 二维投影：去掉 z 后各自视为 z=0 的平面问题
    p_2d = np.array([[[-5.0, 0.0, 0.0]]])
    w_2d = np.array([[[-2.0, 0.0, 0.0]]])
    md_2d = float(min_closed_segment_distance(p_2d, w_2d, TAU)[0, 0])
    assert md_3d == pytest.approx(1.5, rel=0.0, abs=1e-15)
    assert md_2d < D_PLAN
    # 三维判定为安全 → 规划器直接采用 v_pref
    result = _planner(default_cfg).plan(state)
    assert np.array_equal(result.velocities[0], np.array([-2.0, 0.0, 0.0]))
    assert int(result.diagnostics.candidate_count_tested[0]) == 1


def test_vo_active_neighbor_uses_v_j(default_cfg):
    # 邻机以 (-2,0,0) 同向运动，eta 的 v_pref=(-2,0,0) → w=0，距离恒为 10 m → 安全
    state = _two_agent_state((-10.0, 0.0, 0.0), (-2.0, 0.0, 0.0), (-2.0, 0.0, 0.0))
    result = _planner(default_cfg).plan(state)
    assert np.array_equal(result.velocities[0], np.array([-2.0, 0.0, 0.0]))
    assert int(result.diagnostics.candidate_count_tested[0]) == 1
    # 若错误地把活动邻机当成冻结体（w = c），同一候选会被判危险
    p = np.array([[-10.0, 0.0, 0.0]])
    vj = np.array([[-2.0, 0.0, 0.0]])
    assert not _reference_danger(
        np.array([-2.0, 0.0, 0.0]), p, v_j=vj, frozen_j=np.zeros(1, bool)
    )
    assert _reference_danger(
        np.array([-2.0, 0.0, 0.0]), p, v_j=vj, frozen_j=np.ones(1, bool)
    )


# ---------------------------------------------------------------------------
# 起始重叠、回退与诊断
# ---------------------------------------------------------------------------
def test_vo_initial_overlap_forced_fallback_diagnostics(default_cfg):
    # 邻机距 ego 0.5 m（< d_plan）：任意候选在 t=0 即危险 → 必然回退
    state = _two_agent_state((0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    result = _planner(default_cfg).plan(state)
    diag = result.diagnostics
    assert bool(diag.sample_exhausted[0]) is True
    assert bool(diag.used_fallback[0]) is True
    assert int(diag.initial_overlap_pairs[0]) == 1
    # 回退按 TTC（此处全为 0）→ v_pref 偏差最小者 = v_pref=(2,0,0)，它朝重叠邻机方向，
    # 窗口内最近距离到 0 → 残差 = d_plan - 0 = 1.0（回退不保证安全，必须如实报告）。
    assert float(diag.constraint_violation_max[0]) == pytest.approx(
        1.0, rel=0.0, abs=1e-12
    )
    assert not bool(diag.solver_infeasible[0])
    assert not bool(diag.numerical_error[0])
    # 回退仍测试了全部候选
    n_cand = _candidate_set(default_cfg).build(V_MAX, np.zeros(3), np.array([2.0, 0.0, 0.0])).shape[0]
    assert int(diag.candidate_count_tested[0]) == n_cand
    # 回退速度有限且在速度上限内
    assert np.all(np.isfinite(result.velocities[0]))
    assert np.linalg.norm(result.velocities[0]) <= V_MAX + 1e-8


def test_vo_diagnostics_shapes_and_values_random(default_cfg):
    rng = np.random.default_rng(11)
    n = 20
    positions = rng.uniform(-10.0, 10.0, size=(n, 3))
    velocities = rng.uniform(-1.5, 1.5, size=(n, 3))
    v_pref = rng.uniform(-2.0, 2.0, size=(n, 3))
    norms = np.linalg.norm(v_pref, axis=1, keepdims=True)
    v_pref = v_pref / np.maximum(norms, 1e-12) * np.minimum(norms, 2.0)
    active = np.arange(n, dtype=np.int64)
    state = _make_state(positions, velocities, v_pref, active)
    result = _planner(default_cfg).plan(state)
    a = n
    assert result.velocities.shape == (a, 3)
    assert result.velocities.dtype == np.float64
    assert np.all(np.isfinite(result.velocities))
    assert np.all(np.linalg.norm(result.velocities, axis=1) <= V_MAX + 1e-8)

    diag = result.diagnostics
    for name in (
        "sample_exhausted",
        "used_fallback",
        "solver_infeasible",
        "constraint_violation_max",
        "candidate_count_tested",
        "initial_overlap_pairs",
        "numerical_error",
    ):
        assert np.asarray(getattr(diag, name)).shape == (a,), name
    assert diag.solver_infeasible.dtype == bool
    assert not np.any(diag.solver_infeasible)
    assert not np.any(diag.numerical_error)
    assert np.array_equal(diag.sample_exhausted, diag.used_fallback)
    assert np.all(diag.candidate_count_tested >= 1)
    assert np.all(diag.candidate_count_tested <= 387)
    assert np.all(diag.constraint_violation_max >= 0.0)
    # 初始重叠计数与直接的 p_ij 距离一致
    distances = np.linalg.norm(state.pair_offsets(), axis=2)
    expected_overlap = np.sum(distances < D_PLAN, axis=1)
    assert np.array_equal(diag.initial_overlap_pairs, expected_overlap.astype(np.int32))
    # extra 诊断
    assert diag.extra is not None
    assert diag.extra["preferred_speed"].shape == (a,)
    assert diag.extra["fallback_ttc"].shape == (a,)
    assert np.allclose(
        diag.extra["preferred_speed"], np.linalg.norm(v_pref, axis=1), atol=1e-15
    )


def test_vo_selection_matches_bruteforce(default_cfg):
    """规划器级集成：在随机小场景上与全量暴力选择逐位一致。"""
    rng = np.random.default_rng(2026)
    cset = _candidate_set(default_cfg)
    planner = VO3DPlanner(default_cfg, cset)
    n = 7
    for _ in range(4):
        positions = rng.uniform(-6.0, 6.0, size=(n, 3))
        velocities = rng.uniform(-1.2, 1.2, size=(n, 3))
        v_pref = rng.uniform(-2.0, 2.0, size=(n, 3))
        norms = np.linalg.norm(v_pref, axis=1, keepdims=True)
        v_pref = v_pref / np.maximum(norms, 1e-12) * np.minimum(norms, 2.0)
        active = np.arange(n, dtype=np.int64)
        state = _make_state(positions, velocities, v_pref, active)
        result = planner.plan(state)
        for a in range(n):
            solver_ids = state.neighbor_ids[a]
            p = state.positions[solver_ids] - state.positions[a]
            vj = state.velocities[solver_ids]
            cand = cset.build(V_MAX, state.velocities[a], state.v_pref[a])
            danger = danger_flags(
                cand,
                model="vo",
                v_i=state.velocities[a],
                v_j=vj,
                frozen_j=state.frozen[solver_ids],
                p_ij=p,
                tau=TAU,
                d_plan=D_PLAN,
            )
            idx = select_exhaustive(cand, danger, state.velocities[a], state.v_pref[a])
            if idx >= 0:
                expected = cand[idx]
                expected_exhausted = False
            else:
                idx, _ = fallback_select(cand, vj, p, D_PLAN, state.v_pref[a])
                expected = cand[idx]
                expected_exhausted = True
            assert np.array_equal(result.velocities[a], expected)
            assert bool(result.diagnostics.sample_exhausted[a]) == expected_exhausted


# ---------------------------------------------------------------------------
# 冻结体语义、只读快照、确定性、空活动集
# ---------------------------------------------------------------------------
def test_vo_frozen_neighbors_match_rvo_and_static_semantics(default_cfg):
    from uavbench.planners.rvo3d import RVO3DPlanner

    # 两个冻结邻机；v_pref 与左侧冻结体冲突，需要搜索
    positions = np.array(
        [[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64
    )
    velocities = np.zeros((3, 3), dtype=np.float64)
    v_pref = np.array([[-2.0, 0.0, 0.0]], dtype=np.float64)
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
        vo_result.diagnostics.sample_exhausted, rvo_result.diagnostics.sample_exhausted
    )
    # 冻结邻机的相对速度均等于 c（两模型一致）
    p = np.array([[-3.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    frozen = np.ones(2, dtype=bool)
    for model in ("vo", "rvo"):
        flags = danger_flags(
            np.array([[-2.0, 0.0, 0.0]]),
            model=model,
            v_i=np.zeros(3),
            v_j=np.zeros((2, 3)),
            frozen_j=frozen,
            p_ij=p,
            tau=TAU,
            d_plan=D_PLAN,
        )
        md = min_relative_distances(
            np.array([[-2.0, 0.0, 0.0]]),
            model=model,
            v_i=np.zeros(3),
            v_j=np.zeros((2, 3)),
            frozen_j=frozen,
            p_ij=p,
            tau=TAU,
        )
        assert bool(flags[0]) == bool(np.any(md[0] < D_PLAN))


def test_vo_plan_accepts_readonly_snapshot_and_is_deterministic(default_cfg):
    rng = np.random.default_rng(3)
    n = 10
    positions = rng.uniform(-5.0, 5.0, size=(n, 3))
    velocities = rng.uniform(-1.0, 1.0, size=(n, 3))
    v_pref = rng.uniform(-2.0, 2.0, size=(n, 3))
    state = _make_state(positions, velocities, v_pref, np.arange(n, dtype=np.int64))
    # 与仿真器一致：快照数组全部只读
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
    planner = _planner(default_cfg)
    first = planner.plan(state)
    second = planner.plan(state)
    assert np.array_equal(first.velocities, second.velocities)
    assert np.array_equal(
        first.diagnostics.candidate_count_tested,
        second.diagnostics.candidate_count_tested,
    )
    assert np.array_equal(state.positions, positions_before)
    assert np.array_equal(state.velocities, velocities_before)


def test_vo_empty_active_state(default_cfg):
    state = _make_state(
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        np.zeros((0, 3)),
        active_ids=np.zeros(0, dtype=np.int64),
        frozen=np.ones(3, dtype=bool),
        stages=np.zeros(3, dtype=np.int8),
    )
    result = _planner(default_cfg).plan(state)
    assert result.velocities.shape == (0, 3)
    assert result.velocities.dtype == np.float64
    diag = result.diagnostics
    assert diag.sample_exhausted.shape == (0,)
    assert diag.constraint_violation_max.shape == (0,)
    assert diag.extra is not None
    assert diag.extra["preferred_speed"].shape == (0,)


def test_vo_velocity_never_exceeds_v_max(default_cfg):
    rng = np.random.default_rng(5)
    n = 30
    positions = rng.uniform(-8.0, 8.0, size=(n, 3))
    velocities = rng.uniform(-2.0, 2.0, size=(n, 3))
    v_pref = rng.uniform(-2.0, 2.0, size=(n, 3))
    result = _planner(default_cfg).plan(
        _make_state(positions, velocities, v_pref, np.arange(n, dtype=np.int64))
    )
    assert np.all(np.linalg.norm(result.velocities, axis=1) <= V_MAX + 1e-8)


# ---------------------------------------------------------------------------
# 性能基准（默认跳过，--bench 启用）
# ---------------------------------------------------------------------------
def _bench_states(default_cfg: "object") -> dict[str, PlanState]:
    n = N_AGENTS
    rng = np.random.default_rng(0)

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    states: dict[str, PlanState] = {}
    # (a) 对跖点起点：20 m 球面，v_pref 指向中心
    d = np.random.default_rng(1).normal(size=(n, 3))
    pos = unit(d) * 20.0
    states["antipodal_start"] = _make_state(
        pos, np.zeros((n, 3)), -unit(d) * V_MAX, np.arange(n, dtype=np.int64)
    )
    # (b) 拥堵中心：半径 0.1–1.2 m 内 100 机、随机速度（最坏情形，几乎全部回退）
    d2 = rng.normal(size=(n, 3))
    pos2 = unit(d2) * rng.uniform(0.1, 1.2, size=(n, 1))
    states["dense_congested"] = _make_state(
        pos2,
        unit(rng.normal(size=(n, 3))) * V_MAX,
        unit(rng.normal(size=(n, 3))) * V_MAX,
        np.arange(n, dtype=np.int64),
    )
    return states


@pytest.mark.bench
def test_vo_planning_bench_1200_steps(default_cfg):
    """N=100、M=99、单线程：外推 1200 步规划时间必须 < 180 s（CONTRACTS §7）。"""
    planner = _planner(default_cfg)
    worst = 0.0
    lines = []
    for name, state in _bench_states(default_cfg).items():
        planner.plan(state)  # warmup
        n_iter = 20
        t0 = time.perf_counter()
        for _ in range(n_iter):
            result = planner.plan(state)
        per_step = (time.perf_counter() - t0) / n_iter
        estimated = per_step * 1200.0
        worst = max(worst, estimated)
        lines.append(
            f"VO {name}: {per_step * 1000.0:.1f} ms/step, "
            f"est 1200 steps = {estimated:.1f} s, "
            f"exhausted={int(result.diagnostics.sample_exhausted.sum())}/{N_AGENTS}"
        )
    print("\n" + "\n".join(lines))
    assert worst < 180.0, f"estimated 1200-step VO planning time {worst:.1f}s >= 180 s"
