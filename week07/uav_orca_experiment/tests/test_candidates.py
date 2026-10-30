# OWNER: vo-rvo-dev
"""共享候选集合与候选安全搜索测试（CONTRACTS §7、handoff §5.2/§5.3）。

覆盖：
* CandidateSet 的固定索引顺序、去重容差、速度层截断、场景级旋转确定性/不可变性；
* 3D 有限预测窗危险判定核 vs 直接线段几何定义的随机对拍；
* 命中/远离/擦边/预测窗外/起始重叠等解析用例；
* TTC 回退排序（最大化最小 TTC → v_pref 偏差 → 固定索引）；
* “先测偏好最优候选 + 分块枚举”的加速搜索与全量暴力搜索逐位一致（随机对拍）；
* 邻机列表顺序无关性、无邻机、初始重叠诊断。
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench import geometry
from uavbench.planners.candidates import (
    CANDIDATE_DEDUPE_TOL,
    CandidateSet,
    build_candidate_set,
    candidate_rotation,
    danger_flags,
    fallback_select,
    fibonacci_directions,
    first_contact_times,
    min_closed_segment_distance,
    min_relative_distances,
    preference_keys,
    preference_order,
    relative_velocities,
    rotate_directions,
    search_velocities,
    select_exhaustive,
    squared_distances,
)

V_MAX = 2.0
TAU = 5.0
D_PLAN = 1.0


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
@pytest.fixture()
def cset(default_cfg):
    """默认配置 + 种子 1000 的场景级候选集合。"""
    return build_candidate_set(default_cfg, 1000)


def _rand_ball(rng: np.random.Generator, n: int, v_max: float = V_MAX) -> np.ndarray:
    """n 个均匀方向 × 均匀速度（<= v_max）的 (n,3)。"""
    d = rng.normal(size=(n, 3))
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return d * rng.uniform(0.0, v_max, size=(n, 1))


def _exhaustive_search(
    cset: CandidateSet,
    *,
    model: str,
    v_current: np.ndarray,
    v_pref: np.ndarray,
    p_ij: np.ndarray,
    v_j: np.ndarray,
    frozen_j: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """全量暴力搜索（测试参考）：逐候选判定 + 偏好字典序选择 / TTC 回退。"""
    cand = cset.build(V_MAX, v_current, v_pref)
    danger = danger_flags(
        cand,
        model=model,
        v_i=v_current,
        v_j=v_j,
        frozen_j=frozen_j,
        p_ij=p_ij,
        tau=TAU,
        d_plan=D_PLAN,
    )
    index = select_exhaustive(cand, danger, v_current, v_pref)
    if index >= 0:
        return np.array(cand[index], copy=True), False
    index, _ = fallback_select(cand, v_j, p_ij, D_PLAN, v_pref)
    return np.array(cand[index], copy=True), True


# ---------------------------------------------------------------------------
# CandidateSet：构造、顺序、去重、截断、旋转、不可变
# ---------------------------------------------------------------------------
def test_default_candidate_set_structure(default_cfg, cset):
    assert cset.dirs.shape == (128, 3)
    assert cset.dirs.dtype == np.float64
    assert cset.speeds == (0.5, 1.0, 2.0)
    assert cset.include_zero and cset.include_current and cset.include_pref
    assert cset.n_template_candidates() == 3 + 128 * 3 == 387
    assert default_cfg.planners.candidate.directions == 128
    assert np.allclose(np.linalg.norm(cset.dirs, axis=1), 1.0, atol=1e-15)
    # 基础方向与 geometry 的 Fibonacci 公式同构（旋转前）
    assert np.array_equal(fibonacci_directions(128), geometry.fibonacci_sphere_directions(128))


def test_build_fixed_index_order():
    cs = CandidateSet(dirs=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), speeds=(0.5, 1.0))
    vc = np.array([0.25, 0.0, 0.0])
    vp = np.array([0.0, 0.25, 0.0])
    cand = cs.build(V_MAX, vc, vp)
    # [零, 当前, v_pref, (dir0,0.5),(dir0,1.0),(dir1,0.5),(dir1,1.0)]
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert np.array_equal(cand, expected)
    # 方向索引升序、层升序：先固定层再换层是错误的
    assert np.array_equal(cand[6], np.array([0.0, 1.0, 0.0]))


def test_build_dedupes_with_first_occurrence_kept():
    cs = CandidateSet(dirs=np.array([[1.0, 0.0, 0.0]]), speeds=(1.0,))
    # 零速度 == 当前速度：当前速度被去重
    cand = cs.build(V_MAX, np.zeros(3), np.zeros(3))
    assert np.array_equal(cand, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    # 当前速度 == 网格候选：网格项被去重，前缀保留
    cand2 = cs.build(V_MAX, np.array([1.0, 0.0, 0.0]), np.zeros(3))
    assert cand2.shape == (2, 3)
    assert np.array_equal(cand2[1], np.array([1.0, 0.0, 0.0]))
    # v_pref == 当前速度：v_pref 被去重
    cand3 = cs.build(V_MAX, np.array([0.5, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]))
    assert cand3.shape == (3, 3)
    assert np.array_equal(cand3[1], np.array([0.5, 0.0, 0.0]))


def test_dedupe_tolerance_is_1e_9():
    cs = CandidateSet(
        dirs=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        speeds=(1.0,),
        include_zero=False,
    )
    # 当前速度与网格候选 (1,0,0) 差 5e-10 <= 1e-9 → 网格项去重
    cand = cs.build(V_MAX, np.array([1.0 + 5e-10, 0.0, 0.0]), np.zeros(3))
    assert cand.shape == (3, 3)  # 当前、v_pref、(0,1,0)
    # 差 2e-9 > 1e-9 → 网格项保留
    cand2 = cs.build(V_MAX, np.array([1.0 + 2e-9, 0.0, 0.0]), np.zeros(3))
    assert cand2.shape == (4, 3)
    assert np.isclose(CANDIDATE_DEDUPE_TOL, 1e-9)


def test_speed_layer_clipping_to_v_max(cset):
    cand = cset.build(1.5, np.array([0.3, 0.4, 0.0]), np.array([-0.6, 0.8, 0.0]))
    norms = np.linalg.norm(cand, axis=1)
    assert np.all(norms <= 1.5 + 1e-12)
    assert np.isclose(norms.max(), 1.5, rtol=0.0, atol=1e-12)
    # 层截断后去重：v_max=0.3 时三层都截断为 0.3 → 只有一层
    assert cset.effective_speeds(0.3) == (min(0.5, 0.3),)
    cand2 = cset.build(0.3, np.array([0.1, 0.2, 0.3]), np.array([-0.4, 0.5, 0.6]))
    assert cand2.shape == (128 + 3, 3)  # 1 层 × 128 方向 + 零/当前/v_pref
    assert np.all(np.linalg.norm(cand2, axis=1) <= 0.3 + 1e-12)
    # 默认 v_max 下三层完整保留
    assert cset.effective_speeds(V_MAX) == (0.5, 1.0, 2.0)
    assert cset.build(V_MAX, np.array([0.3, 0.4, 0.0]), np.array([-0.6, 0.8, 0.0])).shape[0] == 387


def test_build_index_map():
    cs = CandidateSet(dirs=np.array([[1.0, 0.0, 0.0]]), speeds=(0.5, 1.0))
    mapping = cs.build_index_map(5)
    expected = np.array(
        [
            [0, 0, 0],  # zero
            [1, 0, 0],  # current
            [2, 0, 0],  # pref
            [3, 0, 0],  # dir0 × 0.5
            [3, 0, 1],  # dir0 × 1.0
        ],
        dtype=np.int32,
    )
    assert np.array_equal(mapping, expected)
    with pytest.raises(ValueError):
        cs.build_index_map(3 + 1 * 2 + 1)


def test_build_deterministic_and_does_not_mutate_inputs(cset):
    vc = np.array([0.3, -0.2, 0.1])
    vp = np.array([-1.0, 0.5, 0.0])
    vc0 = vc.copy()
    vp0 = vp.copy()
    c1 = cset.build(V_MAX, vc, vp)
    c2 = cset.build(V_MAX, vc, vp)
    assert np.array_equal(c1, c2)
    assert np.array_equal(vc, vc0)
    assert np.array_equal(vp, vp0)
    assert c1.flags.writeable


def test_candidate_set_dirs_are_read_only(cset):
    assert not cset.dirs.flags.writeable
    with pytest.raises(ValueError):
        cset.dirs[0, 0] = 1.0


def test_scene_rotation_deterministic_and_orthonormal(default_cfg):
    cs1 = build_candidate_set(default_cfg, 1234)
    cs2 = build_candidate_set(default_cfg, 1234)
    assert np.array_equal(cs1.dirs, cs2.dirs)
    cs3 = build_candidate_set(default_cfg, 1235)
    assert not np.array_equal(cs1.dirs, cs3.dirs)
    # 同一 seed 对应同一旋转矩阵，且整体旋转由 STREAM_CANDIDATE 决定
    rot = candidate_rotation(1234)
    assert rot.shape == (3, 3)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-15)
    assert np.array_equal(cs1.dirs, rotate_directions(fibonacci_directions(128), rot))
    assert np.array_equal(cs1.dirs, CandidateSet.from_config(default_cfg, rotation=rot).dirs)


# ---------------------------------------------------------------------------
# 相对速度定义与危险判定（几何层）
# ---------------------------------------------------------------------------
def _single_candidate_danger(
    c, p, *, model="vo", v_i=None, v_j=None, frozen=False, tau=TAU, d_plan=D_PLAN
) -> bool:
    c = np.asarray(c, dtype=np.float64).reshape(1, 3)
    p = np.asarray(p, dtype=np.float64).reshape(1, 3)
    vi = np.zeros(3) if v_i is None else np.asarray(v_i, dtype=np.float64)
    vj = np.zeros((1, 3)) if v_j is None else np.asarray(v_j, dtype=np.float64).reshape(1, 3)
    flag = danger_flags(
        c,
        model=model,
        v_i=vi,
        v_j=vj,
        frozen_j=np.array([bool(frozen)]),
        p_ij=p,
        tau=tau,
        d_plan=d_plan,
    )
    return bool(flag[0])


def test_relative_velocity_model_definitions():
    cand = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vi = np.array([0.5, -0.5, 0.0])
    vj = np.array([[0.25, 0.0, 0.0], [-0.25, 0.5, 0.0]])
    frozen = np.array([False, True])

    w_vo = relative_velocities(cand, model="vo", v_i=vi, v_j=vj, frozen_j=frozen)
    assert np.array_equal(w_vo[:, 0, :], cand - vj[0][None, :])
    assert np.array_equal(w_vo[:, 1, :], cand)  # 冻结邻机 w = c

    w_rvo = relative_velocities(cand, model="rvo", v_i=vi, v_j=vj, frozen_j=frozen)
    assert np.array_equal(w_rvo[:, 0, :], 2.0 * cand - (vi + vj[0])[None, :])
    assert np.array_equal(w_rvo[:, 1, :], cand)  # 冻结邻机仍为 w = c


def test_danger_flags_match_direct_segment_definition():
    rng = np.random.default_rng(7)
    n_cand, m = 23, 11
    for model in ("vo", "rvo"):
        cand = rng.uniform(-2.0, 2.0, size=(n_cand, 3))
        vi = rng.uniform(-1.5, 1.5, size=3)
        vj = rng.uniform(-1.5, 1.5, size=(m, 3))
        frozen = rng.random(m) < 0.3
        p = rng.uniform(-15.0, 15.0, size=(m, 3))
        flags = danger_flags(
            cand,
            model=model,
            v_i=vi,
            v_j=vj,
            frozen_j=frozen,
            p_ij=p,
            tau=TAU,
            d_plan=D_PLAN,
        )
        w = relative_velocities(cand, model=model, v_i=vi, v_j=vj, frozen_j=frozen)
        md = min_closed_segment_distance(p, w, TAU)
        expected = np.any(md < D_PLAN, axis=1)
        assert np.array_equal(flags, expected), model
        mrd = min_relative_distances(
            cand, model=model, v_i=vi, v_j=vj, frozen_j=frozen, p_ij=p, tau=TAU
        )
        assert np.allclose(mrd, md, rtol=0.0, atol=1e-12), model


def test_known_hit_grazing_away_outside_window():
    # 精确命中：p=(-10,0,0), w=(-2,0,0)，接触 t=4.5 <= tau
    assert _single_candidate_danger([-2.0, 0.0, 0.0], [-10.0, 0.0, 0.0])
    # 远离：p=(-2,0,0), w=(+2,0,0)
    assert not _single_candidate_danger([2.0, 0.0, 0.0], [-2.0, 0.0, 0.0])
    # 预测窗外：p=(-20,0,0), w=(-2,0,0)，接触 t=9.5 > tau=5
    assert not _single_candidate_danger([-2.0, 0.0, 0.0], [-20.0, 0.0, 0.0])
    # 同一个几何把窗口放大到 10 s 就命中
    assert _single_candidate_danger(
        [-2.0, 0.0, 0.0], [-20.0, 0.0, 0.0], tau=10.0
    )
    # 擦边相切：p=(1,1,0), w=(1,0,0)，最近距离恰为 d_plan=1 → 严格 < 不成立 → 安全
    p = np.array([[1.0, 1.0, 0.0]])
    w = np.array([[[1.0, 0.0, 0.0]]])
    md = float(min_closed_segment_distance(p, w, 2.0)[0, 0])
    assert md == pytest.approx(1.0, rel=0.0, abs=1e-15)
    assert not _single_candidate_danger([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], tau=2.0)
    # 起始重叠：p=(0.5,0,0) < d_plan → t=0 即危险
    assert _single_candidate_danger([0.0, 0.0, 0.0], [0.5, 0.0, 0.0])


def test_danger_is_strict_below_d_plan():
    # 最近距离 1.2 → 安全；0.8 → 危险（同一相对速度方向，改变 y 偏移）
    assert not _single_candidate_danger([-2.0, 0.0, 0.0], [-10.0, 1.2, 0.0])
    assert _single_candidate_danger([-2.0, 0.0, 0.0], [-10.0, 0.8, 0.0])


# ---------------------------------------------------------------------------
# TTC 回退
# ---------------------------------------------------------------------------
def test_first_contact_times_analytic():
    p = np.array([[-10.0, 0.0, 0.0]])
    vj = np.zeros((1, 3))
    cand = np.array(
        [
            [-2.0, 0.0, 0.0],  # 迎面：t=4.5 进入 d_plan=1
            [2.0, 0.0, 0.0],  # 远离：无接触
            [0.0, 0.0, 0.0],  # 静止：无接触
            [-2.0, 2.0, 0.0],  # 斜向：最近距离 sqrt(50) → 无接触
        ]
    )
    ttc = first_contact_times(cand, vj, p, D_PLAN)
    assert ttc[0] == pytest.approx(4.5, rel=0.0, abs=1e-15)
    assert np.isinf(ttc[1])
    assert np.isinf(ttc[2])
    assert np.isinf(ttc[3])
    # 起始重叠 → 0
    ttc2 = first_contact_times(
        np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)), np.array([[0.5, 0.0, 0.0]]), D_PLAN
    )
    assert ttc2[0] == 0.0
    # 相切路径 min distance == d_plan：不构成严格接触 → +inf
    ttc3 = first_contact_times(
        np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)), np.array([[1.0, 1.0, 0.0]]), D_PLAN
    )
    assert np.isinf(ttc3[0])
    # 空邻机列表 → 全 +inf
    ttc4 = first_contact_times(
        np.array([[2.0, 0.0, 0.0]]), np.zeros((0, 3)), np.zeros((0, 3)), D_PLAN
    )
    assert np.isinf(ttc4[0])


def test_fallback_select_ttc_then_pref_then_index():
    p = np.array([[-10.0, 0.0, 0.0]])
    vj = np.zeros((1, 3))
    cand = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    # 候选 1/2 的 TTC 均为 +inf，d_pref 均为 4（v_pref=0）→ 固定索引 1 胜出
    idx, ttc = fallback_select(cand, vj, p, D_PLAN, np.zeros(3))
    assert idx == 1
    assert np.isinf(ttc)
    # 改变 v_pref 打破平局：候选 2 更接近 v_pref
    idx2, _ = fallback_select(cand, vj, p, D_PLAN, np.array([0.0, 2.0, 0.0]))
    assert idx2 == 2
    # 全部起始重叠 → TTC 全 0，按 v_pref 偏差选择
    p2 = np.array([[0.5, 0.0, 0.0]])
    idx3, ttc3 = fallback_select(cand, np.zeros((1, 3)), p2, D_PLAN, np.array([2.0, 0.0, 0.0]))
    assert idx3 == 1
    assert ttc3 == 0.0
    # 最大化最小 TTC 优先于 v_pref 偏差：候选 1 的 TTC 有限（4.5），候选 0 为 inf
    idx4, ttc4 = fallback_select(
        np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        vj,
        p,
        D_PLAN,
        np.array([-2.0, 0.0, 0.0]),  # v_pref 最接近候选 0
    )
    assert idx4 == 1
    assert np.isinf(ttc4)


# ---------------------------------------------------------------------------
# 偏好键 / 暴力选择
# ---------------------------------------------------------------------------
def test_preference_lexicographic_order():
    cand = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    vc = np.array([2.0, 3.0, 0.0])
    vp = np.array([2.0, 0.0, 0.0])
    order, d_pref, d_cur = preference_order(cand, vc, vp)
    assert np.array_equal(d_pref, np.array([1.0, 2.0, 0.0, 1.0]))
    assert np.array_equal(d_cur, np.array([10.0, 5.0, 9.0, 4.0]))
    # d_pref 主键：候选 2 (0)，然后 dpref=1 的候选 3/0（按 dcur 4<10），最后候选 1
    assert np.array_equal(order, np.array([2, 3, 0, 1]))
    dpref2, dcur2 = preference_keys(cand, vc, vp)
    assert np.array_equal(dpref2, d_pref)
    assert np.array_equal(dcur2, d_cur)
    assert np.array_equal(squared_distances(cand, vp), d_pref)


def test_select_exhaustive_lexicographic():
    cand = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    vc = np.array([0.0, 0.0, 0.0])
    vp = np.array([2.0, 0.0, 0.0])
    # 全部可接受：d_pref = 1, 0, 8 → 候选 1
    assert select_exhaustive(cand, np.zeros(3, dtype=bool), vc, vp) == 1
    # 候选 1 危险：候选 0（d_pref=1）胜出
    assert select_exhaustive(cand, np.array([False, True, False]), vc, vp) == 0
    # 无可接受候选
    assert select_exhaustive(cand, np.ones(3, dtype=bool), vc, vp) == -1
    # d_pref 相同时用 ||c-v_i||^2 打破平局（v_pref=0 → 两者 d_pref 均为 1）
    cand2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    vc2 = np.array([1.0, 2.5, 0.0])
    assert select_exhaustive(cand2, np.zeros(2, dtype=bool), vc2, np.zeros(3)) == 1
    # d_pref 与 d_cur 都相同 → 固定索引最小者
    assert select_exhaustive(cand2, np.zeros(2, dtype=bool), np.zeros(3), np.zeros(3)) == 0


# ---------------------------------------------------------------------------
# 加速搜索 vs 全量暴力：逐位一致（CONTRACTS §7 硬性要求）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["vo", "rvo"])
def test_accelerated_search_equals_exhaustive_bitwise(model, cset):
    rng = np.random.default_rng(20261030 if model == "vo" else 7)
    n_active = 4
    for trial in range(6):
        m = int(rng.integers(1, 8))
        p = rng.uniform(-12.0, 12.0, size=(n_active, m, 3))
        vj = rng.uniform(-1.2, 1.2, size=(n_active, m, 3))
        frozen = rng.random((n_active, m)) < 0.25
        mask = np.ones((n_active, m), dtype=bool)
        mask[rng.random((n_active, m)) < 0.1] = False
        mask[:, 0] = True  # 每机至少一个有效邻机
        vc = _rand_ball(rng, n_active)
        vp = _rand_ball(rng, n_active)
        result = search_velocities(
            cset,
            model=model,
            v_max=V_MAX,
            tau=TAU,
            d_plan=D_PLAN,
            v_current=vc,
            v_pref=vp,
            p_ij=p,
            v_j=vj,
            frozen_j=frozen,
            neighbor_mask=mask,
        )
        for a in range(n_active):
            sel = mask[a]
            expected, expected_exhausted = _exhaustive_search(
                cset,
                model=model,
                v_current=vc[a],
                v_pref=vp[a],
                p_ij=p[a][sel],
                v_j=vj[a][sel],
                frozen_j=frozen[a][sel],
            )
            assert np.array_equal(result.velocities[a], expected), (
                f"{model} trial={trial} agent={a}"
            )
            assert bool(result.sample_exhausted[a]) == expected_exhausted
            assert bool(result.used_fallback[a]) == expected_exhausted
            # 候选测试数有界且确定
            n_cand = cset.build(V_MAX, vc[a], vp[a]).shape[0]
            assert 1 <= int(result.candidate_count_tested[a]) <= n_cand
            if expected_exhausted:
                assert int(result.candidate_count_tested[a]) == n_cand


@pytest.mark.parametrize("model", ["vo", "rvo"])
def test_forced_fallback_equals_exhaustive_and_counts_all_candidates(model, cset):
    # 邻机全部落在 d_plan 内 → 任意候选在 t=0 即危险 → 必然 sample_exhausted
    dirs = fibonacci_directions(8)
    p = (dirs * 0.5)[None, :, :]  # (A=1, M=8, 3)
    m = p.shape[1]
    vc = np.zeros((1, 3))
    vp = np.array([[2.0, 0.0, 0.0]])
    res = search_velocities(
        cset,
        model=model,
        v_max=V_MAX,
        tau=TAU,
        d_plan=D_PLAN,
        v_current=vc,
        v_pref=vp,
        p_ij=p,
        v_j=np.zeros((1, m, 3)),
        frozen_j=np.zeros((1, m), dtype=bool),
        neighbor_mask=np.ones((1, m), dtype=bool),
    )
    assert bool(res.sample_exhausted[0]) is True
    assert bool(res.used_fallback[0]) is True
    cand = cset.build(V_MAX, vc[0], vp[0])
    assert int(res.candidate_count_tested[0]) == cand.shape[0]
    expected, expected_exhausted = _exhaustive_search(
        cset,
        model=model,
        v_current=vc[0],
        v_pref=vp[0],
        p_ij=p[0],
        v_j=np.zeros((m, 3)),
        frozen_j=np.zeros(m, dtype=bool),
    )
    assert expected_exhausted is True
    assert np.array_equal(res.velocities[0], expected)
    # 回退速度仍接受与主路径相同的安全检测（不会被标为安全）
    danger = danger_flags(
        res.velocities[0][None, :],
        model=model,
        v_i=vc[0],
        v_j=np.zeros((m, 3)),
        frozen_j=np.zeros(m, dtype=bool),
        p_ij=p[0],
        tau=TAU,
        d_plan=D_PLAN,
    )
    assert bool(danger[0]) is True


def test_pref_first_shortcut_tests_single_candidate(cset):
    # 单个 12 m 外迎面邻机：tau=5 时首次接触在 5.5 s（窗口外）→ v_pref 安全
    p = np.array([[[-12.0, 0.0, 0.0]]])
    m = 1
    res = search_velocities(
        cset,
        model="vo",
        v_max=V_MAX,
        tau=TAU,
        d_plan=D_PLAN,
        v_current=np.zeros((1, 3)),
        v_pref=np.array([[-2.0, 0.0, 0.0]]),
        p_ij=p,
        v_j=np.zeros((1, m, 3)),
        frozen_j=np.zeros((1, m), dtype=bool),
        neighbor_mask=np.ones((1, m), dtype=bool),
    )
    assert int(res.candidate_count_tested[0]) == 1
    assert not bool(res.sample_exhausted[0])
    # 逐位等于 v_pref（候选数组中的 v_pref 行）
    assert np.array_equal(res.velocities[0], np.array([-2.0, 0.0, 0.0]))
    # 但同一几何在 10 s 预测窗内首次接触 5.5 s → 参考判定为危险
    dangerous = danger_flags(
        np.array([[-2.0, 0.0, 0.0]]),
        model="vo",
        v_i=np.zeros(3),
        v_j=np.zeros((1, 3)),
        frozen_j=np.zeros(1, dtype=bool),
        p_ij=p[0],
        tau=10.0,
        d_plan=D_PLAN,
    )
    assert bool(dangerous[0]) is True


def test_search_independent_of_neighbor_order(cset):
    rng = np.random.default_rng(99)
    n_active, m = 3, 9
    p = rng.uniform(-8.0, 8.0, size=(n_active, m, 3))
    vj = rng.uniform(-1.0, 1.0, size=(n_active, m, 3))
    frozen = rng.random((n_active, m)) < 0.2
    mask = np.ones((n_active, m), dtype=bool)
    mask[:, -1] = False
    vc = _rand_ball(rng, n_active)
    vp = _rand_ball(rng, n_active)
    base = search_velocities(
        cset,
        model="rvo",
        v_max=V_MAX,
        tau=TAU,
        d_plan=D_PLAN,
        v_current=vc,
        v_pref=vp,
        p_ij=p,
        v_j=vj,
        frozen_j=frozen,
        neighbor_mask=mask,
    )
    perm = rng.permutation(m)
    shuffled = search_velocities(
        cset,
        model="rvo",
        v_max=V_MAX,
        tau=TAU,
        d_plan=D_PLAN,
        v_current=vc,
        v_pref=vp,
        p_ij=p[:, perm, :],
        v_j=vj[:, perm, :],
        frozen_j=frozen[:, perm],
        neighbor_mask=mask[:, perm],
    )
    assert np.array_equal(base.velocities, shuffled.velocities)
    assert np.array_equal(base.candidate_count_tested, shuffled.candidate_count_tested)
    assert np.array_equal(base.initial_overlap_pairs, shuffled.initial_overlap_pairs)


def test_search_no_neighbors(cset):
    n_active = 2
    res = search_velocities(
        cset,
        model="vo",
        v_max=V_MAX,
        tau=TAU,
        d_plan=D_PLAN,
        v_current=np.zeros((n_active, 3)),
        v_pref=np.array([[2.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        p_ij=np.zeros((n_active, 0, 3)),
        v_j=np.zeros((n_active, 0, 3)),
        frozen_j=np.zeros((n_active, 0), dtype=bool),
        neighbor_mask=np.zeros((n_active, 0), dtype=bool),
    )
    assert np.array_equal(res.velocities, np.array([[2.0, 0.0, 0.0], [0.0, -1.0, 0.0]]))
    assert np.array_equal(res.candidate_count_tested, np.ones(n_active, dtype=np.int32))
    assert not np.any(res.sample_exhausted)
    assert np.array_equal(res.initial_overlap_pairs, np.zeros(n_active, dtype=np.int32))
    assert np.array_equal(res.constraint_violation_max, np.zeros(n_active))


def test_initial_overlap_and_violation_diagnostics(cset):
    # 机0 与邻机重叠 0.5 m（< d_plan），机1 相距 10 m
    p = np.array([[[-0.5, 0.0, 0.0]], [[-10.0, 0.0, 0.0]]])
    m = 1
    res = search_velocities(
        cset,
        model="vo",
        v_max=V_MAX,
        tau=TAU,
        d_plan=D_PLAN,
        v_current=np.zeros((2, 3)),
        v_pref=np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        p_ij=p,
        v_j=np.zeros((2, m, 3)),
        frozen_j=np.zeros((2, m), dtype=bool),
        neighbor_mask=np.ones((2, m), dtype=bool),
    )
    assert res.initial_overlap_pairs.tolist() == [1, 0]
    # 机0 起始重叠 → 全部候选危险 → 回退
    assert bool(res.sample_exhausted[0]) is True
    assert bool(res.sample_exhausted[1]) is False
    # 返回速度对重叠机对的最小距离 = 0.5 → 残差 = 1 - 0.5 = 0.5
    assert res.constraint_violation_max[0] == pytest.approx(0.5, rel=0.0, abs=1e-12)
    assert res.constraint_violation_max[1] == 0.0
    assert np.all(np.isfinite(res.velocities))


def test_search_rejects_bad_shapes(cset):
    with pytest.raises(ValueError):
        search_velocities(
            cset,
            model="vo",
            v_max=V_MAX,
            tau=TAU,
            d_plan=D_PLAN,
            v_current=np.zeros((2, 2)),
            v_pref=np.zeros((2, 3)),
            p_ij=np.zeros((2, 1, 3)),
            v_j=np.zeros((2, 1, 3)),
            frozen_j=np.zeros((2, 1), dtype=bool),
            neighbor_mask=np.ones((2, 1), dtype=bool),
        )
    with pytest.raises(ValueError):
        search_velocities(
            cset,
            model="nope",
            v_max=V_MAX,
            tau=TAU,
            d_plan=D_PLAN,
            v_current=np.zeros((1, 3)),
            v_pref=np.zeros((1, 3)),
            p_ij=np.zeros((1, 1, 3)),
            v_j=np.zeros((1, 1, 3)),
            frozen_j=np.zeros((1, 1), dtype=bool),
            neighbor_mask=np.ones((1, 1), dtype=bool),
        )
