# OWNER: orca-dev
"""Geometric and behavioural tests for the 3D-ORCA planner (CONTRACTS.md §8).

Covers the handoff §8 ORCA row: head-on, parallel, oblique, fully symmetric, single
active agent vs a frozen sphere, mixed frozen/active neighbours, overlap recovery and an
empty feasible intersection; plus the experiment-specific adaptation rules
(alpha=0.5 active-active / alpha=1.0 active-frozen, no reciprocal frozen constraint),
neighbour-order independence, determinism and diagnostics semantics.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench.config import DEFAULT_CONFIG_PATH, load_config
from uavbench.navigation import STAGE_ARRIVED, STAGE_TO_GOAL
from uavbench.planners import orca3d
from uavbench.planners.base import PlanState
from uavbench.planners.orca3d import (
    ALPHA_ACTIVE_ACTIVE,
    ALPHA_ACTIVE_FROZEN,
    ORCA3DPlanner,
    build_orca_planes,
)

CFG = load_config(DEFAULT_CONFIG_PATH)
V_MAX = float(CFG.simulation.v_max)
D_PLAN = float(CFG.simulation.d_plan)
TAU = float(CFG.planners.orca.tau)
DT = float(CFG.experiment.dt)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def make_state(positions, velocities, v_pref, *, frozen_mask=None, dt: float = DT):
    """Build a PlanState whose neighbours are all other agents (frozen ones included)."""
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    n = pos.shape[0]
    frozen = (
        np.zeros(n, dtype=bool)
        if frozen_mask is None
        else np.asarray(frozen_mask, dtype=bool)
    )
    active = np.flatnonzero(~frozen).astype(np.int64)
    n_active = active.shape[0]
    vp_raw = np.asarray(v_pref, dtype=np.float64)
    if vp_raw.ndim == 1:
        vp_raw = vp_raw.reshape(1, 3)
    if vp_raw.shape[0] == n_active:
        vp = vp_raw.copy()
    elif vp_raw.shape[0] == n:
        vp = vp_raw[active].copy()
    else:
        raise ValueError(f"v_pref has {vp_raw.shape[0]} rows, expected {n_active} or {n}")
    stages = np.full(n, STAGE_ARRIVED, dtype=np.int8)
    stages[active] = STAGE_TO_GOAL
    width = max(n - 1, 0)
    neighbor_ids = np.full((n_active, width), -1, dtype=np.int64)
    neighbor_mask = np.zeros((n_active, width), dtype=bool)
    for a, i in enumerate(active):
        others = np.array([j for j in range(n) if j != i], dtype=np.int64)
        neighbor_ids[a, : others.size] = others
        neighbor_mask[a, : others.size] = True
    return PlanState(
        t=0.0,
        dt=dt,
        positions=pos,
        velocities=vel,
        stages=stages,
        frozen=frozen,
        active_ids=active,
        waypoints=np.zeros((n_active, 3), dtype=np.float64),
        v_pref=vp,
        neighbor_ids=neighbor_ids,
        neighbor_mask=neighbor_mask,
    )


def planes_for(state: PlanState, agent_index: int = 0):
    """Rebuild the planes of one active agent exactly as the planner does (sorted IDs)."""
    i = int(state.active_ids[agent_index])
    ids = state.neighbor_ids[agent_index][state.neighbor_mask[agent_index]]
    ids = np.sort(ids, kind="stable")
    frozen = state.frozen[ids]
    nbr_vel = np.where(frozen[:, None], 0.0, state.velocities[ids])
    return build_orca_planes(
        state.positions[i],
        state.velocities[i],
        state.positions[ids],
        nbr_vel,
        frozen,
        d_plan=D_PLAN,
        time_horizon=TAU,
        dt=state.dt,
        agent_id=i,
        neighbor_ids=ids,
    )


def residuals(planes, velocity) -> np.ndarray:
    return np.einsum("ij,ij->i", planes.normal, planes.point - np.asarray(velocity))


def max_residual(planes, velocity) -> float:
    r = residuals(planes, velocity)
    return float(np.max(np.maximum(0.0, r))) if r.size else 0.0


def speed(v) -> float:
    return float(np.sqrt(np.asarray(v, dtype=np.float64) @ np.asarray(v, dtype=np.float64)))


# --------------------------------------------------------------------------------------
# backend / constructor contract
# --------------------------------------------------------------------------------------
def test_constructor_and_backend_contract():
    planner = ORCA3DPlanner(CFG)
    assert planner.name == "orca"
    assert planner.d_plan == D_PLAN
    assert planner.v_max == V_MAX
    assert planner.time_horizon == TAU
    assert planner.backend == "python"
    assert orca3d.active_backend() == "python"
    assert orca3d.resolve_backend("auto") == "python"
    assert orca3d.resolve_backend("python") == "python"
    assert orca3d.numba_available() is False
    assert orca3d.numba_import_error()  # concrete reason recorded for provenance
    with pytest.raises(RuntimeError):
        orca3d.resolve_backend("numba")
    with pytest.raises(ValueError):
        orca3d.resolve_backend("bogus")
    # candidate_set is accepted and ignored (shared build_planner signature)
    planner2 = ORCA3DPlanner(CFG, candidate_set=object())
    assert planner2.name == "orca"


def test_no_neighbours_returns_clipped_pref():
    positions = np.array([[0.0, 0.0, 0.0]])
    velocities = np.zeros((1, 3))
    state = make_state(positions, velocities, [[1.0, 2.0, 0.0]])
    # drop the neighbour column entirely
    state = PlanState(
        t=state.t,
        dt=state.dt,
        positions=state.positions,
        velocities=state.velocities,
        stages=state.stages,
        frozen=state.frozen,
        active_ids=state.active_ids,
        waypoints=state.waypoints,
        v_pref=np.array([[1.0, 2.0, 0.0]]),
        neighbor_ids=np.zeros((1, 0), dtype=np.int64),
        neighbor_mask=np.zeros((1, 0), dtype=bool),
    )
    result = ORCA3DPlanner(CFG).plan(state)
    expected = np.array([1.0, 2.0, 0.0]) * (V_MAX / speed([1.0, 2.0, 0.0]))
    assert np.allclose(result.velocities[0], expected, atol=1e-12, rtol=0.0)
    d = result.diagnostics
    assert d.candidate_count_tested[0] == 0
    assert not d.used_fallback[0]
    assert d.constraint_violation_max[0] == 0.0
    assert d.extra["n_planes"][0] == 0


# --------------------------------------------------------------------------------------
# canonical geometric cases (handoff §8 ORCA row)
# --------------------------------------------------------------------------------------
def test_head_on_two_agents_fully_symmetric():
    """Exact head-on collision course: official code divides by zero, we must not NaN."""
    positions = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    velocities = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    v_pref = velocities.copy()
    state = make_state(positions, velocities, v_pref)
    result = ORCA3DPlanner(CFG).plan(state)
    v = result.velocities
    assert np.all(np.isfinite(v))
    assert speed(v[0]) <= V_MAX + 1e-8
    assert speed(v[1]) <= V_MAX + 1e-8
    # symmetry of the pair: v_0 == -v_1
    assert np.allclose(v[0], -v[1], atol=1e-12, rtol=0.0)
    # radial head-on must be broken laterally (deterministic degenerate direction)
    assert abs(v[0][1]) > 1e-3
    # both planes are the deterministic degenerate cone case
    assert result.diagnostics.extra["degenerate_planes"][0] == 1
    assert result.diagnostics.extra["degenerate_planes"][1] == 1
    # feasible instance: the ORCA halfspaces are satisfied
    for a in range(2):
        p = planes_for(state, a)
        assert max_residual(p, v[a]) <= 1e-8
    assert not result.diagnostics.numerical_error.any()


def test_parallel_pair_no_conflict():
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.1, 0.0]])
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    state = make_state(positions, velocities, velocities.copy())
    result = ORCA3DPlanner(CFG).plan(state)
    # moving side by side in the same direction is already safe: keep pref velocity
    assert np.allclose(result.velocities, velocities, atol=1e-12, rtol=0.0)
    assert not result.diagnostics.used_fallback.any()
    for a in range(2):
        p = planes_for(state, a)
        assert max_residual(p, result.velocities[a]) <= 1e-12


def test_oblique_feasible_pair_satisfies_halfspaces():
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 0.5]])
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    v_pref = np.array([[1.5, 0.0, 0.5], [-1.5, 0.0, -0.5]])
    state = make_state(positions, velocities, v_pref)
    result = ORCA3DPlanner(CFG).plan(state)
    v = result.velocities
    assert np.all(np.isfinite(v))
    for a in range(2):
        assert speed(v[a]) <= V_MAX + 1e-8
        p = planes_for(state, a)
        assert max_residual(p, v[a]) <= 1e-8
    assert not result.diagnostics.solver_infeasible.any()


def test_tetrahedron_fully_symmetric_four_agents():
    """4-agent strictly symmetric scene: outputs must stay finite, bounded and symmetric."""
    dirs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    positions = 5.0 * dirs
    velocities = np.zeros((4, 3))
    v_pref = -V_MAX * dirs  # everyone heads for its antipodal point
    state = make_state(positions, velocities, v_pref)
    result = ORCA3DPlanner(CFG).plan(state)
    v = result.velocities
    assert np.all(np.isfinite(v))
    speeds = np.array([speed(row) for row in v])
    assert np.all(speeds <= V_MAX + 1e-8)
    # tetrahedral symmetry: identical expected speed magnitude for all four agents
    assert np.allclose(speeds, speeds[0], atol=1e-12, rtol=0.0)
    for a in range(4):
        p = planes_for(state, a)
        assert max_residual(p, v[a]) <= 1e-8


# --------------------------------------------------------------------------------------
# frozen-sphere adaptation
# --------------------------------------------------------------------------------------
def test_alpha_active_vs_frozen_explicit_halfspace_geometry():
    """Explicit expected geometry for the same pair, alpha=0.5 vs alpha=1.0.

    Active agent at the origin with v_i=(1,0,0); neighbour at (3,0.5,0) with v_j=0.
    The official cone branch gives (independent scalar derivation)::

        n = (-0.1690658744178874, -0.9856047534926539, 0)
        u = (-0.028583269892685496, -0.16663212947966552, 0)   (n·u = 0.16906587441789106)
        point(alpha) = v_i + alpha*u

    so the active-active plane passes through v_i + 0.5u and the frozen plane through
    v_i + u.  With v_pref=(2,0,0) both constraints bind and the frozen case requires
    twice the velocity correction along n.
    """
    own_pos = np.zeros(3)
    own_vel = np.array([1.0, 0.0, 0.0])
    nbr_pos = np.array([[3.0, 0.5, 0.0]])
    nbr_vel = np.zeros((1, 3))
    expected_normal = np.array([-0.1690658744178874, -0.9856047534926539, 0.0])
    expected_u = np.array([-0.028583269892685496, -0.16663212947966552, 0.0])

    active_planes = build_orca_planes(
        own_pos, own_vel, nbr_pos, nbr_vel, np.array([False]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    frozen_planes = build_orca_planes(
        own_pos, own_vel, nbr_pos, nbr_vel, np.array([True]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    assert np.allclose(active_planes.normal[0], expected_normal, atol=1e-12, rtol=0.0)
    assert np.allclose(frozen_planes.normal[0], expected_normal, atol=1e-12, rtol=0.0)
    assert active_planes.alpha[0] == ALPHA_ACTIVE_ACTIVE == 0.5
    assert frozen_planes.alpha[0] == ALPHA_ACTIVE_FROZEN == 1.0
    assert np.allclose(
        active_planes.point[0] - own_vel, 0.5 * expected_u, atol=1e-12, rtol=0.0
    )
    assert np.allclose(
        frozen_planes.point[0] - own_vel, expected_u, atol=1e-12, rtol=0.0
    )
    # hard-coded explicit expected points (independent scalar derivation, float64)
    assert np.allclose(
        active_planes.point[0],
        [0.9857083650536572, -0.08331606473983276, 0.0],
        atol=1e-12,
        rtol=0.0,
    )
    assert np.allclose(
        frozen_planes.point[0],
        [0.9714167301073146, -0.16663212947966552, 0.0],
        atol=1e-12,
        rtol=0.0,
    )

    # Solver: the same v_pref violates both halfspaces, frozen needs full responsibility.
    pref = np.array([2.0, 0.0, 0.0])
    active_sol = orca3d.solve_orca_velocity(
        active_planes.normal, active_planes.point, v_max=V_MAX, v_pref=pref
    )
    frozen_sol = orca3d.solve_orca_velocity(
        frozen_planes.normal, frozen_planes.point, v_max=V_MAX, v_pref=pref
    )
    assert np.allclose(
        active_sol.velocity, [1.9571250951609724, -0.2499481942194946, 0.0],
        atol=1e-12, rtol=0.0,
    )
    assert np.allclose(
        frozen_sol.velocity, [1.9428334602146295, -0.3332642589593274, 0.0],
        atol=1e-12, rtol=0.0,
    )
    # frozen correction along n is exactly twice the active-active correction
    corr_active = float(active_planes.normal[0] @ (active_sol.velocity - own_vel))
    corr_frozen = float(frozen_planes.normal[0] @ (frozen_sol.velocity - own_vel))
    assert corr_frozen == pytest.approx(2.0 * corr_active, abs=1e-12)
    assert max_residual(active_planes, active_sol.velocity) <= 1e-12
    assert max_residual(frozen_planes, frozen_sol.velocity) <= 1e-12
    assert not active_sol.used_fallback
    assert not frozen_sol.used_fallback


def test_active_frozen_plan_differs_from_active_active_and_ignores_frozen_velocity():
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.5, 0.0]])
    velocities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    v_pref = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    state_frozen = make_state(positions, velocities, v_pref, frozen_mask=[False, True])
    res_frozen = ORCA3DPlanner(CFG).plan(state_frozen)
    assert state_frozen.active_ids.tolist() == [0]

    state_active = make_state(positions, velocities, v_pref)
    res_active = ORCA3DPlanner(CFG).plan(state_active)

    # agent 0 sees the same geometry, but alpha=1.0 in the frozen case requires the
    # active agent to take the full correction instead of half of it
    assert res_frozen.velocities[0][1] == pytest.approx(-0.33326425895932738, abs=1e-12)
    assert res_active.velocities[0][1] == pytest.approx(-0.24994819421949466, abs=1e-12)
    assert res_frozen.velocities[0][1] < res_active.velocities[0][1]
    assert not np.allclose(res_frozen.velocities[0], res_active.velocities[0])
    # active-active must emit the reciprocal plane as well: one plane per side
    assert res_active.diagnostics.candidate_count_tested.tolist() == [1, 1]
    # only the active side is planned in the frozen case
    assert res_frozen.diagnostics.candidate_count_tested.tolist() == [1]
    assert res_frozen.diagnostics.extra["n_frozen_neighbors"].tolist() == [1]

    # The frozen neighbour velocity is ignored (must be treated as exactly zero):
    # feeding a bogus frozen velocity must not change the plane.
    own_vel = np.array([1.0, 0.0, 0.0])
    nbr_pos = np.array([[3.0, 0.5, 0.0]])
    p_zero = build_orca_planes(
        np.zeros(3), own_vel, nbr_pos, np.zeros((1, 3)), np.array([True]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    p_bogus = build_orca_planes(
        np.zeros(3), own_vel, nbr_pos, np.array([[7.0, -4.0, 9.0]]), np.array([True]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    assert np.array_equal(p_zero.normal, p_bogus.normal)
    assert np.array_equal(p_zero.point, p_bogus.point)


def test_mixed_frozen_and_active_neighbours_plane_order_and_alpha():
    # agent 0 active; agent 1 active at +x; agent 2 frozen at +y
    positions = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
    )
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    v_pref = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    state = make_state(positions, velocities, v_pref, frozen_mask=[False, False, True])
    result = ORCA3DPlanner(CFG).plan(state)
    assert state.active_ids.tolist() == [0, 1]
    planes = planes_for(state, 0)
    # neighbour IDs are sorted ascending: [1 (active), 2 (frozen)]
    assert planes.normal.shape[0] == 2
    assert np.allclose(planes.alpha, [ALPHA_ACTIVE_ACTIVE, ALPHA_ACTIVE_FROZEN])
    assert result.diagnostics.candidate_count_tested.tolist() == [2, 2]
    assert result.diagnostics.initial_overlap_pairs.tolist() == [0, 0]
    assert result.diagnostics.extra["n_frozen_neighbors"].tolist() == [1, 1]
    v = result.velocities
    assert np.all(np.isfinite(v))
    for a in range(2):
        assert speed(v[a]) <= V_MAX + 1e-8
        p = planes_for(state, a)
        assert max_residual(p, v[a]) <= 1e-8


# --------------------------------------------------------------------------------------
# overlap recovery / empty feasible intersection
# --------------------------------------------------------------------------------------
def test_overlap_recovery_returns_separating_velocity():
    """Pair starts inside the combined radius: official invTimeStep recovery is used."""
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    v_pref = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])  # both push into each other
    state = make_state(positions, velocities, v_pref)
    result = ORCA3DPlanner(CFG).plan(state)
    v = result.velocities
    assert np.all(np.isfinite(v))
    assert result.diagnostics.initial_overlap_pairs.tolist() == [1, 1]
    planes = planes_for(state, 0)
    assert planes.n_overlap == 1
    # official overlap branch: normal along -x, point = v_i + 0.5*u = (-2.5, 0, 0)
    assert np.allclose(planes.normal[0], [-1.0, 0.0, 0.0], atol=1e-15, rtol=0.0)
    assert np.allclose(planes.point[0], [-2.5, 0.0, 0.0], atol=1e-15, rtol=0.0)
    # required speed 2.5 m/s > v_max: infeasible, official LP4 relaxation is entered
    assert result.diagnostics.used_fallback.tolist() == [True, True]
    assert result.diagnostics.solver_infeasible.tolist() == [True, True]
    assert not result.diagnostics.numerical_error.any()
    # returned velocities separate the pair at maximum speed
    relative_separation = float((v[1] - v[0]) @ (positions[1] - positions[0]))
    assert relative_separation > 0.0
    assert v[0][0] < 0.0 and v[1][0] > 0.0
    assert speed(v[0]) <= V_MAX + 1e-8 and speed(v[1]) <= V_MAX + 1e-8


def test_fully_coincident_agents_are_deterministic_and_antisymmetric():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    v_pref = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    state = make_state(positions, velocities, v_pref)
    planner = ORCA3DPlanner(CFG)
    res1 = planner.plan(state)
    res2 = planner.plan(state)
    assert np.array_equal(res1.velocities, res2.velocities)  # deterministic
    v = res1.velocities
    assert np.all(np.isfinite(v))
    assert np.allclose(v[0], -v[1], atol=1e-12, rtol=0.0)
    for a in range(2):
        assert speed(v[a]) <= V_MAX + 1e-8
    assert res1.diagnostics.extra["degenerate_planes"].tolist() == [1, 1]

    # the pair-ID recovery direction is antisymmetric when the pair is reversed
    d_02 = orca3d._pair_id_direction(0, 2)
    d_20 = orca3d._pair_id_direction(2, 0)
    assert np.array_equal(d_02, -d_20)
    assert d_02 @ d_02 == pytest.approx(1.0, abs=1e-15)
    assert orca3d._pair_id_direction(3, 7) @ orca3d._pair_id_direction(3, 9) != 0.0


def test_empty_feasible_intersection_sufficient_certificates():
    # (a) a plane requiring n·v > v_max can never be satisfied
    normal = np.array([[1.0, 0.0, 0.0]])
    point = np.array([[10.0, 0.0, 0.0]])
    sol = orca3d.solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=np.zeros(3))
    assert sol.used_fallback
    assert sol.solver_infeasible
    assert np.all(np.isfinite(sol.velocity))
    assert speed(sol.velocity) <= V_MAX + 1e-8

    # (b) two antiparallel planes with incompatible offsets: x >= 1.5 and x <= -0.5
    normal = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    point = np.array([[1.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    sol = orca3d.solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=np.zeros(3))
    assert sol.used_fallback
    assert sol.solver_infeasible
    assert np.all(np.isfinite(sol.velocity))

    # a genuinely feasible instance is never flagged infeasible
    normal = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    point = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    sol = orca3d.solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=np.zeros(3))
    assert not sol.used_fallback
    assert not sol.solver_infeasible
    assert sol.constraint_violation_max == 0.0


# --------------------------------------------------------------------------------------
# order independence / determinism / diagnostics
# --------------------------------------------------------------------------------------
def test_neighbour_order_independence(rng):
    n = 6
    positions = rng.standard_normal((n, 3)) * 3.0
    velocities = rng.standard_normal((n, 3)) * 0.5
    v_pref = rng.standard_normal((n, 3))
    v_pref *= np.minimum(1.0, V_MAX / np.linalg.norm(v_pref, axis=1))[:, None]
    frozen = np.zeros(n, dtype=bool)
    frozen[-1] = True
    velocities[frozen] = 0.0
    n_active = n - 1
    active = np.flatnonzero(~frozen).astype(np.int64)
    width = n - 1
    ids = np.full((n_active, width), -1, dtype=np.int64)
    mask = np.zeros((n_active, width), dtype=bool)
    for a, i in enumerate(active):
        others = np.array([j for j in range(n) if j != i], dtype=np.int64)
        ids[a, : others.size] = others
        mask[a, : others.size] = True

    def build(neighbor_ids):
        return PlanState(
            t=0.0, dt=DT, positions=positions, velocities=velocities,
            stages=np.where(frozen, STAGE_ARRIVED, STAGE_TO_GOAL).astype(np.int8),
            frozen=frozen, active_ids=active,
            waypoints=np.zeros((n_active, 3)), v_pref=v_pref,
            neighbor_ids=neighbor_ids, neighbor_mask=mask,
        )

    planner = ORCA3DPlanner(CFG)
    base = planner.plan(build(ids))
    # reverse and shuffle every neighbour list
    for seed in (1, 2, 3):
        perm = np.array(ids, copy=True)
        rng2 = np.random.default_rng(seed)
        for a in range(n_active):
            valid = perm[a][mask[a]]
            rng2.shuffle(valid)
            perm[a][mask[a]] = valid
        other = planner.plan(build(perm))
        assert np.array_equal(base.velocities, other.velocities)
        assert np.array_equal(
            base.diagnostics.constraint_violation_max,
            other.diagnostics.constraint_violation_max,
        )
        assert np.array_equal(
            base.diagnostics.used_fallback, other.diagnostics.used_fallback
        )


def test_plan_does_not_mutate_state_and_repeat_is_bitwise_identical():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [1.0, -2.0, 0.5]])
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v_pref = velocities.copy()
    state = make_state(positions, velocities, v_pref, frozen_mask=[False, False, True])
    saved = (
        state.positions.copy(), state.velocities.copy(), state.stages.copy(),
        state.frozen.copy(), state.v_pref.copy(), state.neighbor_ids.copy(),
    )
    planner = ORCA3DPlanner(CFG)
    first = planner.plan(state)
    second = planner.plan(state)
    assert np.array_equal(first.velocities, second.velocities)
    for a, b in zip(
        saved,
        (state.positions, state.velocities, state.stages, state.frozen,
         state.v_pref, state.neighbor_ids),
    ):
        assert np.array_equal(a, b)


def test_diagnostics_shapes_values_and_extra_keys():
    n = 4
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]
    )
    velocities = np.zeros((n, 3))
    v_pref = np.tile([0.5, 0.0, 0.0], (n, 1))
    state = make_state(positions, velocities, v_pref, frozen_mask=[False, False, True, True])
    result = ORCA3DPlanner(CFG).plan(state)
    d = result.diagnostics
    n_active = state.n_active
    assert result.velocities.shape == (n_active, 3)
    assert np.all(np.isfinite(result.velocities))
    assert d.sample_exhausted.shape == (n_active,)
    assert d.sample_exhausted.dtype == bool
    assert d.candidate_count_tested.shape == (n_active,)
    assert d.extra is not None
    for key in ("n_planes", "n_frozen_neighbors", "degenerate_planes"):
        assert key in d.extra
        assert d.extra[key].shape == (n_active,)
    assert d.candidate_count_tested.tolist() == [3, 3]
    assert d.initial_overlap_pairs.tolist() == [0, 0]
    assert d.extra["n_frozen_neighbors"].tolist() == [2, 2]
    assert np.array_equal(d.extra["n_planes"], d.candidate_count_tested)
    assert np.all(d.constraint_violation_max <= 1e-8)
    assert not d.solver_infeasible.any()
    assert not d.numerical_error.any()


def test_no_frozen_agent_is_ever_planned():
    n = 5
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
        ]
    )
    velocities = np.zeros((n, 3))
    frozen = np.array([False, True, True, True, True])
    state = make_state(positions, velocities, [[1.0, 0.0, 0.0]], frozen_mask=frozen)
    result = ORCA3DPlanner(CFG).plan(state)
    assert state.active_ids.tolist() == [0]
    assert result.velocities.shape == (1, 3)
    # frozen neighbours contribute exactly one plane each (no reciprocal planes)
    assert result.diagnostics.candidate_count_tested.tolist() == [4]
    assert result.diagnostics.extra["n_frozen_neighbors"].tolist() == [4]
    # frozen agents' velocities are untouched by the planner
    assert np.array_equal(state.velocities[1:], np.zeros((4, 3)))


def test_dense_random_configuration_is_finite_and_bounded(rng):
    n = 12
    positions = rng.uniform(-1.0, 1.0, size=(n, 3))  # deliberately overlapping crowd
    velocities = rng.uniform(-0.5, 0.5, size=(n, 3))
    v_pref = rng.standard_normal((n, 3))
    v_pref *= np.minimum(1.0, V_MAX / np.linalg.norm(v_pref, axis=1))[:, None]
    state = make_state(positions, velocities, v_pref)
    result = ORCA3DPlanner(CFG).plan(state)
    v = result.velocities
    assert np.all(np.isfinite(v))
    speeds = np.linalg.norm(v, axis=1)
    assert np.all(speeds <= V_MAX + 1e-8)
    d = result.diagnostics
    assert np.all(d.constraint_violation_max >= 0.0)
    # when the LP reports a feasible point, every halfspace residual must be satisfied
    for a in range(n):
        if not d.used_fallback[a]:
            planes = planes_for(state, a)
            assert max_residual(planes, v[a]) <= 1e-8
    assert not d.numerical_error.any()
