# OWNER: orca-dev
"""Numeric verification for the 3D-ORCA planner (CONTRACTS.md §8, handoff §8 numeric row).

Independence strategy
---------------------
* A literal, line-by-line float64 scalar transcription of upstream RVO2-3D
  ``Agent::computeNewVelocity`` / ``linearProgram1..4`` lives in this test module and is
  used as the differential reference for the vectorised production solver.  It is a
  *self-consistent reference port*, not a second copy of production code: it is written
  directly from the official C++ structure with explicit per-plane loops.
* Optimality of the projection ``min ||v - v_pref||^2`` over ``||v|| <= v_max`` and the
  ORCA half-spaces is checked against ``scipy.optimize.minimize`` (SLSQP) as an
  independent convex solver.
* Feasible/infeasible classification is checked against explicit construction and the
  cheap sufficient certificates only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from uavbench.config import DEFAULT_CONFIG_PATH, load_config
from uavbench.planners import orca3d
from uavbench.planners.orca3d import (
    RVO3D_EPSILON,
    build_orca_planes,
    solve_orca_velocity,
)

CFG = load_config(DEFAULT_CONFIG_PATH)
V_MAX = float(CFG.simulation.v_max)
D_PLAN = float(CFG.simulation.d_plan)
TAU = float(CFG.planners.orca.tau)
DT = float(CFG.experiment.dt)
EPS = RVO3D_EPSILON


# ======================================================================================
# Literal float64 scalar port of the official RVO2-3D LP (independent reference)
# ======================================================================================
def _ref_lp1(planes, plane_no, line_point, line_direction, radius, opt_velocity, direction_opt):
    dot_product = float(line_point @ line_direction)
    discriminant = (
        dot_product * dot_product + radius * radius - float(line_point @ line_point)
    )
    if discriminant < 0.0:
        return False, None
    sqrt_disc = math.sqrt(discriminant)
    t_left = -dot_product - sqrt_disc
    t_right = -dot_product + sqrt_disc
    for k in range(plane_no):
        n_k, p_k = planes[k]
        numerator = float((p_k - line_point) @ n_k)
        denominator = float(line_direction @ n_k)
        if denominator * denominator <= EPS:
            if numerator > 0.0:
                return False, None
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_left = max(t_left, t)
        else:
            t_right = min(t_right, t)
        if t_left > t_right:
            return False, None
    if direction_opt:
        if float(opt_velocity @ line_direction) > 0.0:
            result = line_point + t_right * line_direction
        else:
            result = line_point + t_left * line_direction
    else:
        t = float(line_direction @ (opt_velocity - line_point))
        if t < t_left:
            result = line_point + t_left * line_direction
        elif t > t_right:
            result = line_point + t_right * line_direction
        else:
            result = line_point + t * line_direction
    return True, result


def _ref_lp2(planes, plane_no, radius, opt_velocity, direction_opt):
    n_i, p_i = planes[plane_no]
    plane_dist = float(p_i @ n_i)
    radius_sq = radius * radius
    if plane_dist * plane_dist > radius_sq:
        return False, None
    plane_radius_sq = radius_sq - plane_dist * plane_dist
    plane_center = plane_dist * n_i
    if direction_opt:
        plane_opt = opt_velocity - float(opt_velocity @ n_i) * n_i
        plane_opt_length_sq = float(plane_opt @ plane_opt)
        if plane_opt_length_sq <= EPS:
            result = plane_center.copy()
        else:
            result = plane_center + math.sqrt(
                plane_radius_sq / plane_opt_length_sq
            ) * plane_opt
    else:
        result = opt_velocity + float((p_i - opt_velocity) @ n_i) * n_i
        if float(result @ result) > radius_sq:
            plane_result = result - plane_center
            result = plane_center + math.sqrt(
                plane_radius_sq / float(plane_result @ plane_result)
            ) * plane_result
    for k in range(plane_no):
        n_k, p_k = planes[k]
        if float(n_k @ (p_k - result)) > 0.0:
            cross_product = np.cross(n_k, n_i)
            if float(cross_product @ cross_product) <= EPS:
                return False, None
            line_direction = cross_product / math.sqrt(float(cross_product @ cross_product))
            line_normal = np.cross(line_direction, n_i)
            line_point = p_i + (
                float((p_k - p_i) @ n_k) / float(line_normal @ n_k)
            ) * line_normal
            ok, result = _ref_lp1(
                planes, k, line_point, line_direction, radius, opt_velocity, direction_opt
            )
            if not ok:
                return False, None
    return True, result


def _ref_lp3(planes, radius, opt_velocity, direction_opt):
    if direction_opt:
        result = opt_velocity * radius
    elif float(opt_velocity @ opt_velocity) > radius * radius:
        result = opt_velocity * (radius / math.sqrt(float(opt_velocity @ opt_velocity)))
    else:
        result = np.array(opt_velocity, dtype=np.float64, copy=True)
    for i in range(len(planes)):
        n_i, p_i = planes[i]
        if float(n_i @ (p_i - result)) > 0.0:
            temp_result = np.array(result, copy=True)
            ok, result = _ref_lp2(planes, i, radius, opt_velocity, direction_opt)
            if not ok:
                return i, temp_result
    return len(planes), result


def _ref_lp4(planes, begin_plane, radius, result):
    distance = 0.0
    for i in range(begin_plane, len(planes)):
        n_i, p_i = planes[i]
        if float(n_i @ (p_i - result)) > distance:
            proj_planes = []
            for j in range(i):
                n_j, p_j = planes[j]
                cross_product = np.cross(n_j, n_i)
                if float(cross_product @ cross_product) <= EPS:
                    if float(n_i @ n_j) > 0.0:
                        continue
                    proj_point = 0.5 * (p_i + p_j)
                else:
                    line_normal = np.cross(cross_product, n_i)
                    proj_point = p_i + (
                        float((p_j - p_i) @ n_j) / float(line_normal @ n_j)
                    ) * line_normal
                proj_normal = n_j - n_i
                proj_planes.append(
                    (proj_normal / math.sqrt(float(proj_normal @ proj_normal)), proj_point)
                )
            temp_result = np.array(result, copy=True)
            fail, candidate = _ref_lp3(proj_planes, radius, n_i, True)
            result = temp_result if fail < len(proj_planes) else candidate
            distance = float(n_i @ (p_i - result))
    return result


def _ref_compute_velocity(planes, radius, pref):
    """Literal reference of ``Agent::computeNewVelocity`` after plane construction."""
    fail, result = _ref_lp3(planes, radius, pref, False)
    if fail < len(planes):
        result = _ref_lp4(planes, fail, radius, result)
    return fail, result


def _ref_pair_plane(own_pos, own_vel, nbr_pos, nbr_vel, *, d_plan=D_PLAN, tau=TAU, dt=DT,
                    alpha=0.5):
    """Literal float64 transcription of the official active-active plane construction."""
    rel_pos = np.asarray(nbr_pos, dtype=np.float64) - np.asarray(own_pos, dtype=np.float64)
    rel_vel = np.asarray(own_vel, dtype=np.float64) - np.asarray(nbr_vel, dtype=np.float64)
    dist_sq = float(rel_pos @ rel_pos)
    combined_radius = float(d_plan)
    combined_radius_sq = combined_radius * combined_radius
    inv_time_horizon = 1.0 / float(tau)
    if dist_sq > combined_radius_sq:
        w = rel_vel - inv_time_horizon * rel_pos
        w_length_sq = float(w @ w)
        dot_product = float(w @ rel_pos)
        if dot_product < 0.0 and (
            dot_product * dot_product > combined_radius_sq * w_length_sq
        ):
            w_length = math.sqrt(w_length_sq)
            unit_w = w / w_length
            normal = unit_w
            u = (combined_radius * inv_time_horizon - w_length) * unit_w
        else:
            a = dist_sq
            b = float(rel_pos @ rel_vel)
            cross_product = np.cross(rel_pos, rel_vel)
            c = float(rel_vel @ rel_vel) - float(
                cross_product @ cross_product
            ) / (dist_sq - combined_radius_sq)
            t = (b + math.sqrt(b * b - a * c)) / a
            ww = rel_vel - t * rel_pos
            ww_length = math.sqrt(float(ww @ ww))
            unit_ww = ww / ww_length
            normal = unit_ww
            u = (combined_radius * t - ww_length) * unit_ww
    else:
        inv_time_step = 1.0 / float(dt)
        w = rel_vel - inv_time_step * rel_pos
        w_length = math.sqrt(float(w @ w))
        unit_w = w / w_length
        normal = unit_w
        u = (combined_radius * inv_time_step - w_length) * unit_w
    return normal, np.asarray(own_vel, dtype=np.float64) + alpha * u, u


# ======================================================================================
# instance generators
# ======================================================================================
def _random_unit(rng) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def _random_feasible_instance(rng, m: int):
    """Random half-space system with a known strictly feasible point v0."""
    v0 = rng.standard_normal(3)
    v0 *= rng.uniform(0.0, 0.8 * V_MAX) / max(np.linalg.norm(v0), 1e-12)
    planes = []
    for _ in range(m):
        n = _random_unit(rng)
        margin = rng.uniform(0.05, 0.6)
        b = float(n @ v0) - margin
        planes.append((n, b * n))
    pref = _random_unit(rng) * rng.uniform(0.8, 1.4) * V_MAX
    return planes, pref, v0


def _planes_to_arrays(planes):
    if not planes:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (
        np.array([p[0] for p in planes], dtype=np.float64),
        np.array([p[1] for p in planes], dtype=np.float64),
    )


def _objective(velocity, pref) -> float:
    d = np.asarray(velocity, dtype=np.float64) - np.asarray(pref, dtype=np.float64)
    return float(d @ d)


def _max_residual(planes, velocity) -> float:
    if not planes:
        return 0.0
    return max(float(np.maximum(0.0, n @ (p - velocity))) for n, p in planes)


def _slsqp_reference(planes, pref, v_max, x0=None):
    """Independent convex reference: min ||v-pref||^2 over ORCA half-spaces and ball."""
    normals = np.array([p[0] for p in planes]) if planes else np.zeros((0, 3))
    offsets = np.array([float(p[0] @ p[1]) for p in planes]) if planes else np.zeros(0)

    def fun(v):
        d = v - pref
        return float(d @ d)

    def jac(v):
        return 2.0 * (v - pref)

    constraints = [
        {
            "type": "ineq",
            "fun": (lambda v, n=normals[i], b=offsets[i]: float(n @ v - b)),
            "jac": (lambda v, n=normals[i]: n),
        }
        for i in range(len(planes))
    ]
    constraints.append(
        {
            "type": "ineq",
            "fun": lambda v: float(v_max * v_max - v @ v),
            "jac": lambda v: -2.0 * v,
        }
    )
    starts = []
    if x0 is not None:
        starts.append(np.array(x0, dtype=np.float64, copy=True))
    starts.append(np.array(pref, dtype=np.float64, copy=True))
    for start in starts:
        norm = float(np.linalg.norm(start))
        if norm > v_max:
            start *= v_max / norm
        result = minimize(
            fun,
            start,
            jac=jac,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success:
            return result
    return result


# ======================================================================================
# (a) feasible random instances: residuals, speed limit, objective optimality
# ======================================================================================
@pytest.mark.parametrize("seed", range(12))
def test_feasible_random_instances_residuals_speed_and_optimality(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(8):
        m = int(rng.integers(1, 6))
        planes, pref, v0 = _random_feasible_instance(rng, m)
        normal, point = _planes_to_arrays(planes)
        solution = solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=pref)
        v = solution.velocity
        assert np.all(np.isfinite(v))
        assert float(np.linalg.norm(v)) <= V_MAX + 1e-8
        residual = _max_residual(planes, v)
        assert residual <= 1e-8, f"constraint residual {residual!r} for feasible instance"
        assert solution.constraint_violation_max <= 1e-8
        # feasible instance: official LP3 succeeds, no fallback and no infeasibility flag
        assert not solution.used_fallback
        assert not solution.solver_infeasible
        assert not solution.numerical_error
        # independent convex reference, started from the known feasible point
        ref = _slsqp_reference(planes, pref, V_MAX, x0=v0)
        assert ref.success, ref.message
        objective = _objective(v, pref)
        assert objective == pytest.approx(float(ref.fun), abs=1e-6, rel=0.0)
        # the optimum itself must be feasible w.r.t. the ball too
        assert _max_residual(planes, ref.x) <= 1e-6


def test_random_feasible_instances_pref_inside_ball_and_outside():
    rng = np.random.default_rng(7)
    for m in (0, 1, 2, 3, 5, 8):
        planes, _, v0 = _random_feasible_instance(rng, m)
        normal, point = _planes_to_arrays(planes)
        for pref_scale in (0.0, 0.6, 1.0, 1.3):
            pref = _random_unit(rng) * pref_scale * V_MAX
            sol = solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=pref)
            assert np.linalg.norm(sol.velocity) <= V_MAX + 1e-8
            assert _max_residual(planes, sol.velocity) <= 1e-8
            ref = _slsqp_reference(planes, pref, V_MAX, x0=v0)
            assert ref.success
            assert _objective(sol.velocity, pref) == pytest.approx(
                float(ref.fun), abs=1e-6, rel=0.0
            )


# ======================================================================================
# (b) differential check against the literal scalar port of the official algorithm
# ======================================================================================
def test_differential_against_literal_scalar_port_on_random_plane_sets():
    rng = np.random.default_rng(2026)
    checked_feasible = 0
    checked_infeasible = 0
    for _ in range(400):
        m = int(rng.integers(0, 9))
        planes = []
        for _ in range(m):
            n = _random_unit(rng)
            # offsets spread well beyond the max-speed ball: both feasible and
            # infeasible subsets occur
            planes.append((n, rng.uniform(-3.5, 3.5) * n))
        pref = _random_unit(rng) * rng.uniform(0.0, 1.5) * V_MAX
        normal, point = _planes_to_arrays(planes)

        fail_prod, res_prod = orca3d._linear_program3(
            normal, point, V_MAX, pref, False
        )
        fail_ref, res_ref = _ref_lp3(planes, V_MAX, pref, False)
        assert fail_prod == fail_ref
        if fail_prod < m:
            checked_infeasible += 1
            res_prod = orca3d._linear_program4(
                normal, point, fail_prod, V_MAX, res_prod.copy()
            )
            res_ref = _ref_lp4(planes, fail_ref, V_MAX, res_ref.copy())
        else:
            checked_feasible += 1
        assert np.allclose(res_prod, res_ref, atol=1e-9, rtol=0.0)

        solution = solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=pref)
        if not solution.used_fallback:
            assert np.all(np.isfinite(solution.velocity))
    assert checked_feasible > 0 and checked_infeasible > 0


def test_differential_pair_plane_construction_against_official_scalar_port():
    rng = np.random.default_rng(31)
    for _ in range(300):
        own_pos = rng.standard_normal(3) * rng.uniform(0.0, 3.0)
        own_vel = rng.standard_normal(3) * rng.uniform(0.0, 2.0)
        nbr_pos = own_pos + rng.standard_normal(3) * rng.uniform(0.0, 6.0)
        nbr_vel = rng.standard_normal(3) * rng.uniform(0.0, 2.0)
        ref_n, ref_point, ref_u = _ref_pair_plane(own_pos, own_vel, nbr_pos, nbr_vel)
        if not np.all(np.isfinite(ref_n)):  # official 0/0 cases are covered separately
            continue
        planes = build_orca_planes(
            own_pos,
            own_vel,
            nbr_pos.reshape(1, 3),
            nbr_vel.reshape(1, 3),
            np.array([False]),
            d_plan=D_PLAN,
            time_horizon=TAU,
            dt=DT,
            agent_id=0,
            neighbor_ids=np.array([1]),
        )
        if planes.degenerate[0]:
            continue
        assert np.allclose(planes.normal[0], ref_n, atol=1e-12, rtol=0.0)
        assert np.allclose(planes.point[0], ref_point, atol=1e-12, rtol=0.0)
        # the vector u itself is implied by point = v_i + 0.5 u
        assert np.allclose(2.0 * (planes.point[0] - own_vel), ref_u, atol=1e-12, rtol=0.0)


def test_solve_orca_velocity_equals_official_id_order_semantics():
    """The violation-ordered feasible pre-pass must not change the official semantics.

    * feasible systems: the result is the exact projection of v_pref, i.e. it agrees with
      the literal ID-order official LP3 result (tolerance 1e-9 = different rounding path);
    * infeasible systems: the production solver falls back to the unchanged official
      ID-order LP3 -> LP4 path, so the result must agree with the literal reference.
    """
    rng = np.random.default_rng(777)
    n_feasible = 0
    n_infeasible = 0
    for _ in range(200):
        m = int(rng.integers(3, 9))
        planes = []
        for _ in range(m):
            n = _random_unit(rng)
            planes.append((n, rng.uniform(-3.0, 3.0) * n))
        pref = _random_unit(rng) * rng.uniform(0.0, 1.4) * V_MAX
        normal, point = _planes_to_arrays(planes)
        solution = solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=pref)
        fail_ref, ref_result = _ref_compute_velocity(planes, V_MAX, pref)
        if fail_ref == m:
            n_feasible += 1
            assert not solution.used_fallback
            assert np.allclose(solution.velocity, ref_result, atol=1e-9, rtol=0.0)
            assert _max_residual(planes, solution.velocity) <= 1e-8
        else:
            n_infeasible += 1
            assert solution.used_fallback
            ref_lp4 = _ref_lp4(planes, fail_ref, V_MAX, ref_result.copy())
            assert np.allclose(solution.velocity, ref_lp4, atol=1e-9, rtol=0.0)
    assert n_feasible > 0 and n_infeasible > 0


# ======================================================================================
# (c) infeasibility detection: sufficient certificates only
# ======================================================================================
def test_infeasibility_requires_sufficient_certificate():
    # certificate (a): a single plane whose required value exceeds v_max
    n = np.array([[1.0, 0.0, 0.0]])
    p = np.array([[V_MAX + 0.5, 0.0, 0.0]])
    sol = solve_orca_velocity(n, p, v_max=V_MAX, v_pref=np.zeros(3))
    assert sol.used_fallback and sol.solver_infeasible
    assert np.all(np.isfinite(sol.velocity))

    # certificate (b): antiparallel planes with incompatible offsets
    n = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    p = np.array([[1.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    sol = solve_orca_velocity(n, p, v_max=V_MAX, v_pref=np.zeros(3))
    assert sol.used_fallback and sol.solver_infeasible

    # a non-certificate failure (tight but feasible) must NOT be labelled infeasible
    rng = np.random.default_rng(5)
    for _ in range(50):
        planes, pref, _ = _random_feasible_instance(rng, 2)
        normal, point = _planes_to_arrays(planes)
        sol = solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=pref)
        if sol.used_fallback:  # numerical edge case: still not a proof of infeasibility
            assert not sol.solver_infeasible or orca3d._infeasibility_certificate(
                normal, point, V_MAX
            )


def test_constructed_infeasible_instance_from_incompatible_orbits():
    """Two frozen neighbours on opposite sides demand mutually exclusive velocities."""
    # Build planes by hand: n·v >= 1.5 and -n·v >= 1.5  ->  v_x >= 1.5 and v_x <= -1.5
    normal = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    point = np.array([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]])
    solution = solve_orca_velocity(normal, point, v_max=V_MAX, v_pref=np.zeros(3))
    assert solution.used_fallback
    assert solution.solver_infeasible
    assert np.all(np.isfinite(solution.velocity))
    assert np.linalg.norm(solution.velocity) <= V_MAX + 1e-8
    # official LP4 relaxation is a min-max-violation heuristic.  For this 1-D system the
    # min-max solution is v_x = 0 with max violation 1.5 (symmetric, both planes share
    # the unavoidable violation).  Check it against a brute-force 1-D minimisation.
    assert solution.constraint_violation_max == pytest.approx(1.5, abs=1e-9)
    grid = np.linspace(-V_MAX, V_MAX, 20001)
    best = np.min(np.maximum(1.5 - grid, 1.5 + grid))
    assert solution.constraint_violation_max == pytest.approx(float(best), abs=1e-3)


# ======================================================================================
# (d) symmetry / permutation / translation invariance
# ======================================================================================
def _plan_configuration(planner, positions, velocities, v_pref, frozen=None):
    from uavbench.navigation import STAGE_ARRIVED, STAGE_TO_GOAL
    from uavbench.planners.base import PlanState

    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    n = pos.shape[0]
    frozen = np.zeros(n, dtype=bool) if frozen is None else np.asarray(frozen, dtype=bool)
    active = np.flatnonzero(~frozen).astype(np.int64)
    n_active = active.shape[0]
    width = max(n - 1, 0)
    ids = np.full((n_active, width), -1, dtype=np.int64)
    mask = np.zeros((n_active, width), dtype=bool)
    for a, i in enumerate(active):
        others = np.array([j for j in range(n) if j != i], dtype=np.int64)
        ids[a, : others.size] = others
        mask[a, : others.size] = True
    stages = np.full(n, STAGE_ARRIVED, dtype=np.int8)
    stages[active] = STAGE_TO_GOAL
    state = PlanState(
        t=0.0,
        dt=DT,
        positions=pos,
        velocities=vel,
        stages=stages,
        frozen=frozen,
        active_ids=active,
        waypoints=np.zeros((n_active, 3)),
        v_pref=np.asarray(v_pref, dtype=np.float64).reshape(n_active, 3),
        neighbor_ids=ids,
        neighbor_mask=mask,
    )
    return planner.plan(state)


def test_rotation_covariance_of_planner(rng):
    from uavbench import geometry

    n = 6
    positions = rng.standard_normal((n, 3)) * 2.5
    velocities = rng.standard_normal((n, 3)) * 0.4
    v_pref = rng.standard_normal((n, 3))
    v_pref *= np.minimum(1.0, V_MAX / np.linalg.norm(v_pref, axis=1))[:, None]
    planner = orca3d.ORCA3DPlanner(CFG)
    base = _plan_configuration(planner, positions, velocities, v_pref)
    assert not base.diagnostics.used_fallback.any()  # generic feasible instance
    for _ in range(5):
        rot = geometry.random_rotation_matrix(rng)
        rotated = _plan_configuration(
            planner, positions @ rot.T, velocities @ rot.T, v_pref @ rot.T
        )
        assert np.allclose(rotated.velocities, base.velocities @ rot.T, atol=1e-9, rtol=0.0)
        assert np.array_equal(
            rotated.diagnostics.used_fallback, base.diagnostics.used_fallback
        )


def test_translation_invariance_of_planner(rng):
    n = 6
    positions = rng.standard_normal((n, 3)) * 2.0
    velocities = rng.standard_normal((n, 3)) * 0.4
    v_pref = rng.standard_normal((n, 3))
    v_pref *= np.minimum(1.0, V_MAX / np.linalg.norm(v_pref, axis=1))[:, None]
    planner = orca3d.ORCA3DPlanner(CFG)
    base = _plan_configuration(planner, positions, velocities, v_pref)
    shifted = _plan_configuration(
        planner, positions + np.array([37.0, -12.0, 5.5]), velocities, v_pref
    )
    # ORCA depends on relative positions only; translation changes the float64 rounding
    # of p_j - p_i, hence an exactness tolerance rather than bit equality.
    assert np.allclose(base.velocities, shifted.velocities, atol=1e-12, rtol=0.0)


def test_neighbour_permutation_gives_identical_planes_and_velocities(rng):
    n = 5
    positions = rng.standard_normal((n, 3)) * 2.0
    velocities = rng.standard_normal((n, 3)) * 0.3
    i = 0
    ids = np.array([1, 2, 3, 4])
    frozen = np.array([False, True, False, False])
    nbr_vel = np.where(frozen[:, None], 0.0, velocities[ids])
    base = build_orca_planes(
        positions[i], velocities[i], positions[ids], nbr_vel, frozen,
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=i, neighbor_ids=ids,
    )
    expected_order = np.argsort(ids, kind="stable")
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(ids.size)
        shuffled = build_orca_planes(
            positions[i], velocities[i], positions[ids][perm], nbr_vel[perm],
            frozen[perm],
            d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=i,
            neighbor_ids=ids[perm],
        )
        # production planner sorts by stable agent ID before building planes
        order = np.argsort(ids[perm], kind="stable")
        assert np.array_equal(shuffled.normal[order], base.normal)
        assert np.array_equal(shuffled.point[order], base.point)
        assert np.array_equal(shuffled.alpha[order], base.alpha)
        assert np.array_equal(shuffled.degenerate[order], base.degenerate)
    assert expected_order.tolist() == [0, 1, 2, 3]


# ======================================================================================
# (e) overlap recovery and frozen alpha numeric checks
# ======================================================================================
def test_overlap_recovery_separating_velocity_numeric():
    """Official invTimeStep overlap branch: pair inside combined radius separates.

    For two active agents the required separating speed along the line of centres is
    ``0.5 * inv_dt * (d_plan - ||p_ij||)`` per agent; with ``inv_dt = 10`` and
    ``v_max = 2`` separations below ``0.6 * d_plan`` are infeasible (LP4 fallback) and
    larger separations are feasible.
    """
    rng = np.random.default_rng(99)
    for separation_fraction, expect_fallback in ((0.05, True), (0.2, True),
                                                 (0.4, True), (0.55, True),
                                                 (0.7, False), (0.85, False),
                                                 (0.99, False)):
        for _ in range(3):
            direction = _random_unit(rng)
            separation = separation_fraction * D_PLAN
            positions = np.array([np.zeros(3), separation * direction])
            velocities = np.zeros((2, 3))
            # push the pair into each other
            v_pref = np.array([V_MAX * direction, -V_MAX * direction])
            base = _plan_configuration(
                orca3d.ORCA3DPlanner(CFG), positions, velocities, v_pref
            )
            v = base.velocities
            assert np.all(np.isfinite(v))
            assert np.linalg.norm(v[0]) <= V_MAX + 1e-8
            assert np.linalg.norm(v[1]) <= V_MAX + 1e-8
            # initial overlap pair counted for both sides
            assert base.diagnostics.initial_overlap_pairs.tolist() == [1, 1]
            assert base.diagnostics.used_fallback.tolist() == [
                expect_fallback, expect_fallback
            ]
            # separating relative motion along the line of centres
            separation_rate = float((v[1] - v[0]) @ direction)
            assert separation_rate > 0.0, (separation_fraction, separation_rate, v)


def test_overlap_planes_match_official_recovery_formula():
    """Explicit expected overlap plane: w = -invTimeStep * p_ij, point = v_i + alpha*u."""
    inv_dt = 1.0 / DT
    combined_radius = D_PLAN
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    planes = build_orca_planes(
        positions[0], velocities[0], positions[1:], velocities[1:],
        np.array([False]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    rel_pos = positions[1] - positions[0]
    w = -inv_dt * rel_pos
    w_len = np.linalg.norm(w)
    unit_w = w / w_len
    expected_u = (combined_radius * inv_dt - w_len) * unit_w
    assert np.allclose(planes.normal[0], unit_w, atol=1e-15, rtol=0.0)
    assert np.allclose(
        planes.point[0], velocities[0] + 0.5 * expected_u, atol=1e-15, rtol=0.0
    )
    # alpha=0.5 for active-active, 1.0 when the neighbour is frozen (same geometry)
    frozen_planes = build_orca_planes(
        positions[0], velocities[0], positions[1:], velocities[1:],
        np.array([True]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    assert np.allclose(frozen_planes.normal[0], unit_w, atol=1e-15, rtol=0.0)
    assert np.allclose(
        frozen_planes.point[0] - velocities[0],
        1.0 * expected_u,
        atol=1e-15,
        rtol=0.0,
    )
    assert np.allclose(
        frozen_planes.point[0] - planes.point[0], 0.5 * expected_u, atol=1e-15, rtol=0.0
    )


def test_frozen_alpha_requires_larger_correction_inside_collision_cone():
    """Dangerous pair (inside the collision cone): n·u > 0 and alpha matters linearly."""
    own_pos = np.zeros(3)
    own_vel = np.array([1.0, 0.0, 0.0])
    nbr_pos = np.array([[3.0, 0.5, 0.0]])
    nbr_vel = np.zeros((1, 3))
    active = build_orca_planes(
        own_pos, own_vel, nbr_pos, nbr_vel, np.array([False]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    frozen = build_orca_planes(
        own_pos, own_vel, nbr_pos, nbr_vel, np.array([True]),
        d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=0, neighbor_ids=np.array([1]),
    )
    n = active.normal[0]
    u = 2.0 * (active.point[0] - own_vel)
    n_dot_u = float(n @ u)
    assert n_dot_u > 0.0  # inside the collision cone: correction is a real violation
    assert frozen.point[0] - active.point[0] == pytest.approx(0.5 * u, abs=1e-15)
    pref = np.array([2.0, 0.0, 0.0])
    sol_a = solve_orca_velocity(active.normal, active.point, v_max=V_MAX, v_pref=pref)
    sol_f = solve_orca_velocity(frozen.normal, frozen.point, v_max=V_MAX, v_pref=pref)
    required_a = float(n @ active.point[0])
    required_f = float(n @ frozen.point[0])
    assert required_f - required_a == pytest.approx(0.5 * n_dot_u, abs=1e-12)
    assert float(n @ sol_a.velocity) == pytest.approx(required_a, abs=1e-12)
    assert float(n @ sol_f.velocity) == pytest.approx(required_f, abs=1e-12)
    # frozen case moves the active agent further from its preferred velocity
    assert np.linalg.norm(sol_f.velocity - pref) > np.linalg.norm(sol_a.velocity - pref)


# ======================================================================================
# diagnostics consistency
# ======================================================================================
def test_constraint_violation_diagnostic_matches_independent_recomputation(rng):
    n = 8
    positions = rng.uniform(-2.0, 2.0, size=(n, 3))
    velocities = rng.standard_normal((n, 3)) * 0.5
    v_pref = rng.standard_normal((n, 3))
    v_pref *= np.minimum(1.0, V_MAX / np.linalg.norm(v_pref, axis=1))[:, None]
    result = _plan_configuration(orca3d.ORCA3DPlanner(CFG), positions, velocities, v_pref)
    for a in range(n):
        ids = np.sort(np.array([j for j in range(n) if j != a], dtype=np.int64))
        planes = build_orca_planes(
            positions[a], velocities[a], positions[ids], velocities[ids],
            np.zeros(ids.size, dtype=bool),
            d_plan=D_PLAN, time_horizon=TAU, dt=DT, agent_id=a, neighbor_ids=ids,
        )
        residual = np.maximum(
            0.0,
            np.einsum("ij,ij->i", planes.normal, planes.point - result.velocities[a]),
        )
        expected = float(np.max(residual)) if residual.size else 0.0
        assert result.diagnostics.constraint_violation_max[a] == pytest.approx(
            expected, abs=1e-15
        )
        assert result.diagnostics.candidate_count_tested[a] == ids.size


# ======================================================================================
# performance benchmark (python backend), skipped unless --bench
# ======================================================================================
@pytest.mark.bench
def test_bench_python_backend_100_agents_99_neighbours_1200_steps():
    """100 agents x 99 planes x 1200 steps on one thread must stay below 180 s."""
    import time

    from uavbench import geometry
    from uavbench.navigation import STAGE_TO_GOAL, desired_velocities
    from uavbench.planners.base import PlanState

    n_agents = 100
    n_steps = 1200
    directions = geometry.fibonacci_sphere_directions(n_agents)
    positions = directions * float(CFG.simulation.init_sphere_radius)
    goals = -positions
    velocities = np.zeros((n_agents, 3))
    active_ids = np.arange(n_agents, dtype=np.int64)
    frozen = np.zeros(n_agents, dtype=bool)
    stages = np.full(n_agents, STAGE_TO_GOAL, dtype=np.int8)
    width = n_agents - 1
    neighbor_ids = np.empty((n_agents, width), dtype=np.int64)
    for i in range(n_agents):
        neighbor_ids[i] = np.concatenate(
            [np.arange(i, dtype=np.int64), np.arange(i + 1, n_agents, dtype=np.int64)]
        )
    neighbor_mask = np.ones((n_agents, width), dtype=bool)

    planner = orca3d.ORCA3DPlanner(CFG)
    assert planner.backend == "python"
    fallback_steps = 0
    start = time.perf_counter()
    for k in range(n_steps):
        v_pref = desired_velocities(goals, positions, V_MAX, DT)
        state = PlanState(
            t=k * DT,
            dt=DT,
            positions=positions,
            velocities=velocities,
            stages=stages,
            frozen=frozen,
            active_ids=active_ids,
            waypoints=goals,
            v_pref=v_pref,
            neighbor_ids=neighbor_ids,
            neighbor_mask=neighbor_mask,
        )
        result = planner.plan(state)
        fallback_steps += int(np.count_nonzero(result.diagnostics.used_fallback))
        velocities = result.velocities
        positions = positions + velocities * DT
    elapsed = time.perf_counter() - start
    print(
        f"\nORCA python backend benchmark: {n_steps} steps in {elapsed:.2f} s "
        f"({1e3 * elapsed / n_steps:.2f} ms/step, {fallback_steps} fallback agent-steps)"
    )
    assert np.all(np.isfinite(velocities))
    assert elapsed < 180.0, (
        f"python backend too slow: {elapsed:.2f} s > 180 s for "
        f"{n_agents} agents x {width} planes x {n_steps} steps"
    )
