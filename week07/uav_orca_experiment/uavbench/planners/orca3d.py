"""3D-ORCA planner: faithful float64 NumPy port of RVO2-3D (Apache-2.0).

Upstream
--------
* Repository: https://github.com/snape/RVO2-3D
* Commit:     f940bfba8bac9532ecae948f11639085e8cbf277 (``refs/heads/main``)
* Files:      ``src/Agent.cc`` (``Agent::computeNewVelocity``, ``linearProgram1`` …
              ``linearProgram4`` and the ``Line`` helper), ``src/Agent.h``,
              ``src/Vector3.h``, ``src/Plane.h`` semantics.
* License:    Apache-2.0 (see ``THIRD_PARTY_NOTICES.md`` in the project root).

This module is a port, not a wrapper: the algorithmic structure, branch conditions and
comparisons of the official 3-D linear-program solver are kept, with every quantity in
float64 NumPy instead of the upstream ``float``.  Deviations are listed at the end of
this docstring and in ``THIRD_PARTY_NOTICES.md``.

ORCA plane construction (CONTRACTS.md §8, handoff §5.4)
------------------------------------------------------
For every active agent ``i`` and neighbour ``j`` with
``p_ij = p_j - p_i``, ``combinedRadius = d_plan``, ``timeHorizon = tau``::

    active-active : relVel = v_i - v_j, alpha = 0.5
    active-frozen : relVel = v_i - 0,   alpha = 1.0   (no reciprocal plane for j)
    frozen-frozen : never planned

``u`` and the unit normal ``n`` follow the official cut-off-circle / cone / overlap
branches; the half-space is::

    plane.point  = v_i + alpha * u
    plane.normal = n                      (unit, pointing to the allowed side)
    constraint   : n · (v - plane.point) >= 0

The velocity is the solution of ``min ||v - v_pref||^2`` over ``||v|| <= v_max`` and all
half-spaces, using the official ``linearProgram3`` (feasible case) and the official
``linearProgram4`` min-max-violation relaxation when ``linearProgram3`` reports a failed
plane.  ``used_fallback`` is true iff ``linearProgram4`` is entered, exactly as
``planeFail < orcaPlanes_.size()`` in upstream ``computeNewVelocity``.

Diagnostics (CONTRACTS.md §6)
-----------------------------
* ``used_fallback``            official ``linearProgram4`` relaxation used.
* ``solver_infeasible``        set only by a cheap *sufficient* infeasibility
                               certificate (see :func:`_infeasibility_certificate`),
                               never by solver non-convergence.
* ``constraint_violation_max`` ``max_i max(0, n_i·point_i - n_i·v)`` of the returned
                               velocity.
* ``candidate_count_tested``   number of ORCA planes = neighbour count.
* ``initial_overlap_pairs``    neighbours with ``||p_ij|| < d_plan``.
* ``numerical_error``          a non-finite intermediate had to be replaced by the
                               deterministic recovery convention.
* ``extra`` keys               ``n_planes``, ``n_frozen_neighbors``,
                               ``degenerate_planes`` (deterministic recovery used).

Deviations from upstream (also recorded in THIRD_PARTY_NOTICES.md)
-----------------------------------------------------------------
1. **float64 instead of float32.**  Upstream compiles with ``float``; RVO3D_EPSILON is a
   *squared-magnitude* comparison tolerance (``denominator^2``, ``|cross|^2``,
   ``|planeOptVelocity|^2``).  We keep the literal value ``1e-5`` to preserve the
   official branch structure; this is far above float64 machine epsilon but the numeric
   tests verify that it does not change generic feasible-instant optimum (objective
   agrees with an independent SLSQP reference to 1e-6).  Tightening it was not needed;
   any future tightening is a documented parameter change only.
2. **Frozen-sphere adaptation** (experiment protocol, not upstream): full collision
   responsibility for the active agent (``alpha = 1.0``, frozen neighbour velocity 0,
   no reciprocal constraint emitted).  Frozen agents are never planned.
3. **Neighbour order** is stable ascending agent ID instead of the upstream KD-tree
   order, so the result is independent of neighbour enumeration order (synchrony test).
4. **Degenerate recovery**: exactly coincident agents with zero relative velocity, and
   the measure-zero radial cone cases where upstream divides by a zero vector, get a
   deterministic unit direction that is antisymmetric in the ``(i, j)`` pair: reversing
   the pair negates the direction.  The official formulas are attempted first; the
   recovery convention only replaces non-finite / division-by-zero intermediates.
   This is a recovery convention, it does not restore lost safety guarantees.
5. **Violation-ordered exact pre-pass (performance only).**  ``linearProgram3`` is an
   incremental projection; when every sub-problem succeeds its final point is the exact
   projection of ``v_pref`` and is mathematically independent of the plane enumeration
   order.  ``solve_orca_velocity`` therefore first runs the same official
   ``linearProgram3`` on planes ordered by decreasing initial violation for systems with
   more than two violated planes (stable tie-break = official plane order); on success it
   returns that exact optimum.  On failure it runs the unchanged official ID-order
   ``linearProgram3`` → ``linearProgram4`` path, so infeasible instances keep the
   documented official LP4 relaxation and its plane order.  Semantics are unchanged, only
   the number of LP2/LP1 sub-problems is reduced; validated by the differential tests.
6. **Numba backend not shipped.**  ``planners.backend`` accepts ``auto|python|numba``;
   in this environment numba 0.57.0 cannot import against numpy 1.26.4 ("Numba needs
   NumPy 1.24 or less"), no installation is allowed, and a jitted LP cannot be
   parity-validated.  ``auto`` therefore resolves to ``"python"`` deterministically and
   an explicit ``"numba"`` request raises :class:`RuntimeError` instead of silently
   changing the backend.  ``active_backend()`` reports the actual backend for
   provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import geometry
from ..config import Config
from .base import Diagnostics, PlanResult, PlanState

__all__ = [
    "ALPHA_ACTIVE_ACTIVE",
    "ALPHA_ACTIVE_FROZEN",
    "ORCA3DPlanner",
    "OrcaPlanes",
    "OrcaSolution",
    "RVO3D_EPSILON",
    "active_backend",
    "build_orca_planes",
    "numba_available",
    "numba_import_error",
    "resolve_backend",
    "solve_orca_velocity",
]

#: Upstream ``RVO3D_EPSILON``: a small positive number used as threshold on *squared*
#: magnitudes (``denominator * denominator``, ``absSq(cross)``, ``absSq(velocity)``).
RVO3D_EPSILON: float = 1.0e-5

#: Fraction of the ORCA correction assigned to the active agent for active-active pairs
#: (upstream ``0.5F * u``).
ALPHA_ACTIVE_ACTIVE: float = 0.5
#: Full correction for active-frozen pairs (experiment adaptation, handoff §5.4).
ALPHA_ACTIVE_FROZEN: float = 1.0

#: Norm below which a direction/relative velocity is treated as a zero vector.  Pure
#: floating-point degeneracy guard, not an algorithmic tolerance: the corresponding
#: instances send the *official* code into a 0/0 division.
DEGENERATE_NORM: float = 1e-12

#: Upper bound on the noise-level projection of an LP result back onto the max-speed
#: sphere (m/s).  Only tolerance-level corrections are allowed (handoff §2.2).
_SPEED_CLAMP_FLAG: float = 1.0e-6

_VEL_EPS: float = geometry.VEL_EPS

_MASK64: int = (1 << 64) - 1
_REF_AXIS_1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_REF_AXIS_2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)


# --------------------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------------------
_NUMBA_PROBE_DONE = False
_NUMBA_IMPORT_ERROR: str | None = None
_ACTIVE_BACKEND: str = "python"


def _probe_numba() -> None:
    """Detect whether numba can be imported at all (cached).  Only used for reporting."""
    global _NUMBA_PROBE_DONE, _NUMBA_IMPORT_ERROR
    if _NUMBA_PROBE_DONE:
        return
    _NUMBA_PROBE_DONE = True
    try:  # pragma: no cover - depends on the environment
        import numba  # noqa: F401

        _NUMBA_IMPORT_ERROR = (
            "numba imported but no parity-validated numba ORCA LP backend is shipped "
            "in this port; python is the only validated production backend"
        )
    except Exception as exc:  # noqa: BLE001 - report the concrete import failure
        _NUMBA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def numba_import_error() -> str | None:
    """Reason why numba cannot be used (``None`` only if a validated jit backend existed)."""
    _probe_numba()
    return _NUMBA_IMPORT_ERROR


def numba_available() -> bool:
    """False: no parity-validated numba ORCA LP backend is shipped in this port."""
    return False


def resolve_backend(requested: str) -> str:
    """Resolve ``planners.backend`` (``auto|python|numba``) to the actual backend.

    ``auto`` deterministically resolves to ``"python"``.  ``"numba"`` raises instead of
    silently changing semantics because the numba backend is unavailable/unvalidated.
    """
    if requested == "python":
        return "python"
    if requested == "auto":
        return "numba" if numba_available() else "python"
    if requested == "numba":
        _probe_numba()
        raise RuntimeError(
            "planners.backend='numba' requested but no validated numba backend is "
            f"available ({numba_import_error()}); use 'python' or 'auto'"
        )
    raise ValueError(f"unknown planner backend: {requested!r}")


def active_backend() -> str:
    """Actual backend of the most recently constructed ORCA planner (provenance)."""
    return _ACTIVE_BACKEND


def _set_active_backend(backend: str) -> None:
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


# --------------------------------------------------------------------------------------
# Deterministic degenerate directions
# --------------------------------------------------------------------------------------
def _splitmix64(x: int) -> int:
    """Deterministic 64-bit integer mixer (SplitMix64 finalizer)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _pair_id_direction(agent_id: int, neighbor_id: int) -> np.ndarray:
    """Deterministic unit direction for the unordered pair, antisymmetric in ``(i, j)``.

    ``d(j, i) == -d(i, j)`` holds exactly (integer mixing plus a final sign flip), so the
    two members of a coincident pair receive complementary constraints and the whole
    planner remains deterministic and order independent.
    """
    i = int(agent_id)
    j = int(neighbor_id)
    if i == j:
        raise ValueError("degenerate pair direction requires distinct agent ids")
    flipped = i > j
    a, b = (i, j) if i < j else (j, i)
    h = _splitmix64(((a & 0xFFFFFFFF) << 32) ^ (b & 0xFFFFFFFF))
    h1 = _splitmix64(h)
    h2 = _splitmix64(h1)
    z = 1.0 - 2.0 * ((h & 0xFFFFFFFF) / 4294967296.0)
    phi = 2.0 * math.pi * ((h1 & 0xFFFFFFFF) / 4294967296.0)
    r = math.sqrt(max(0.0, 1.0 - z * z))
    direction = np.array([r * math.cos(phi), r * math.sin(phi), z], dtype=np.float64)
    if h2 == 0:  # pragma: no cover - unreachable, keeps the contract obvious
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.sqrt(direction @ direction))
    if norm <= DEGENERATE_NORM:  # pragma: no cover - unreachable
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    return -direction if flipped else direction


def _degenerate_directions(
    rel_pos: np.ndarray, agent_id: int, neighbor_ids: np.ndarray
) -> np.ndarray:
    """Unit directions orthogonal to ``rel_pos`` (or ID based if ``rel_pos`` is zero).

    ``cross(rel_pos, ref)`` changes sign when the pair is reversed (``rel_pos`` negates),
    so the returned direction is antisymmetric up to a fixed reference axis choice; the
    fully-coincident case falls back to :func:`_pair_id_direction`.
    """
    rel_pos = np.asarray(rel_pos, dtype=np.float64).reshape(-1, 3)
    k = rel_pos.shape[0]
    out = np.zeros((k, 3), dtype=np.float64)
    if k == 0:
        return out
    c1 = np.cross(rel_pos, _REF_AXIS_1)
    n1 = np.sqrt(np.einsum("ij,ij->i", c1, c1))
    good = n1 > DEGENERATE_NORM
    out[good] = c1[good] / n1[good, None]
    rest = np.flatnonzero(~good)
    if rest.size:
        c2 = np.cross(rel_pos[rest], _REF_AXIS_2)
        n2 = np.sqrt(np.einsum("ij,ij->i", c2, c2))
        good2 = n2 > DEGENERATE_NORM
        idx2 = rest[good2]
        out[idx2] = c2[good2] / n2[good2, None]
        for kk in rest[~good2]:
            out[kk] = _pair_id_direction(agent_id, int(neighbor_ids[kk]))
    # Last-resort normalisation (guards against underflow in exotic inputs).
    lens = np.sqrt(np.einsum("ij,ij->i", out, out))
    bad = lens <= DEGENERATE_NORM
    if np.any(bad):  # pragma: no cover - unreachable for finite inputs
        out[bad] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        lens = np.sqrt(np.einsum("ij,ij->i", out, out))
    return out / lens[:, None]


# --------------------------------------------------------------------------------------
# ORCA plane construction
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OrcaPlanes:
    """ORCA half-spaces for one active agent (CONTRACTS.md §8)."""

    normal: np.ndarray  # (M,3) float64, unit normals
    point: np.ndarray  # (M,3) float64, plane.point = v_i + alpha * u
    alpha: np.ndarray  # (M,) float64, 0.5 active-active / 1.0 active-frozen
    degenerate: np.ndarray  # (M,) bool, deterministic recovery convention used
    numerical_error: bool  # non-finite intermediate was replaced
    n_overlap: int  # neighbours with ||p_ij|| <= d_plan


def _unit_rows(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Row-wise unit vectors; rows with ``valid=False`` are left as zero."""
    out = np.zeros_like(x)
    lengths = np.sqrt(np.einsum("ij,ij->i", x, x))
    np.divide(x, lengths[:, None], out=out, where=valid[:, None])
    return out


def build_orca_planes(
    own_position,
    own_velocity,
    neighbor_positions,
    neighbor_velocities,
    neighbor_frozen,
    *,
    d_plan: float,
    time_horizon: float,
    dt: float,
    agent_id: int = 0,
    neighbor_ids=None,
) -> OrcaPlanes:
    """Build the ORCA planes for one active agent.

    ``neighbor_positions``/``neighbor_velocities`` are ``(M,3)`` in the (stable ID
    sorted) neighbour order; ``neighbor_frozen`` is ``(M,)`` bool.  Frozen neighbours are
    treated as zero-velocity bodies with ``alpha = 1.0``.

    The implementation is vectorised over neighbours but follows ``computeNewVelocity``
    branch by branch:
    ``distSq > combinedRadiusSq`` → one of the official sub-branches
    (``distSq <= combinedRadiusSq`` → the official ``invTimeStep`` overlap recovery).
    """
    p_i = np.asarray(own_position, dtype=np.float64).reshape(3)
    v_i = np.asarray(own_velocity, dtype=np.float64).reshape(3)
    p_j = np.asarray(neighbor_positions, dtype=np.float64).reshape(-1, 3)
    v_j = np.asarray(neighbor_velocities, dtype=np.float64).reshape(-1, 3)
    frozen = np.asarray(neighbor_frozen, dtype=bool).reshape(-1)
    m = p_j.shape[0]
    if neighbor_ids is None:
        ids = np.arange(m, dtype=np.int64)
    else:
        ids = np.asarray(neighbor_ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] != m:
            raise ValueError("neighbor_ids must have the same length as neighbor_positions")

    normal = np.zeros((m, 3), dtype=np.float64)
    u = np.zeros((m, 3), dtype=np.float64)
    degenerate = np.zeros(m, dtype=bool)
    alpha = np.where(frozen, ALPHA_ACTIVE_FROZEN, ALPHA_ACTIVE_ACTIVE)
    if m == 0:
        return OrcaPlanes(
            normal=normal,
            point=np.zeros((0, 3), dtype=np.float64),
            alpha=alpha,
            degenerate=degenerate,
            numerical_error=False,
            n_overlap=0,
        )

    combined_radius = float(d_plan)
    combined_radius_sq = combined_radius * combined_radius
    inv_time_horizon = 1.0 / float(time_horizon)
    inv_time_step = 1.0 / float(dt)

    rel_pos = p_j - p_i[None, :]
    frozen_v = np.where(frozen[:, None], 0.0, v_j)
    rel_vel = v_i[None, :] - frozen_v
    dist_sq = np.einsum("ij,ij->i", rel_pos, rel_pos)
    overlap = dist_sq <= combined_radius_sq
    numerical_error = False

    # ---- distSq > combinedRadiusSq: official no-collision branches ---------------------
    non_overlap = np.flatnonzero(~overlap)
    if non_overlap.size:
        rp = rel_pos[non_overlap]
        rv = rel_vel[non_overlap]
        w = rv - inv_time_horizon * rp
        w_len_sq = np.einsum("ij,ij->i", w, w)
        dot_product = np.einsum("ij,ij->i", w, rp)
        # Official condition: dotProduct < 0 and dotProduct^2 > combinedRadiusSq*wLenSq.
        cutoff = (dot_product < 0.0) & (
            dot_product * dot_product > combined_radius_sq * w_len_sq
        )
        cone = ~cutoff

        if np.any(cutoff):
            sub = non_overlap[cutoff]
            wc = w[cutoff]
            w_len = np.sqrt(w_len_sq[cutoff])
            bad = w_len <= DEGENERATE_NORM
            unit_w = _unit_rows(wc, ~bad)
            if not np.all(np.isfinite(unit_w)):
                numerical_error = True
            normal[sub] = unit_w
            u[sub] = (combined_radius * inv_time_horizon - w_len)[:, None] * unit_w
            if np.any(bad):
                bad_sub = sub[bad]
                lat = _degenerate_directions(rel_pos[bad_sub], agent_id, ids[bad_sub])
                normal[bad_sub] = lat
                u[bad_sub] = (combined_radius * inv_time_horizon) * lat
                degenerate[bad_sub] = True

        if np.any(cone):
            sub = non_overlap[cone]
            rpc = rp[cone]
            rvc = rv[cone]
            a = dist_sq[sub]
            b = np.einsum("ij,ij->i", rpc, rvc)
            cross_p = np.cross(rpc, rvc)
            cross_sq = np.einsum("ij,ij->i", cross_p, cross_p)
            denom = a - combined_radius_sq
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                c = np.einsum("ij,ij->i", rvc, rvc) - cross_sq / denom
                disc = b * b - a * c
            negative_disc = ~(disc >= 0.0)
            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                t = (b + sqrt_disc) / a
            finite_t = np.isfinite(t)
            if not (np.all(finite_t) and not np.any(negative_disc)):
                numerical_error = True
            t_safe = np.where(finite_t, t, 0.0)
            ww = rvc - t_safe[:, None] * rpc
            ww_len = np.sqrt(np.einsum("ij,ij->i", ww, ww))
            bad = negative_disc | ~finite_t | (ww_len <= DEGENERATE_NORM)
            unit_ww = _unit_rows(ww, ~bad)
            normal[sub] = unit_ww
            u[sub] = (combined_radius * t_safe - ww_len)[:, None] * unit_ww
            if np.any(bad):
                bad_sub = sub[bad]
                lat = _degenerate_directions(rel_pos[bad_sub], agent_id, ids[bad_sub])
                t_bad = np.maximum(t_safe[bad], 0.0)
                normal[bad_sub] = lat
                u[bad_sub] = (combined_radius * t_bad)[:, None] * lat
                degenerate[bad_sub] = True

    # ---- distSq <= combinedRadiusSq: official invTimeStep overlap recovery -------------
    overlap_idx = np.flatnonzero(overlap)
    if overlap_idx.size:
        rp = rel_pos[overlap_idx]
        rv = rel_vel[overlap_idx]
        w = rv - inv_time_step * rp
        w_len = np.sqrt(np.einsum("ij,ij->i", w, w))
        bad = w_len <= DEGENERATE_NORM
        unit_w = _unit_rows(w, ~bad)
        normal[overlap_idx] = unit_w
        u[overlap_idx] = (combined_radius * inv_time_step - w_len)[:, None] * unit_w
        if np.any(bad):
            bad_sub = overlap_idx[bad]
            lat = _degenerate_directions(rel_pos[bad_sub], agent_id, ids[bad_sub])
            normal[bad_sub] = lat
            u[bad_sub] = (combined_radius * inv_time_step) * lat
            degenerate[bad_sub] = True

    point = v_i[None, :] + alpha[:, None] * u

    # Final safety net: nothing non-finite may leave this function (CONTRACTS.md §8).
    bad_rows = ~np.isfinite(normal).all(axis=1) | ~np.isfinite(point).all(axis=1)
    if np.any(bad_rows):
        numerical_error = True
        idx = np.flatnonzero(bad_rows)
        lat = _degenerate_directions(rel_pos[idx], agent_id, ids[idx])
        normal[idx] = lat
        u[idx] = (combined_radius * inv_time_step) * lat
        point[idx] = v_i[None, :] + alpha[idx, None] * u[idx]

    return OrcaPlanes(
        normal=normal,
        point=point,
        alpha=alpha,
        degenerate=degenerate,
        numerical_error=numerical_error,
        # CONTRACTS.md §6/§7: initial_overlap_pairs counts ||p_ij|| *<* d_plan; the
        # official plane branch itself uses distSq <= combinedRadiusSq (measure zero).
        n_overlap=int(np.count_nonzero(dist_sq < combined_radius_sq)),
    )


# --------------------------------------------------------------------------------------
# Official linear-program solver (linearProgram1 … linearProgram4), float64 NumPy port
# --------------------------------------------------------------------------------------
def _project_to_ball(v: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.sqrt(v @ v))
    if norm <= radius:
        return np.array(v, dtype=np.float64, copy=True)
    return v * (radius / norm)


def _linear_program1(
    normal: np.ndarray,
    point: np.ndarray,
    plane_no: int,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    radius: float,
    opt_velocity: np.ndarray,
    direction_opt: bool,
):
    """Official ``linearProgram1``: 1-D LP on ``line`` inside the max-speed sphere."""
    dot_product = float(line_point @ line_direction)
    discriminant = (
        dot_product * dot_product + radius * radius - float(line_point @ line_point)
    )
    if discriminant < 0.0:
        return False, None
    sqrt_discriminant = math.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    if plane_no > 0:
        planes_n = normal[:plane_no]
        planes_p = point[:plane_no]
        numerator = np.einsum("ij,ij->i", planes_p - line_point, planes_n)
        denominator = planes_n @ line_direction
        parallel = denominator * denominator <= RVO3D_EPSILON
        if np.any(parallel & (numerator > 0.0)):
            return False, None
        active = ~parallel
        if np.any(active):
            t = numerator[active] / denominator[active]
            left = denominator[active] >= 0.0
            if np.any(left):
                t_left = max(t_left, float(np.max(t[left])))
            if np.any(~left):
                t_right = min(t_right, float(np.min(t[~left])))
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


def _linear_program2(
    normal: np.ndarray,
    point: np.ndarray,
    plane_no: int,
    radius: float,
    opt_velocity: np.ndarray,
    direction_opt: bool,
):
    """Official ``linearProgram2``: 2-D LP on plane ``plane_no`` inside the sphere."""
    n_i = normal[plane_no]
    p_i = point[plane_no]
    plane_dist = float(p_i @ n_i)
    radius_sq = radius * radius
    if plane_dist * plane_dist > radius_sq:
        return False, None
    plane_radius_sq = radius_sq - plane_dist * plane_dist
    plane_center = plane_dist * n_i

    if direction_opt:
        plane_opt_velocity = opt_velocity - float(opt_velocity @ n_i) * n_i
        plane_opt_length_sq = float(plane_opt_velocity @ plane_opt_velocity)
        if plane_opt_length_sq <= RVO3D_EPSILON:
            result = np.array(plane_center, dtype=np.float64, copy=True)
        else:
            result = plane_center + math.sqrt(
                plane_radius_sq / plane_opt_length_sq
            ) * plane_opt_velocity
    else:
        result = opt_velocity + float((p_i - opt_velocity) @ n_i) * n_i
        if float(result @ result) > radius_sq:
            plane_result = result - plane_center
            plane_result_sq = float(plane_result @ plane_result)
            if plane_result_sq <= 0.0:  # pragma: no cover - only with NaN/underflow
                return False, None
            result = plane_center + math.sqrt(
                plane_radius_sq / plane_result_sq
            ) * plane_result

    j = 0
    while j < plane_no:
        violations = np.einsum(
            "ij,ij->i", point[j:plane_no] - result, normal[j:plane_no]
        )
        violated = np.flatnonzero(violations > 0.0)
        if violated.size == 0:
            return True, result
        k = j + int(violated[0])
        cross_product = np.cross(normal[k], n_i)
        cross_sq = float(cross_product @ cross_product)
        if cross_sq <= RVO3D_EPSILON:
            return False, None
        line_direction = cross_product / math.sqrt(cross_sq)
        line_normal = np.cross(line_direction, n_i)
        denominator = float(line_normal @ normal[k])
        if denominator == 0.0:  # pragma: no cover - degenerate numeric guard
            return False, None
        line_point = p_i + (
            float((point[k] - p_i) @ normal[k]) / denominator
        ) * line_normal
        if not np.all(np.isfinite(line_point)):  # pragma: no cover - safety guard
            return False, None
        ok, result = _linear_program1(
            normal,
            point,
            k,
            line_point,
            line_direction,
            radius,
            opt_velocity,
            direction_opt,
        )
        if not ok:
            return False, None
        j = k + 1
    return True, result


def _linear_program3(
    normal: np.ndarray,
    point: np.ndarray,
    radius: float,
    opt_velocity: np.ndarray,
    direction_opt: bool,
):
    """Official ``linearProgram3``.  Returns ``(fail_index, result)``."""
    m = normal.shape[0]
    opt_norm_sq = float(opt_velocity @ opt_velocity)
    if direction_opt:
        result = opt_velocity * radius
    elif opt_norm_sq > radius * radius:
        result = opt_velocity * (radius / math.sqrt(opt_norm_sq))
    else:
        result = np.array(opt_velocity, dtype=np.float64, copy=True)

    i = 0
    while i < m:
        violations = np.einsum("ij,ij->i", point[i:] - result, normal[i:])
        violated = np.flatnonzero(violations > 0.0)
        if violated.size == 0:
            return m, result
        i = i + int(violated[0])
        temp_result = result
        ok, result = _linear_program2(normal, point, i, radius, opt_velocity, direction_opt)
        if not ok:
            return i, temp_result
        i += 1
    return m, result


def _linear_program4(
    normal: np.ndarray,
    point: np.ndarray,
    begin_plane: int,
    radius: float,
    result: np.ndarray,
) -> np.ndarray:
    """Official ``linearProgram4``: min-max-violation relaxation (infeasible fallback)."""
    m = normal.shape[0]
    distance = 0.0
    start = int(begin_plane)
    while start < m:
        violations = np.einsum("ij,ij->i", point[start:] - result, normal[start:])
        records = np.flatnonzero(violations > distance)
        if records.size == 0:
            return result
        i = start + int(records[0])

        if i == 0:
            proj_normal = np.zeros((0, 3), dtype=np.float64)
            proj_point = np.zeros((0, 3), dtype=np.float64)
        else:
            plane_n = normal[:i]
            plane_p = point[:i]
            cross_product = np.cross(plane_n, normal[i])
            cross_sq = np.einsum("ij,ij->i", cross_product, cross_product)
            parallel = cross_sq <= RVO3D_EPSILON
            same_direction = (plane_n @ normal[i]) > 0.0
            keep = (~parallel) | (~same_direction)
            proj_point = np.zeros((i, 3), dtype=np.float64)
            opposite = parallel & (~same_direction)
            if np.any(opposite):
                proj_point[opposite] = 0.5 * (point[i] + plane_p[opposite])
            non_parallel = ~parallel
            if np.any(non_parallel):
                cp = cross_product[non_parallel]
                line_normal = np.cross(cp, normal[i])
                np_den = np.einsum("ij,ij->i", line_normal, plane_n[non_parallel])
                np_num = np.einsum(
                    "ij,ij->i", plane_p[non_parallel] - point[i], plane_n[non_parallel]
                )
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    proj_point[non_parallel] = point[i] + (
                        np_num / np_den
                    )[:, None] * line_normal
            diff = plane_n - normal[i][None, :]
            diff_len = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            proj_normal = _unit_rows(diff, diff_len > 0.0)
            proj_normal = proj_normal[keep]
            proj_point = proj_point[keep]

        temp_result = result
        fail, candidate = _linear_program3(
            proj_normal, proj_point, radius, normal[i], True
        )
        if fail < proj_normal.shape[0]:
            result = temp_result
        else:
            result = candidate
        distance = float((point[i] - result) @ normal[i])
        start = i + 1
    return result


def _infeasibility_certificate(
    normal: np.ndarray, point: np.ndarray, v_max: float
) -> bool:
    """Cheap *sufficient* certificate that the ORCA half-space system is infeasible.

    The max-speed sphere is convex and every ``normal`` is a unit vector, therefore

    * (a) a single plane with ``n·point > v_max + VEL_EPS`` is infeasible, because
      ``n·v <= |v| <= v_max < n·point`` for every admissible ``v``;
    * (b) two antiparallel planes ``n_j = -n_i`` with
      ``n_i·point_i + n_j·point_j > VEL_EPS`` are infeasible, because they require
      ``n·v >= a`` and ``-n·v >= b``, i.e. ``-b <= n·v >= a`` with ``a > -b``.

    No other situation is reported as infeasible: solver non-convergence is not proof
    (CONTRACTS.md §8, handoff §5.4).
    """
    m = normal.shape[0]
    if m == 0:
        return False
    finite = np.isfinite(normal).all(axis=1) & np.isfinite(point).all(axis=1)
    if not np.all(finite):
        return False
    offsets = np.einsum("ij,ij->i", normal, point)
    if np.any(offsets > v_max + _VEL_EPS):
        return True
    if m < 2:
        return False
    gram = normal @ normal.T
    # Antiparallel detection: unit normals with n_i·n_j <= -1 + 1e-12.
    antiparallel = gram <= (-1.0 + 1e-12)
    incompatible = (offsets[:, None] + offsets[None, :]) > _VEL_EPS
    return bool(np.any(antiparallel & incompatible))


# --------------------------------------------------------------------------------------
# Solve one agent's ORCA LP
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OrcaSolution:
    """Result of solving the ORCA LP for one agent."""

    velocity: np.ndarray  # (3,) float64, finite and ||v|| <= v_max (+ tolerance clamp)
    used_fallback: bool  # official linearProgram4 relaxation was entered
    solver_infeasible: bool  # cheap sufficient certificate proved infeasibility
    constraint_violation_max: float  # m/s, max(0, n·point - n·v) over planes
    numerical_error: bool  # a non-finite intermediate had to be replaced


def solve_orca_velocity(
    normal,
    point,
    *,
    v_max: float,
    v_pref,
    velocity_eps: float = _VEL_EPS,
) -> OrcaSolution:
    """Solve ``min ||v - v_pref||²`` s.t. ``||v|| <= v_max`` and all ORCA half-spaces.

    Thin wrapper around the official ``linearProgram3`` → ``linearProgram4`` logic.
    ``normal``/``point`` must be shape ``(M,3)``; ``normal`` rows are unit vectors.

    Performance note (semantics-preserving): ``linearProgram3`` is an incremental
    projection, so *when every sub-problem succeeds* its final point is the exact
    projection of ``v_pref`` onto the feasible set and is independent of the plane
    enumeration order.  For systems with several violated half-spaces we therefore first
    run the same official ``linearProgram3`` with planes ordered by decreasing violation
    at ``v_pref`` (stable tie-break = original order).  If that ordered run succeeds we
    return its (exact) optimum.  If it reports a failed plane we re-run the unchanged
    official ID-order ``linearProgram3`` → ``linearProgram4`` path, so infeasible
    instances get bit-for-bit the documented official behaviour including the LP4
    min-max-violation relaxation order.  See ``THIRD_PARTY_NOTICES.md``.
    """
    normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    point = np.asarray(point, dtype=np.float64).reshape(-1, 3)
    pref = np.asarray(v_pref, dtype=np.float64).reshape(3)
    v_max = float(v_max)
    m = normal.shape[0]
    if normal.shape != point.shape:
        raise ValueError("normal and point must have the same (M,3) shape")

    if m == 0:
        result = _project_to_ball(pref, v_max)
        return OrcaSolution(result, False, False, 0.0, False)

    used_fallback = False
    solver_infeasible = False
    numerical_error = False
    result = None

    # Fast exact path for systems that are (almost certainly) feasible: violation-order
    # pre-pass of the official linearProgram3.  Skipped for cheap systems (<= 2 violated
    # planes) where the ordered setup would cost more than it saves.
    if m > 2:
        offsets = np.einsum("ij,ij->i", normal, point)
        violations_at_pref = offsets - normal @ pref
        if int(np.count_nonzero(violations_at_pref > 0.0)) > 2:
            order = np.argsort(-violations_at_pref, kind="stable")
            fail_ordered, result_ordered = _linear_program3(
                normal[order], point[order], v_max, pref, False
            )
            if fail_ordered == m:
                # The fast path must only return genuinely feasible points: a near-
                # parallel EPS mis-classification inside LP1/LP2 must never turn into a
                # silent constraint violation that the official path would have caught.
                ordered_violation = np.einsum(
                    "ij,ij->i", normal, point - result_ordered[None, :]
                )
                if float(np.max(ordered_violation, initial=-np.inf)) <= 1e-7:
                    result = result_ordered

    if result is None:
        fail_index, result = _linear_program3(normal, point, v_max, pref, False)
        used_fallback = fail_index < m
        if used_fallback:
            result = _linear_program4(normal, point, fail_index, v_max, result)
            solver_infeasible = _infeasibility_certificate(normal, point, v_max)

    result = np.asarray(result, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(result)):
        numerical_error = True
        result = _project_to_ball(pref, v_max)

    speed = float(np.sqrt(result @ result))
    if speed > v_max:
        if speed - v_max > _SPEED_CLAMP_FLAG * max(1.0, v_max):
            numerical_error = True
        result = result * (v_max / speed)

    violation = np.maximum(
        0.0, np.einsum("ij,ij->i", normal, point - result[None, :])
    )
    constraint_violation_max = float(np.max(violation)) if violation.size else 0.0
    return OrcaSolution(
        velocity=result,
        used_fallback=bool(used_fallback),
        solver_infeasible=bool(solver_infeasible),
        constraint_violation_max=constraint_violation_max,
        numerical_error=bool(numerical_error),
    )


# --------------------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------------------
class ORCA3DPlanner:
    """3-D ORCA planner with the experiment's frozen-sphere adaptation.

    Parameters
    ----------
    cfg:
        Frozen project :class:`uavbench.config.Config`; uses
        ``simulation.d_plan``, ``simulation.v_max`` and ``planners.orca.tau``.
    candidate_set:
        Ignored; accepted for constructor compatibility with the shared
        ``build_planner`` interface (ORCA has no candidate sampling).
    """

    name = "orca"

    def __init__(self, cfg: Config, candidate_set=None):  # noqa: ARG002 - interface parity
        self.cfg = cfg
        self.candidate_set = candidate_set
        self.backend = resolve_backend(cfg.planners.backend)
        _set_active_backend(self.backend)
        self.d_plan = float(cfg.simulation.d_plan)
        self.v_max = float(cfg.simulation.v_max)
        self.time_horizon = float(cfg.planners.orca.tau)

    # -- public API ---------------------------------------------------------------------
    def plan(self, state: PlanState) -> PlanResult:
        active_ids = np.asarray(state.active_ids, dtype=np.int64).reshape(-1)
        n_active = int(active_ids.shape[0])
        if n_active == 0:
            empty = np.zeros(0, dtype=np.int32)
            return PlanResult(
                velocities=np.zeros((0, 3), dtype=np.float64),
                diagnostics=Diagnostics(
                    sample_exhausted=np.zeros(0, dtype=bool),
                    used_fallback=np.zeros(0, dtype=bool),
                    solver_infeasible=np.zeros(0, dtype=bool),
                    constraint_violation_max=np.zeros(0, dtype=np.float64),
                    candidate_count_tested=empty.copy(),
                    initial_overlap_pairs=empty.copy(),
                    numerical_error=np.zeros(0, dtype=bool),
                    extra={
                        "n_planes": empty.copy(),
                        "n_frozen_neighbors": empty.copy(),
                        "degenerate_planes": empty.copy(),
                    },
                ),
            )

        positions = np.asarray(state.positions, dtype=np.float64)
        velocities = np.asarray(state.velocities, dtype=np.float64)
        frozen = np.asarray(state.frozen, dtype=bool)
        v_pref = np.asarray(state.v_pref, dtype=np.float64)
        neighbor_ids = np.asarray(state.neighbor_ids, dtype=np.int64)
        neighbor_mask = np.asarray(state.neighbor_mask, dtype=bool)
        dt = float(state.dt)

        out_velocities = np.zeros((n_active, 3), dtype=np.float64)
        used_fallback = np.zeros(n_active, dtype=bool)
        solver_infeasible = np.zeros(n_active, dtype=bool)
        violation_max = np.zeros(n_active, dtype=np.float64)
        candidate_count = np.zeros(n_active, dtype=np.int32)
        overlap_pairs = np.zeros(n_active, dtype=np.int32)
        numerical_error = np.zeros(n_active, dtype=bool)
        n_planes = np.zeros(n_active, dtype=np.int32)
        n_frozen = np.zeros(n_active, dtype=np.int32)
        degenerate_count = np.zeros(n_active, dtype=np.int32)

        for a in range(n_active):
            agent_i = int(active_ids[a])
            mask = neighbor_mask[a]
            ids = neighbor_ids[a][mask]
            if ids.size > 1:
                # Stable ascending ID order: the result must not depend on enumeration
                # order (synchrony contract), and the official KD-tree order is not
                # available/reproducible here.
                order = np.argsort(ids, kind="stable")
                ids = ids[order]
            nbr_pos = positions[ids]
            nbr_frozen = frozen[ids]
            nbr_vel = np.where(nbr_frozen[:, None], 0.0, velocities[ids])
            planes = build_orca_planes(
                positions[agent_i],
                velocities[agent_i],
                nbr_pos,
                nbr_vel,
                nbr_frozen,
                d_plan=self.d_plan,
                time_horizon=self.time_horizon,
                dt=dt,
                agent_id=agent_i,
                neighbor_ids=ids,
            )
            solution = solve_orca_velocity(
                planes.normal,
                planes.point,
                v_max=self.v_max,
                v_pref=v_pref[a],
            )
            out_velocities[a] = solution.velocity
            used_fallback[a] = solution.used_fallback
            solver_infeasible[a] = solution.solver_infeasible
            violation_max[a] = solution.constraint_violation_max
            numerical_error[a] = bool(
                solution.numerical_error or planes.numerical_error
            )
            candidate_count[a] = planes.normal.shape[0]
            overlap_pairs[a] = planes.n_overlap
            n_planes[a] = planes.normal.shape[0]
            n_frozen[a] = int(np.count_nonzero(nbr_frozen))
            degenerate_count[a] = int(np.count_nonzero(planes.degenerate))

        diagnostics = Diagnostics(
            sample_exhausted=np.zeros(n_active, dtype=bool),  # ORCA does not sample
            used_fallback=used_fallback,
            solver_infeasible=solver_infeasible,
            constraint_violation_max=violation_max,
            candidate_count_tested=candidate_count,
            initial_overlap_pairs=overlap_pairs,
            numerical_error=numerical_error,
            extra={
                "n_planes": n_planes,
                "n_frozen_neighbors": n_frozen,
                "degenerate_planes": degenerate_count,
            },
        )
        return PlanResult(velocities=out_velocities, diagnostics=diagnostics)
