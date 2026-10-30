"""共享几何与连续安全原语。

所有函数均为 float64、纯函数、支持 numpy 广播：形状约定为 ``(..., 3)``，
最后一维是三维向量。仅用于仿真器、规划器与测试；指标模块可使用等价向量化实现，
但必须与本模块结果在测试中一致（见 CONTRACTS.md §2）。
"""

from __future__ import annotations

import numpy as np

#: 长度容差（米）
GEOM_EPS: float = 1e-8
#: 速度容差（米/秒）
VEL_EPS: float = 1e-8


def _as_vec3(x: np.ndarray | list | tuple) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected last dimension 3, got shape {arr.shape}")
    return arr


def norm(x: np.ndarray) -> np.ndarray:
    """欧氏范数，``(..., 3) -> (...)``。"""
    return np.linalg.norm(np.asarray(x, dtype=np.float64), axis=-1)


def sq_norm(x: np.ndarray) -> np.ndarray:
    """范数平方，``(..., 3) -> (...)``。"""
    a = np.asarray(x, dtype=np.float64)
    return np.einsum("...i,...i->...", a, a)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.einsum("...i,...i->...", a, b)


def normalize(x: np.ndarray) -> np.ndarray:
    """逐行单位化；零向量原样返回（调用方自行处理退化）。"""
    a = np.asarray(x, dtype=np.float64)
    n = np.sqrt(sq_norm(a))
    out = np.zeros_like(a)
    nz = n > 0.0
    out[nz] = a[nz] / n[nz][..., None]
    return out


def segment_point_min_dist_sq(p0: np.ndarray, p1: np.ndarray, origin: np.ndarray | None = None):
    """线段 ``p0 -> p1`` 到 ``origin``（默认原点）的最近距离平方。

    参数形状 ``(..., 3)``，返回形状 ``(...)``。
    """
    p0 = _as_vec3(p0)
    p1 = _as_vec3(p1)
    o = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    d = p1 - p0
    r = p0 - o
    dd = sq_norm(d)
    denom = np.where(dd > 0.0, dd, 1.0)
    t = np.clip(-dot(r, d) / denom, 0.0, 1.0)
    closest = r + t[..., None] * d
    return sq_norm(closest)


def segment_min_distance(p0: np.ndarray, p1: np.ndarray, origin: np.ndarray | None = None):
    """线段到点的最近距离（米）。"""
    return np.sqrt(np.maximum(segment_point_min_dist_sq(p0, p1, origin), 0.0))


def segment_sphere_inside_interval(
    p: np.ndarray,
    dp: np.ndarray,
    radius: float,
    *,
    eps: float = GEOM_EPS,
) -> tuple[float, float] | None:
    """求 ``p + t*dp``（``t in [0, 1]``）落在球 ``||x|| < radius`` 内的参数区间。

    返回 ``(t_enter, t_exit)``；若区间退化为一点（相切或恰好端点）返回 ``None``。
    起点已在球内时 ``t_enter = 0``；整段在球内返回 ``(0.0, 1.0)``。
    ``eps`` 用于把 |radius| 量级的容差归入几何判定。
    """
    p = _as_vec3(p)
    dp = _as_vec3(dp)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    a = float(sq_norm(dp))
    b = float(dot(p, dp))
    c = float(sq_norm(p)) - radius * radius

    if a <= eps * eps:
        # 静止：距离恒定
        return (0.0, 1.0) if c < -eps * radius else None

    disc = b * b - a * c
    if disc <= 0.0:
        # 切点或完全在球外；严格 < radius 时相切不构成区间
        return (0.0, 1.0) if c < -eps * radius else None
    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b - sqrt_disc) / a
    t2 = (-b + sqrt_disc) / a

    if c < -eps * radius:
        # 起点在球内
        t_out = min(max(t2, 0.0), 1.0)
        if t_out <= 0.0:
            return None
        return (0.0, t_out)

    t_in = max(t1, 0.0)
    t_out = min(t2, 1.0)
    if t_in >= t_out:
        return None
    return (t_in, t_out)


def segment_sphere_enter_time(
    p: np.ndarray,
    dp: np.ndarray,
    radius: float,
    *,
    eps: float = GEOM_EPS,
    closed: bool = True,
) -> float | None:
    """``||p + t*dp|| <= radius`` 在 ``t in [0, 1]`` 内最早出现的参数。

    ``closed=True`` 用于中心访问（达到边界算进入，``<=``）；``closed=False`` 用于严格
    侵犯判定（``<``）。起点已在球内 → ``0.0``。无进入 → ``None``。
    """
    p = _as_vec3(p)
    dp = _as_vec3(dp)
    c = float(sq_norm(p)) - radius * radius
    if closed:
        if c <= eps * radius:
            return 0.0
    else:
        if c < -eps * radius:
            return 0.0
    a = float(sq_norm(dp))
    b = float(dot(p, dp))
    if a <= eps * eps:
        return None
    disc = b * b - a * c
    if disc < 0.0:
        return None
    sqrt_disc = float(np.sqrt(max(disc, 0.0)))
    t1 = (-b - sqrt_disc) / a
    t2 = (-b + sqrt_disc) / a
    # t1 为进入根（从外到内）；若 t1 在 [0,1] 内则最早进入为 max(t1, 0)
    for t in (t1, t2):
        if -eps <= t <= 1.0 + eps:
            t_clip = min(max(t, 0.0), 1.0)
            if closed:
                return t_clip
            if t_clip > 0.0:
                return t_clip
    return None


def point_inside_sphere(p: np.ndarray, radius: float, *, eps: float = GEOM_EPS) -> np.ndarray:
    """``||p|| <= radius + eps``，返回布尔数组。"""
    p = _as_vec3(p)
    return sq_norm(p) <= (radius + eps) * (radius + eps)


def project_onto_ball(v: np.ndarray, v_max: float) -> np.ndarray:
    """把速度投影到半径 ``v_max`` 的球内（逐行）。"""
    v = np.asarray(v, dtype=np.float64)
    n = np.sqrt(sq_norm(v))
    scale = np.ones_like(n)
    over = n > v_max
    scale[over] = v_max / n[over]
    return v * scale[..., None]


def closest_point_on_segment_2d(
    p0: np.ndarray, p1: np.ndarray, origin: np.ndarray | None = None
) -> np.ndarray:
    """线段上距原点最近的点（与 :func:`segment_point_min_dist_sq` 对应）。"""
    p0 = _as_vec3(p0)
    p1 = _as_vec3(p1)
    o = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    d = p1 - p0
    r = p0 - o
    dd = sq_norm(d)
    denom = np.where(dd > 0.0, dd, 1.0)
    t = np.clip(-dot(r, d) / denom, 0.0, 1.0)
    return p0 + t[..., None] * d


def rotating_direction_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """由两向量构造正交方向（退化时用固定参考轴），返回与二者均正交的单位向量。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.cross(a, b)
    n = np.sqrt(sq_norm(c))
    out = np.zeros_like(c)
    good = n > GEOM_EPS
    out[good] = c[good] / n[good][..., None]
    if np.any(~good):
        ref = np.array([1.0, 0.0, 0.0])
        alt = np.array([0.0, 1.0, 0.0])
        for idx in np.argwhere(~good):
            ia = a[tuple(idx)]
            na = np.sqrt(sq_norm(ia))
            if na <= GEOM_EPS:
                out[tuple(idx)] = ref
                continue
            u = ia / na
            cand = np.cross(u, ref)
            if np.sqrt(sq_norm(cand)) <= GEOM_EPS:
                cand = np.cross(u, alt)
            out[tuple(idx)] = cand / np.sqrt(sq_norm(cand))
    return out


def fibonacci_sphere_directions(n: int, *, offset: float = 0.5) -> np.ndarray:
    """手写 Fibonacci 球面单位方向集合（handoff §3.1 同构，用于候选方向/初始化）。"""
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + offset) / n
    z = np.clip(z, -1.0, 1.0)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """罗德里格斯旋转矩阵（3,3）。axis 会被单位化。"""
    u = np.asarray(axis, dtype=np.float64)
    n = float(np.sqrt(sq_norm(u)))
    if n <= GEOM_EPS:
        return np.eye(3)
    u = u / n
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    ux, uy, uz = u
    return np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ],
        dtype=np.float64,
    )


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """由标准正态四元数得到的均匀随机旋转矩阵（确定性由 rng 决定）。"""
    q = rng.standard_normal(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def tangential_unit_vectors(
    dirs: np.ndarray, rng: np.random.Generator, *, max_tries: int = 64
) -> np.ndarray:
    """为每个单位方向 ``u`` 生成切平面内均匀单位向量。

    先取标准三维高斯向量并投影到切平面再归一化；退化时重采样（最多 ``max_tries``）。
    """
    u = _as_vec3(dirs)
    if u.ndim != 2:
        raise ValueError("dirs must have shape (n,3)")
    n = u.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    pending = np.ones(n, dtype=bool)
    for _ in range(max_tries):
        if not np.any(pending):
            break
        idx = np.flatnonzero(pending)
        g = rng.standard_normal((idx.size, 3))
        # 投影到 u 的正交平面
        proj = g - np.sum(g * u[idx], axis=1, keepdims=True) * u[idx]
        lens = np.sqrt(sq_norm(proj))
        ok = lens > 1e-9
        good_idx = idx[ok]
        out[good_idx] = proj[ok] / lens[ok][:, None]
        pending[good_idx] = False
    if np.any(pending):
        raise RuntimeError("degenerate tangent resampling exceeded max_tries")
    return out
