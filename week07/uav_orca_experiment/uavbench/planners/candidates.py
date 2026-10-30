"""共享候选速度集合与候选安全搜索（CONTRACTS.md §7、handoff §5.2/§5.3）。

本模块被 M1 (:class:`~uavbench.planners.vo3d.VO3DPlanner`) 与 M2
(:class:`~uavbench.planners.rvo3d.RVO3DPlanner`) 共用，M1/M2 必须拿到同一个
:class:`CandidateSet` 对象（场景级方向旋转只做一次）。

候选集合（CONTRACTS §7，固定索引顺序）
-------------------------------------
1. 零速度 / 当前速度 / ``v_pref``（按 ``include_*`` 开关，依次在前）；
2. 方向 × 速度层：方向索引升序、层升序；速度层按 ``v_max`` 截断并去重；
3. 去重容差 ``CANDIDATE_DEDUPE_TOL = 1e-9 m/s``，保留首次出现索引；
4. 默认 ``D=128``、三层 → 最多 ``3 + 128*3 = 387`` 个候选。

3D 有限预测窗危险判定（handoff §5.2，闭合线段 ``t in [0, tau]``）
-----------------------------------------------------------------
``p_ij = p_j - p_i``，候选速度 ``c``::

    VO : w = c - v_j
    RVO: w = 2c - v_i - v_j
    冻结邻机: w = c            （M1 与 M2 相同，不做互惠推理）

    危险 ⇔ min_{t in [0, tau]} ||p_ij - t*w|| < d_plan

闭区间使“当前已重叠”必然判危险（并计入 ``initial_overlap_pairs``）。
判定用球—线段最近距离的稳定二次式，不做二维投影/斥力近似。

实现上把三种情形统一成 ``w = c - B`` 且时间窗 ``[0, h]``::

    VO  活动邻机: B = v_j,             h = tau
    RVO 活动邻机: B = (v_i + v_j) / 2, h = 2*tau
    冻结邻机    : B = 0,               h = tau

RVO 的等价性来自 ``2c-v_i-v_j = 2*(c - (v_i+v_j)/2)``，令 ``u = 2t`` 后
``min_{t in [0,tau]} ||p - 2t(c-s)|| = min_{u in [0,2tau]} ||p - u(c-s)||``；
测试对每个候选逐点核对原始定义。

搜索与回退（CONTRACTS §7、handoff §5.3）
----------------------------------------
* 偏好键（字典序）：``||c-v_pref||^2`` → ``||c-v_i||^2`` → 固定候选索引；
* 先测偏好最优候选（通常是 ``v_pref``），若安全即返回（``candidate_count_tested=1``）；
  否则按偏好顺序分块向量化评测，第一个可接受候选即字典序最小者，
  与全量暴力结果逐位一致（测试覆盖）；
* 无可接受候选：``sample_exhausted=True``、``used_fallback=True``，按物理预测
  ``w = c - v_j``（与模型无关）计算全邻机最早接触时间 TTC（已重叠 0，无接触
  ``+inf``），最大化最小 TTC → 最小化 ``||c-v_pref||^2`` → 固定索引。

全部数组 float64、无全局可变状态；``plan()`` 对同一输入逐位可重复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .. import geometry
from ..rng import STREAM_CANDIDATE, make_rng

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from ..config import Config

__all__ = [
    "CANDIDATE_DEDUPE_TOL",
    "CandidateSet",
    "SearchResult",
    "build_candidate_set",
    "candidate_rotation",
    "danger_flags",
    "fallback_select",
    "fibonacci_directions",
    "first_contact_times",
    "min_closed_segment_distance",
    "min_relative_distances",
    "preference_keys",
    "preference_order",
    "relative_velocities",
    "rotate_directions",
    "search_velocities",
    "select_exhaustive",
    "squared_distances",
]

#: 候选去重容差（m/s，CONTRACTS §7）
CANDIDATE_DEDUPE_TOL: float = 1e-9

#: 视为“零相对速度”的阈值（m/s），仅用于二次式退化处理
_REL_VEL_EPS: float = 1e-12
_A_TOL: float = _REL_VEL_EPS * _REL_VEL_EPS

#: 邻机剔除的安全余量（m）：只在可证安全时剔除，不影响严格危险判定
_CULL_SLACK: float = 1e-9

#: 逐块枚举候选的块大小（几何增长：先测 v_pref，再做中等规模向量化评估；
#: 剩余部分一次性扫描。基准显示比等步长小块更快且对“全量不可接受”最坏情形友好）
_CHUNK_SIZES: tuple[int, ...] = (1, 16, 64)

_VO = "vo"
_RVO = "rvo"
_MODELS = (_VO, _RVO)


# ---------------------------------------------------------------------------
# 方向与候选集合
# ---------------------------------------------------------------------------
def fibonacci_directions(n_directions: int) -> np.ndarray:
    """Fibonacci 球面单位方向 ``(D,3)``（与 handoff §3.1 同构，仅用于方向）。"""
    return geometry.fibonacci_sphere_directions(int(n_directions))


def rotate_directions(dirs: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """用正交旋转矩阵整体旋转方向集合，并重新归一化以抑制漂移。"""
    d = np.asarray(dirs, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"dirs must have shape (D,3), got {d.shape}")
    if r.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3,3), got {r.shape}")
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(r))):
        raise ValueError("dirs/rotation must be finite")
    return geometry.normalize(d @ r.T)


def candidate_rotation(seed: int) -> np.ndarray:
    """场景级候选方向整体旋转矩阵 ``(3,3)``（``candidate_rng`` = STREAM_CANDIDATE）。"""
    rng = make_rng(int(seed), STREAM_CANDIDATE)
    return geometry.random_rotation_matrix(rng)


def _validate_ball_vector(v: np.ndarray, v_max: float, *, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    n = float(np.sqrt(geometry.sq_norm(arr)))
    if n > float(v_max) + geometry.VEL_EPS:
        arr = arr * (float(v_max) / n)
    return np.array(arr, dtype=np.float64, copy=True)


def _dedupe_rows(rows: np.ndarray, tol: float) -> np.ndarray:
    """按首次出现顺序做欧氏容差去重（构造期一次性 O(D^2)，D<=256）。"""
    tol2 = float(tol) * float(tol)
    kept: list[np.ndarray] = []
    for i in range(rows.shape[0]):
        cand = rows[i]
        duplicate = any(float(geometry.sq_norm(cand - other)) <= tol2 for other in kept)
        if not duplicate:
            kept.append(cand)
    return np.array(kept, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class CandidateSet:
    """固定索引顺序的候选速度集合（场景级构建一次，只读、确定性）。

    ``dirs`` 必须是**已经旋转好**的 ``(D,3)`` 单位方向；如需从配置构造并施加
    场景旋转，使用 :meth:`from_config` / :func:`build_candidate_set`。
    ``build`` 对每个活动机当步的 ``v_current``/``v_pref`` 生成具体候选数组。
    """

    dirs: np.ndarray
    speeds: tuple[float, ...] = (0.5, 1.0, 2.0)
    include_zero: bool = True
    include_current: bool = True
    include_pref: bool = True
    #: 每个 v_max 的静态网格缓存（纯缓存，不改变结果）
    _grid_cache: dict[float, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, compare=False, repr=False
    )

    # ---- 构造 -------------------------------------------------------------
    def __post_init__(self) -> None:
        raw = np.asarray(self.dirs, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(f"dirs must have shape (D,3), got {raw.shape}")
        if raw.shape[0] == 0:
            raise ValueError("dirs must be non-empty")
        if not np.all(np.isfinite(raw)):
            raise ValueError("dirs must be finite")
        norms = np.sqrt(geometry.sq_norm(raw))
        if np.any(np.abs(norms - 1.0) > 1e-9):
            raise ValueError("dirs must be unit vectors (|n-1|<=1e-9)")
        dirs = _dedupe_rows(raw, CANDIDATE_DEDUPE_TOL)
        dirs.setflags(write=False)
        object.__setattr__(self, "dirs", dirs)

        speeds = tuple(sorted(float(s) for s in self.speeds))
        if not speeds:
            raise ValueError("speeds must be non-empty")
        for s in speeds:
            if not np.isfinite(s) or s <= 0.0:
                raise ValueError(f"speed layer must be positive and finite, got {s}")
        object.__setattr__(self, "speeds", speeds)
        object.__setattr__(self, "include_zero", bool(self.include_zero))
        object.__setattr__(self, "include_current", bool(self.include_current))
        object.__setattr__(self, "include_pref", bool(self.include_pref))
        object.__setattr__(self, "_grid_cache", {})

    @classmethod
    def from_directions(
        cls,
        n_directions: int = 128,
        *,
        rotation: np.ndarray | None = None,
        speeds: tuple[float, ...] = (0.5, 1.0, 2.0),
        include_zero: bool = True,
        include_current: bool = True,
        include_pref: bool = True,
    ) -> "CandidateSet":
        """由 Fibonacci 方向数构造；``rotation`` 为可选的场景级整体旋转。"""
        base = fibonacci_directions(int(n_directions))
        if rotation is not None:
            base = rotate_directions(base, rotation)
        return cls(
            dirs=base,
            speeds=tuple(speeds),
            include_zero=bool(include_zero),
            include_current=bool(include_current),
            include_pref=bool(include_pref),
        )

    @classmethod
    def from_config(
        cls, cfg: "Config", rotation: np.ndarray | None = None
    ) -> "CandidateSet":
        """按 ``cfg.planners.candidate`` 构造，可选施加场景级旋转矩阵。"""
        cand = cfg.planners.candidate
        return cls.from_directions(
            int(cand.directions),
            rotation=rotation,
            speeds=tuple(float(s) for s in cand.speed_layers),
            include_zero=bool(cand.include_zero),
            include_current=bool(cand.include_current),
            include_pref=bool(cand.include_pref),
        )

    # ---- 速度层与静态网格 ------------------------------------------------
    def effective_speeds(self, v_max: float) -> tuple[float, ...]:
        """按 ``v_max`` 截断后的速度层（升序、去重；默认配置 → (0.5, 1.0, 2.0)）。"""
        vmax = float(v_max)
        if not np.isfinite(vmax) or vmax <= 0.0:
            raise ValueError("v_max must be positive and finite")
        out: list[float] = []
        for s in self.speeds:
            eff = min(float(s), vmax)
            if eff <= 0.0:
                continue
            if out and abs(eff - out[-1]) <= CANDIDATE_DEDUPE_TOL:
                continue
            out.append(eff)
        if not out:
            raise ValueError("no positive candidate speed layer")
        return tuple(out)

    def max_speed(self, v_max: float) -> float:
        """候选集中最大的速度层幅值（用于安全的邻机剔除上界）。"""
        return self.effective_speeds(v_max)[-1]

    def n_template_candidates(self) -> int:
        """固定模板（去重前）候选数：``#prefix + D*len(speeds)``。"""
        return self._n_prefix() + int(self.dirs.shape[0]) * len(self.speeds)

    def _n_prefix(self) -> int:
        return (
            int(self.include_zero) + int(self.include_current) + int(self.include_pref)
        )

    def _grid_for(self, v_max: float) -> tuple[np.ndarray, np.ndarray]:
        """``(G,3)`` 静态候选（方向×速度层，方向优先、层升序）与其范数平方。"""
        key = float(v_max)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        layers = np.asarray(self.effective_speeds(key), dtype=np.float64)
        grid = (self.dirs[:, None, :] * layers[None, :, None]).reshape(-1, 3)
        grid.setflags(write=False)
        grid_norms = geometry.sq_norm(grid)
        grid_norms.setflags(write=False)
        self._grid_cache[key] = (grid, grid_norms)
        return grid, grid_norms

    # ---- 每机候选构造 ----------------------------------------------------
    def _build_with_meta(
        self, v_max: float, v_current: np.ndarray, v_pref: np.ndarray
    ) -> tuple[np.ndarray, dict[str, object]]:
        """:meth:`build` 的内部版本，额外返回前缀候选行号与截断后的输入速度。

        meta 键：``"zero"/"current"/"pref"`` → 行号（-1=被去重），
        ``"v_current"``/``"v_pref"`` → 用于偏好键的截断后速度。
        """
        vmax = float(v_max)
        grid, _grid_norms = self._grid_for(vmax)
        vc = _validate_ball_vector(v_current, vmax, name="v_current")
        vp = _validate_ball_vector(v_pref, vmax, name="v_pref")
        prefix: list[tuple[str, np.ndarray]] = []
        if self.include_zero:
            prefix.append(("zero", np.zeros(3, dtype=np.float64)))
        if self.include_current:
            prefix.append(("current", vc))
        if self.include_pref:
            prefix.append(("pref", vp))
        tol2 = CANDIDATE_DEDUPE_TOL * CANDIDATE_DEDUPE_TOL
        kept: list[np.ndarray] = []
        meta: dict[str, object] = {
            "zero": -1,
            "current": -1,
            "pref": -1,
            "v_current": vc,
            "v_pref": vp,
        }
        for kind, cand in prefix:
            duplicate = False
            for other in kept:
                if float(geometry.sq_norm(cand - other)) <= tol2:
                    duplicate = True
                    break
            if not duplicate and grid.shape[0]:
                # 差分式距离：避免展开式 |p|²+|g|²-2p·g 的相消误差（1e-9 容差敏感）
                dist2 = geometry.sq_norm(grid - cand[None, :])
                if float(np.min(dist2)) <= tol2:
                    duplicate = True
            if not duplicate:
                meta[kind] = len(kept)
                kept.append(cand)
        if kept:
            array = np.concatenate([np.asarray(kept, dtype=np.float64), grid], axis=0)
        else:
            array = np.array(grid, dtype=np.float64, copy=True)
        return array, meta

    def build(
        self, v_max: float, v_current: np.ndarray, v_pref: np.ndarray
    ) -> np.ndarray:
        """返回单个活动机的候选速度 ``(C,3)`` float64，固定索引顺序。

        顺序：``[零速度][当前速度][v_pref][(方向0,层0),(方向0,层1),...]``，
        去重容差 1e-9 m/s（保留首次出现）；``v_current``/``v_pref`` 超出
        ``v_max`` 超过 ``VEL_EPS`` 时按 ``v_max`` 截断（协议内输入不变）。
        """
        return self._build_with_meta(v_max, v_current, v_pref)[0]

    def build_index_map(self, n_cand: int) -> np.ndarray:
        """固定构造模板的候选来源标签 ``(n_cand,3)`` int32。

        列为 ``[kind, direction_index, speed_index]``，``kind``: 0=零速度,
        1=当前速度, 2=v_pref, 3=方向×速度层；前缀项的方向/层为 -1。
        模板是**去重前**的固定顺序，用于诊断“第 k 个候选从哪来”。
        """
        n = int(n_cand)
        if n < 0:
            raise ValueError("n_cand must be non-negative")
        total = self.n_template_candidates()
        if n > total:
            raise ValueError(f"n_cand={n} exceeds template size {total}")
        kinds: list[int] = []
        if self.include_zero:
            kinds.append(0)
        if self.include_current:
            kinds.append(1)
        if self.include_pref:
            kinds.append(2)
        out = np.zeros((n, 3), dtype=np.int32)
        out[:, 0] = 3
        n_prefix = min(n, len(kinds))
        if n_prefix:
            out[:n_prefix, 0] = np.asarray(kinds[:n_prefix], dtype=np.int32)
        rest = n - n_prefix
        if rest > 0:
            n_speeds = len(self.speeds)
            flat = np.arange(rest, dtype=np.int32)
            out[n_prefix:, 1] = flat // n_speeds
            out[n_prefix:, 2] = flat % n_speeds
        return out


def build_candidate_set(cfg: "Config", seed: int) -> CandidateSet:
    """场景级候选集合：128 Fibonacci 方向 + 一次整体旋转（``STREAM_CANDIDATE``）。

    同一 ``cfg``/``seed`` → 逐位相同的 ``dirs``；仿真对每个场景调用一次，并把
    同一个对象同时传给 M1(vo) 与 M2(rvo)。
    """
    return CandidateSet.from_config(cfg, rotation=candidate_rotation(seed))


# ---------------------------------------------------------------------------
# 复用扫描缓冲（纯缓存，不改变结果；每个 search_velocities 调用内复用于各机）
# ---------------------------------------------------------------------------
class _ScanBuffers:
    """``(C,M)`` 浮点/布尔暂存区，避免逐机重复分配，保证缓存友好。"""

    __slots__ = ("cap_c", "cap_m", "a", "b", "t", "g", "aux", "disc", "valid")

    def __init__(self, cap_c: int, cap_m: int) -> None:
        self.cap_c = max(int(cap_c), 1)
        self.cap_m = max(int(cap_m), 1)
        shape = (self.cap_c, self.cap_m)
        self.a = np.empty(shape, dtype=np.float64)
        self.b = np.empty(shape, dtype=np.float64)
        self.t = np.empty(shape, dtype=np.float64)
        self.g = np.empty(shape, dtype=np.float64)
        self.aux = np.empty(shape, dtype=np.float64)
        self.disc = np.empty(shape, dtype=np.float64)
        self.valid = np.empty(shape, dtype=bool)


def _danger_flags_ws(
    cand: np.ndarray,
    cc: np.ndarray,
    p: np.ndarray,
    pp: np.ndarray,
    base: np.ndarray,
    b2: np.ndarray,
    pb: np.ndarray,
    horizon: np.ndarray,
    d2: float,
    buf: _ScanBuffers,
) -> np.ndarray:
    """向量化危险判定核（复用缓冲）：``w = c - base``，返回 ``(C,)`` bool。"""
    n = int(cand.shape[0])
    m = int(p.shape[0])
    if m == 0:
        return np.zeros(n, dtype=bool)
    if n > buf.cap_c or m > buf.cap_m:
        raise ValueError("scan buffers too small")
    a = buf.a[:n, :m]
    np.matmul(cand, base.T, out=a)
    a *= -2.0
    a += cc[:, None]
    a += b2[None, :]
    b = buf.b[:n, :m]
    np.matmul(cand, p.T, out=b)
    b -= pb[None, :]
    t = buf.t[:n, :m]
    t.fill(0.0)
    np.divide(b, a, out=t, where=a > _A_TOL)
    np.clip(t, 0.0, horizon[None, :], out=t)
    g = buf.g[:n, :m]
    np.multiply(t, b, out=g)
    g *= -2.0
    g += pp[None, :]
    aux = buf.aux[:n, :m]
    np.multiply(t, t, out=aux)
    aux *= a
    g += aux
    return np.any(g < d2, axis=1)


def _ttc_min_ws(
    cand: np.ndarray,
    v_j: np.ndarray,
    p_ij: np.ndarray,
    d_plan: float,
    buf: _ScanBuffers,
) -> np.ndarray:
    """TTC 回退核：物理预测 ``w = c - v_j`` 的最早接触时间 ``(C,)``。

    已重叠（``||p_ij|| < d_plan``）→ 0；无未来接触 → ``+inf``。
    """
    n = int(cand.shape[0])
    m = int(p_ij.shape[0])
    if m == 0:
        return np.full(n, np.inf, dtype=np.float64)
    if n > buf.cap_c or m > buf.cap_m:
        raise ValueError("scan buffers too small")
    d2 = float(d_plan) * float(d_plan)
    vj = np.asarray(v_j, dtype=np.float64)
    p = np.asarray(p_ij, dtype=np.float64)
    cc = geometry.sq_norm(cand)
    vv = geometry.sq_norm(vj)
    pv = np.einsum("mi,mi->m", p, vj)
    pp = geometry.sq_norm(p)
    a = buf.a[:n, :m]
    np.matmul(cand, vj.T, out=a)
    a *= -2.0
    a += cc[:, None]
    a += vv[None, :]
    b = buf.b[:n, :m]
    np.matmul(cand, p.T, out=b)
    b -= pv[None, :]
    c0 = pp - d2  # (M,)
    inside = c0 < 0.0
    disc = buf.disc[:n, :m]
    np.multiply(b, b, out=disc)
    aux = buf.aux[:n, :m]
    np.multiply(a, c0[None, :], out=aux)
    disc -= aux
    valid = buf.valid[:n, :m]
    np.greater(b, 0.0, out=valid)
    valid &= disc > 0.0
    valid &= a > _A_TOL
    if bool(np.any(inside)):
        valid[:, inside] = False
    t = buf.t[:n, :m]
    np.maximum(disc, 0.0, out=disc)
    np.sqrt(disc, out=disc)
    np.subtract(b, disc, out=t)
    np.divide(t, a, out=t, where=valid)
    t[~valid] = np.inf
    out = np.min(t, axis=1)
    if bool(np.any(inside)):
        out[:] = 0.0
    return out


# ---------------------------------------------------------------------------
# 3D 有限预测窗几何（模型定义层，测试与共享搜索都使用）
# ---------------------------------------------------------------------------
def _as_candidates(candidates: np.ndarray) -> np.ndarray:
    arr = np.asarray(candidates, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"candidates must have shape (C,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("candidates must be finite")
    return arr


def relative_velocities(
    candidates: np.ndarray,
    *,
    model: str,
    v_i: np.ndarray,
    v_j: np.ndarray,
    frozen_j: np.ndarray,
) -> np.ndarray:
    """模型定义的相对速度 ``w``，形状 ``(C,M,3)``（测试/参考实现用）。

    VO：``c - v_j``；RVO：``2c - v_i - v_j``；冻结邻机（两模型统一）：
    ``w = c``。``v_j``/``frozen_j`` 形状 ``(M,3)``/``(M,)``。
    """
    if model not in _MODELS:
        raise ValueError(f"unknown model: {model!r}")
    cand = _as_candidates(candidates)
    vj = np.asarray(v_j, dtype=np.float64)
    fz = np.asarray(frozen_j, dtype=bool)
    if vj.ndim != 2 or vj.shape[1] != 3 or fz.shape != (vj.shape[0],):
        raise ValueError("v_j must be (M,3) and frozen_j (M,)")
    c = cand[:, None, :]
    if model == _VO:
        w = c - vj[None, :, :]
    else:
        u = np.asarray(v_i, dtype=np.float64)[None, None, :] + vj[None, :, :]
        w = 2.0 * c - u
    return np.where(fz[None, :, None], c, w)


def min_closed_segment_distance(
    p_ij: np.ndarray, w: np.ndarray, tau: float
) -> np.ndarray:
    """``min_{t in [0,tau]} ||p_ij - t*w||``（直接几何定义，参考实现）。

    ``p_ij`` 形状 ``(M,3)`` 或可广播；``w`` 形状 ``(...,3)``。等价于线段
    ``p_ij -> p_ij - tau*w`` 到原点的最近距离。
    """
    p = np.asarray(p_ij, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError("p_ij must have last dimension 3")
    ww = np.asarray(w, dtype=np.float64)
    if ww.shape[-1] != 3:
        raise ValueError("w must have last dimension 3")
    return geometry.segment_min_distance(p, p - float(tau) * ww)


def _neighbor_terms(
    model: str, v_i: np.ndarray, v_j: np.ndarray, frozen_j: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """统一形式 ``w = c - base``、时间窗 ``[0, horizon]`` 的邻机参数。"""
    vj = np.asarray(v_j, dtype=np.float64)
    fz = np.asarray(frozen_j, dtype=bool)
    m = int(vj.shape[0])
    if model == _VO:
        base = np.where(fz[:, None], 0.0, vj)
        horizon = np.full(m, float(tau), dtype=np.float64)
        return base, horizon
    if model == _RVO:
        s = 0.5 * (np.asarray(v_i, dtype=np.float64)[None, :] + vj)
        base = np.where(fz[:, None], 0.0, s)
        horizon = np.where(fz, float(tau), 2.0 * float(tau)).astype(np.float64)
        return base, horizon
    raise ValueError(f"unknown model: {model!r}")


def _danger_single_culled(
    c: np.ndarray,
    cc: float,
    p: np.ndarray,
    pp: np.ndarray,
    base: np.ndarray,
    b2: np.ndarray,
    pb: np.ndarray,
    horizon: np.ndarray,
    d_plan: float,
    d2: float,
) -> bool:
    """单候选危险判定（带可证安全邻机剔除），用于“首选候选”快速路径。"""
    w = c[None, :] - base  # (M,3)
    a = geometry.sq_norm(w)  # (M,)
    b = c @ p.T - pb  # (M,) = p·w
    reach = float(d_plan) + horizon * np.sqrt(a)
    keep = pp <= (reach + _CULL_SLACK) ** 2
    if not bool(np.any(keep)):
        return False
    aa = a[keep]
    bb = b[keep]
    pps = pp[keep]
    hh = horizon[keep]
    pos = aa > _A_TOL
    t = np.zeros_like(aa)
    np.divide(bb, aa, out=t, where=pos)
    np.clip(t, 0.0, hh, out=t)
    g = pps - 2.0 * t * bb + t * t * aa
    return bool(np.any(g < d2))


def danger_flags(
    candidates: np.ndarray,
    *,
    model: str,
    v_i: np.ndarray,
    v_j: np.ndarray,
    frozen_j: np.ndarray,
    p_ij: np.ndarray,
    tau: float,
    d_plan: float,
) -> np.ndarray:
    """候选危险标志 ``(C,)`` bool（生产判定核，向量化，闭合时间窗 ``[0,tau]``）。

    危险 ⇔ 存在 ``t in [0,tau]`` 使 ``||p_ij - t*w|| < d_plan``，``w`` 按模型定义。
    本函数是加速搜索使用的同一判定核；全量调用即“暴力枚举”的安全判定部分。
    """
    if model not in _MODELS:
        raise ValueError(f"unknown model: {model!r}")
    cand = _as_candidates(candidates)
    vj = np.asarray(v_j, dtype=np.float64)
    p = np.asarray(p_ij, dtype=np.float64)
    fz = np.asarray(frozen_j, dtype=bool)
    if (
        vj.ndim != 2
        or vj.shape[1] != 3
        or p.shape != vj.shape
        or fz.shape != (p.shape[0],)
    ):
        raise ValueError("v_j (M,3), p_ij (M,3), frozen_j (M,) expected")
    m = p.shape[0]
    if m == 0:
        return np.zeros(cand.shape[0], dtype=bool)
    base, horizon = _neighbor_terms(model, v_i, vj, fz, float(tau))
    cc = geometry.sq_norm(cand)
    pp = geometry.sq_norm(p)
    b2 = geometry.sq_norm(base)
    pb = np.einsum("mi,mi->m", p, base)
    buf = _ScanBuffers(cand.shape[0], m)
    return _danger_flags_ws(
        cand, cc, p, pp, base, b2, pb, horizon, float(d_plan) ** 2, buf
    )


def first_contact_times(
    candidates: np.ndarray, v_j: np.ndarray, p_ij: np.ndarray, d_plan: float
) -> np.ndarray:
    """回退用 TTC：每个候选对全邻机的最早接触时间 ``(C,)``。

    物理预测 ``w = c - v_j``（与 VO/RVO 模型无关；冻结邻机 ``v_j = 0``）。
    已重叠（当前 ``||p_ij|| < d_plan``）记 0；无未来接触记 ``+inf``。
    """
    cand = _as_candidates(candidates)
    vj = np.asarray(v_j, dtype=np.float64)
    p = np.asarray(p_ij, dtype=np.float64)
    if vj.ndim != 2 or vj.shape[1] != 3 or p.shape != vj.shape:
        raise ValueError("v_j and p_ij must both be (M,3)")
    if p.shape[0] == 0:
        return np.full(cand.shape[0], np.inf, dtype=np.float64)
    buf = _ScanBuffers(cand.shape[0], p.shape[0])
    return _ttc_min_ws(cand, vj, p, float(d_plan), buf)


def min_relative_distances(
    candidates: np.ndarray,
    *,
    model: str,
    v_i: np.ndarray,
    v_j: np.ndarray,
    frozen_j: np.ndarray,
    p_ij: np.ndarray,
    tau: float,
) -> np.ndarray:
    """模型定义下每个候选对每个邻机的最小距离 ``(C,M)``（参考/诊断用）。"""
    if model not in _MODELS:
        raise ValueError(f"unknown model: {model!r}")
    cand = _as_candidates(candidates)
    w = relative_velocities(cand, model=model, v_i=v_i, v_j=v_j, frozen_j=frozen_j)
    p = np.asarray(p_ij, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p_ij must be (M,3)")
    p0 = p[None, :, :]
    closest_sq = geometry.segment_point_min_dist_sq(p0, p0 - float(tau) * w)
    return np.sqrt(np.maximum(closest_sq, 0.0))


# ---------------------------------------------------------------------------
# 偏好键、暴力选择与 TTC 回退
# ---------------------------------------------------------------------------
def squared_distances(candidates: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``||c - v||^2``，形状 ``(C,)``（差分式，避免展开式的相消误差）。"""
    cand = _as_candidates(candidates)
    vv = np.asarray(v, dtype=np.float64)
    if vv.shape != (3,):
        raise ValueError("v must have shape (3,)")
    return geometry.sq_norm(cand - vv[None, :])


def preference_keys(
    candidates: np.ndarray, v_current: np.ndarray, v_pref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """偏好键 ``(||c-v_pref||^2, ||c-v_i||^2)``。"""
    return squared_distances(candidates, v_pref), squared_distances(
        candidates, v_current
    )


def preference_order(
    candidates: np.ndarray, v_current: np.ndarray, v_pref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """固定索引顺序下的偏好排序 ``(order, d_pref, d_cur)``。

    字典序：``||c-v_pref||^2`` 升序 → ``||c-v_i||^2`` 升序 → 候选索引升序。
    """
    d_pref, d_cur = preference_keys(candidates, v_current, v_pref)
    n = int(d_pref.shape[0])
    order = np.lexsort((np.arange(n, dtype=np.int64), d_cur, d_pref))
    return order, d_pref, d_cur


def select_exhaustive(
    candidates: np.ndarray,
    dangerous: np.ndarray,
    v_current: np.ndarray,
    v_pref: np.ndarray,
) -> int:
    """暴力选择：在全部安全候选中取偏好字典序最小者；无可接受候选返回 -1。"""
    cand = _as_candidates(candidates)
    danger = np.asarray(dangerous, dtype=bool).reshape(-1)
    if danger.shape != (cand.shape[0],):
        raise ValueError("dangerous must have shape (C,)")
    acceptable = np.flatnonzero(~danger)
    if acceptable.size == 0:
        return -1
    d_pref, d_cur = preference_keys(cand, v_current, v_pref)
    order = np.lexsort((acceptable, d_cur[acceptable], d_pref[acceptable]))
    return int(acceptable[order[0]])


def fallback_select(
    candidates: np.ndarray,
    v_j: np.ndarray,
    p_ij: np.ndarray,
    d_plan: float,
    v_pref: np.ndarray,
) -> tuple[int, float]:
    """TTC 回退排序：最大化最小 TTC → 最小化 ``||c-v_pref||^2`` → 固定索引。

    返回 ``(index, min_ttc)``；最小 TTC 为 ``+inf`` 表示与所有邻机均无未来接触。
    """
    cand = _as_candidates(candidates)
    ttc = first_contact_times(cand, v_j, p_ij, d_plan)
    d_pref = squared_distances(cand, v_pref)
    order = np.lexsort((np.arange(cand.shape[0], dtype=np.int64), d_pref, -ttc))
    index = int(order[0])
    return index, float(ttc[index])


# ---------------------------------------------------------------------------
# 共享批量搜索（M1/M2 的唯一求解入口）
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    """批量候选搜索结果（形状均为 ``(A,)``，与 ``active_ids`` 对齐）。"""

    velocities: np.ndarray  # (A,3) float64
    sample_exhausted: np.ndarray  # (A,) bool
    used_fallback: np.ndarray  # (A,) bool
    candidate_count_tested: np.ndarray  # (A,) int32
    initial_overlap_pairs: np.ndarray  # (A,) int32
    constraint_violation_max: np.ndarray  # (A,) float64
    preferred_speed: np.ndarray  # (A,) float64
    fallback_min_ttc: np.ndarray  # (A,) float64（非回退/无接触为 +inf）


def _constraint_violation_batch(
    velocities: np.ndarray,
    *,
    model: str,
    v_current: np.ndarray,
    p_ij: np.ndarray,
    v_j: np.ndarray,
    frozen_j: np.ndarray,
    neighbor_mask: np.ndarray,
    tau: float,
    d_plan: float,
) -> np.ndarray:
    """批量诊断：返回速度对全部有效邻机的最大约束残差 ``(A,)``。

    ``constraint_violation_max = max(0, d_plan - min_t dist)`` 按模型定义计算，
    无效邻机不参与。
    """
    n_active = int(velocities.shape[0])
    out = np.zeros(n_active, dtype=np.float64)
    m = int(p_ij.shape[1]) if p_ij.ndim == 3 else 0
    if n_active == 0 or m == 0:
        return out
    vc = np.asarray(v_current, dtype=np.float64)
    vj = np.asarray(v_j, dtype=np.float64)
    if model == _RVO:
        base_all = np.where(
            frozen_j[..., None], 0.0, 0.5 * (vc[:, None, :] + vj)
        )
        horizon_all = np.where(frozen_j, float(tau), 2.0 * float(tau))
    else:
        base_all = np.where(frozen_j[..., None], 0.0, vj)
        horizon_all = np.full((n_active, m), float(tau), dtype=np.float64)
    w = np.asarray(velocities, dtype=np.float64)[:, None, :] - base_all
    a = geometry.sq_norm(w)  # (A,M)
    b = np.einsum("ami,ami->am", p_ij, w)
    pp = geometry.sq_norm(p_ij)  # (A,M)
    pos = a > _A_TOL
    t = np.zeros_like(a)
    np.divide(b, a, out=t, where=pos)
    np.clip(t, 0.0, horizon_all, out=t)
    g = pp - 2.0 * t * b + t * t * a
    dist = np.sqrt(np.maximum(g, 0.0))
    viol = np.maximum(float(d_plan) - dist, 0.0)
    viol = np.where(np.asarray(neighbor_mask, dtype=bool), viol, 0.0)
    return np.max(viol, axis=1)


def _search_one(
    candidate_set: CandidateSet,
    *,
    model: str,
    v_max: float,
    tau: float,
    d_plan: float,
    v_current: np.ndarray,
    v_pref: np.ndarray,
    p_ij: np.ndarray,
    v_j: np.ndarray,
    frozen_j: np.ndarray,
    buf: _ScanBuffers,
) -> tuple[np.ndarray, int, bool, float]:
    """单机搜索：返回 ``(velocity, candidate_count_tested, exhausted, min_ttc)``。"""
    cand, meta = candidate_set._build_with_meta(v_max, v_current, v_pref)
    n_cand = int(cand.shape[0])
    d2 = float(d_plan) * float(d_plan)
    vp = np.asarray(meta["v_pref"], dtype=np.float64)
    vc = np.asarray(meta["v_current"], dtype=np.float64)
    pref_row = int(meta["pref"])  # type: ignore[arg-type]

    m = int(p_ij.shape[0])
    if m == 0:
        best = pref_row if pref_row >= 0 else _argmin_preference(cand, vc, vp)
        return np.array(cand[best], copy=True), 1, False, np.inf

    pp = geometry.sq_norm(p_ij)
    base, horizon = _neighbor_terms(model, vc, v_j, frozen_j, float(tau))
    b2 = geometry.sq_norm(base)
    pb = np.einsum("mi,mi->m", p_ij, base)
    cc = geometry.sq_norm(cand)

    # ---- 快速路径：先测偏好最优候选（通常是 v_pref） --------------------
    if pref_row >= 0:
        if not _danger_single_culled(
            cand[pref_row],
            float(cc[pref_row]),
            p_ij,
            pp,
            base,
            b2,
            pb,
            horizon,
            float(d_plan),
            d2,
        ):
            return np.array(cand[pref_row], copy=True), 1, False, np.inf

    # ---- 全量偏好顺序 + 分块向量化评测 --------------------------------
    d_pref = squared_distances(cand, vp)
    d_cur = squared_distances(cand, vc)
    order = np.lexsort((np.arange(n_cand, dtype=np.int64), d_cur, d_pref))

    # 安全上界剔除：仅在可证不可能进入 d_plan 时剔除邻机（不改变判定结果）。
    max_norm = float(np.sqrt(np.max(cc)))
    reach = float(d_plan) + horizon * (max_norm + np.sqrt(b2))
    keep = pp <= (reach + _CULL_SLACK) ** 2
    if not bool(np.all(keep)):
        p_use = p_ij[keep]
        pp_use = pp[keep]
        base_use = base[keep]
        b2_use = b2[keep]
        pb_use = pb[keep]
        horizon_use = horizon[keep]
        m_use = int(p_use.shape[0])
    else:
        p_use = p_ij
        pp_use = pp
        base_use = base
        b2_use = b2
        pb_use = pb
        horizon_use = horizon
        m_use = m
    if m_use == 0:
        if pref_row < 0:
            pref_row = _argmin_preference(cand, vc, vp)
        return np.array(cand[pref_row], copy=True), 1, False, np.inf

    tested = 0
    chosen = -1
    position = 0
    chunk_index = 0
    while position < n_cand:
        if chunk_index < len(_CHUNK_SIZES):
            size = _CHUNK_SIZES[chunk_index]
        else:
            size = n_cand - position
        size = min(size, n_cand - position)
        subset = order[position : position + size]
        flags = _danger_flags_ws(
            cand[subset],
            cc[subset],
            p_use,
            pp_use,
            base_use,
            b2_use,
            pb_use,
            horizon_use,
            d2,
            buf,
        )
        tested += int(size)
        acceptable = np.flatnonzero(~flags)
        if acceptable.size:
            chosen = int(subset[int(acceptable[0])])
            break
        position += size
        chunk_index += 1

    if chosen >= 0:
        return np.array(cand[chosen], copy=True), tested, False, np.inf

    # ---- 回退：物理预测 TTC 排序（全邻机，不做剔除） --------------------
    ttc = _ttc_min_ws(cand, v_j, p_ij, d_plan, buf)
    d_pref = squared_distances(cand, vp)
    fb_order = np.lexsort((np.arange(n_cand, dtype=np.int64), d_pref, -ttc))
    fb_index = int(fb_order[0])
    return np.array(cand[fb_index], copy=True), tested, True, float(ttc[fb_index])


def _argmin_preference(
    cand: np.ndarray, v_current: np.ndarray, v_pref: np.ndarray
) -> int:
    """无邻机时按 ``(||c-v_pref||², ||c-v_i||², 索引)`` 取最优候选。"""
    d_pref = squared_distances(cand, v_pref)
    d_cur = squared_distances(cand, v_current)
    order = np.lexsort(
        (np.arange(int(cand.shape[0]), dtype=np.int64), d_cur, d_pref)
    )
    return int(order[0])


def search_velocities(
    candidate_set: CandidateSet,
    *,
    model: str,
    v_max: float,
    tau: float,
    d_plan: float,
    v_current: np.ndarray,
    v_pref: np.ndarray,
    p_ij: np.ndarray,
    v_j: np.ndarray,
    frozen_j: np.ndarray,
    neighbor_mask: np.ndarray,
) -> SearchResult:
    """批量求解活动机命令速度（M1/M2 共享，向量化 + 确定性回退）。

    参数
    ----
    v_current : (A,3)
        活动机当前速度 ``v_i``（冻结者为 0，已由仿真器保证）。
    v_pref : (A,3)
        期望速度（handoff §4.5）。
    p_ij : (A,M,3)
        ``p_ij = p_j - p_i``；无效位内容任意。
    v_j : (A,M,3)
        邻机当前速度；冻结邻机必须为 0。
    frozen_j : (A,M) bool
        邻机是否冻结（保留在邻居列表中，不剔除）。
    neighbor_mask : (A,M) bool
        有效邻机掩码（``neighbor_ids == -1`` 处为 False）。

    返回 :class:`SearchResult`，速度满足 ``||v|| <= v_max`` 且有限。
    """
    if model not in _MODELS:
        raise ValueError(f"unknown model: {model!r}")
    vc = np.asarray(v_current, dtype=np.float64)
    vp = np.asarray(v_pref, dtype=np.float64)
    offsets = np.asarray(p_ij, dtype=np.float64)
    vj = np.asarray(v_j, dtype=np.float64)
    fz = np.asarray(frozen_j, dtype=bool)
    mask = np.asarray(neighbor_mask, dtype=bool)
    if vc.ndim != 2 or vc.shape[1] != 3:
        raise ValueError("v_current must be (A,3)")
    n_active = int(vc.shape[0])
    if vp.shape != (n_active, 3):
        raise ValueError("v_pref must be (A,3)")
    if offsets.ndim != 3 or offsets.shape[0] != n_active or offsets.shape[2] != 3:
        raise ValueError("p_ij must be (A,M,3)")
    m = int(offsets.shape[1])
    if vj.shape != (n_active, m, 3):
        raise ValueError("v_j must be (A,M,3)")
    if fz.shape != (n_active, m) or mask.shape != (n_active, m):
        raise ValueError("frozen_j/neighbor_mask must be (A,M)")

    velocities = np.zeros((n_active, 3), dtype=np.float64)
    exhausted = np.zeros(n_active, dtype=bool)
    tested = np.zeros(n_active, dtype=np.int32)
    overlap = np.zeros(n_active, dtype=np.int32)
    violation = np.zeros(n_active, dtype=np.float64)
    pref_speed = np.sqrt(geometry.sq_norm(vp))
    fb_ttc = np.full(n_active, np.inf, dtype=np.float64)

    if m > 0 and n_active > 0:
        dist2 = geometry.sq_norm(offsets)
        overlap = np.sum((dist2 < float(d_plan) ** 2) & mask, axis=1).astype(np.int32)

    # 逐机复用同一组 (C,M) 暂存区：避免重复分配，保持缓存热度。
    buf = _ScanBuffers(candidate_set.n_template_candidates(), max(m, 1))
    for a in range(n_active):
        sel = mask[a]
        velocity, n_tested, used_fallback, min_ttc = _search_one(
            candidate_set,
            model=model,
            v_max=float(v_max),
            tau=float(tau),
            d_plan=float(d_plan),
            v_current=vc[a],
            v_pref=vp[a],
            p_ij=offsets[a][sel],
            v_j=vj[a][sel],
            frozen_j=fz[a][sel],
            buf=buf,
        )
        velocities[a] = velocity
        tested[a] = np.int32(n_tested)
        exhausted[a] = bool(used_fallback)
        fb_ttc[a] = min_ttc

    # 诊断：返回速度对全部有效邻机的最大约束残差（批量向量化）。
    violation = _constraint_violation_batch(
        velocities,
        model=model,
        v_current=vc,
        p_ij=offsets,
        v_j=vj,
        frozen_j=fz,
        neighbor_mask=mask,
        tau=float(tau),
        d_plan=float(d_plan),
    )

    return SearchResult(
        velocities=velocities,
        sample_exhausted=exhausted,
        used_fallback=exhausted.copy(),
        candidate_count_tested=tested,
        initial_overlap_pairs=overlap,
        constraint_violation_max=violation,
        preferred_speed=pref_speed,
        fallback_min_ttc=fb_ttc,
    )
