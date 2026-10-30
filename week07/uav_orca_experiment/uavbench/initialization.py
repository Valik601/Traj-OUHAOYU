"""场景初始化与持久化（handoff §3）。

- 确定性基础场景：固定 Fibonacci 球面（N=100, R=20 m），起点在球面、目标严格取负。
- 正式场景：切向角度扰动 ``theta ~ U(-2°, 2°)``，按 §3.2 的重采样规则构造。
- 所有模型从同一 ``scene_*.npz`` 读取；保存 SHA-256 到 ``scenes/manifest.json``。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import geometry
from .config import Config
from .rng import STREAM_SCENE, make_rng

SCENE_SCHEMA_VERSION = 1


class InitializationError(RuntimeError):
    """场景无法满足安全初始条件（不得悄悄降低安全距离）。"""


@dataclass
class Scene:
    """一个已持久化的场景。"""

    scene_id: str
    seed: int
    starts: np.ndarray
    goals: np.ndarray
    agent_ids: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""
    path: Path | None = None

    @property
    def n_agents(self) -> int:
        return int(self.starts.shape[0])


def fibonacci_sphere(n: int, radius: float, *, offset: float = 0.5) -> np.ndarray:
    """handoff §3.1 的确定性 Fibonacci 球面：``p_i = R * u_i``。

    ``z_i = 1 - 2*(i+offset)/N``，``phi_i = i*pi*(3-sqrt(5))``。
    """
    dirs = geometry.fibonacci_sphere_directions(n, offset=offset)
    return float(radius) * dirs


def min_pairwise_distance(points: np.ndarray) -> float:
    """返回全部无序机对的最小欧氏距离（含自身=inf 由三角不等式自然排除）。"""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    best = np.inf
    for i in range(n - 1):
        d = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
        if d.size:
            m = float(np.min(d))
            if m < best:
                best = m
    return best


def _cross_min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """``min_{i != j} ||a_i - b_j||``（用于起点—他机终点交叉邻接诊断）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    best = np.inf
    for i in range(n):
        d = np.linalg.norm(b - a[i][None, :], axis=1)
        d[i] = np.inf  # i == j 不计
        m = float(np.min(d))
        if m < best:
            best = m
    return best


def _check_scene_geometry(
    starts: np.ndarray, goals: np.ndarray, d_safe: float, *, eps: float = geometry.GEOM_EPS
) -> dict[str, Any]:
    n = starts.shape[0]
    min_start = min_pairwise_distance(starts)
    min_goal = min_pairwise_distance(goals)
    min_cross = _cross_min_distance(starts, goals)
    below = 0
    for i in range(n):
        d = np.linalg.norm(starts - goals[i][None, :], axis=1)
        d[i] = np.inf
        below += int(np.count_nonzero(d < d_safe - eps))
    return {
        "min_start_separation": float(min_start),
        "min_goal_separation": float(min_goal),
        "min_start_other_goal": float(min_cross),
        "cross_pairs_below_safety": int(below),
        "min_start_separation_ok": bool(min_start >= d_safe - eps),
        "min_goal_separation_ok": bool(min_goal >= d_safe - eps),
    }


def build_scene_from_directions(
    dirs: np.ndarray,
    sphere_radius: float,
    scene_id: str,
    seed: int = -1,
    meta: dict[str, Any] | None = None,
    d_safe: float = 1.0,
) -> Scene:
    """由单位方向构造严格对称/自定义小场景。"""
    d = np.asarray(dirs, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 3:
        raise InitializationError(f"dirs must have shape (N,3), got {d.shape}")
    norms = np.linalg.norm(d, axis=1)
    if np.any(norms <= 0):
        raise InitializationError("zero direction")
    d = d / norms[:, None]
    starts = float(sphere_radius) * d
    goals = -starts
    checks = _check_scene_geometry(starts, goals, d_safe)
    if not checks["min_start_separation_ok"] or not checks["min_goal_separation_ok"]:
        raise InitializationError(f"scene {scene_id} violates initial safety separation: {checks}")
    meta_out: dict[str, Any] = {"mode": "explicit_directions", "sphere_radius": float(sphere_radius)}
    meta_out.update(checks)
    if meta:
        meta_out.update(meta)
    return Scene(
        scene_id=str(scene_id),
        seed=int(seed),
        starts=starts,
        goals=goals,
        agent_ids=np.arange(starts.shape[0], dtype=np.int64),
        meta=meta_out,
    )


def build_base_fibonacci_scene(cfg: Config) -> Scene:
    """确定性基础场景（未扰动 Fibonacci 球面，handoff §3.1）。"""
    sim = cfg.simulation
    starts = fibonacci_sphere(sim.n_agents, sim.init_sphere_radius)
    return build_scene_from_directions(
        starts / sim.init_sphere_radius,
        sim.init_sphere_radius,
        scene_id="base_fibonacci",
        seed=-1,
        meta={"mode": "fibonacci_sphere_base", "perturbation": "none"},
        d_safe=sim.d_safe,
    )


def _sample_perturbed_directions(
    cfg: Config, rng: np.random.Generator, base_dirs: np.ndarray
) -> np.ndarray:
    """切向高斯 + 角度扰动（handoff §3.2 步骤 1—3）。"""
    sim = cfg.simulation
    theta = rng.uniform(
        -np.deg2rad(sim.perturbation.max_angle_deg),
        np.deg2rad(sim.perturbation.max_angle_deg),
        size=base_dirs.shape[0],
    )
    tangent = geometry.tangential_unit_vectors(base_dirs, rng)
    new_dirs = np.cos(theta)[:, None] * base_dirs + np.sin(theta)[:, None] * tangent
    norms = np.linalg.norm(new_dirs, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise InitializationError("perturbation produced degenerate direction")
    return new_dirs / norms[:, None]


def build_scene(cfg: Config, seed: int | None = None, scene_id: str | None = None) -> Scene:
    """构造一个正式场景；``seed=None`` 返回确定性基础场景。"""
    sim = cfg.simulation
    if seed is None:
        return build_base_fibonacci_scene(cfg)
    if sim.n_agents < 2:
        raise InitializationError("n_agents must be >= 2")
    base_dirs = geometry.fibonacci_sphere_directions(sim.n_agents)
    rng = make_rng(int(seed), STREAM_SCENE)
    last_checks: dict[str, Any] = {}
    max_tries = int(sim.perturbation.resample_max)
    for attempt in range(1, max_tries + 1):
        if sim.perturbation.mode == "none":
            dirs = base_dirs.copy()
        elif sim.perturbation.mode == "tangential_gaussian":
            dirs = _sample_perturbed_directions(cfg, rng, base_dirs)
        else:  # pragma: no cover - 配置校验已拦截
            raise InitializationError(f"unknown perturbation mode: {sim.perturbation.mode}")
        starts = float(sim.init_sphere_radius) * dirs
        goals = -starts
        checks = _check_scene_geometry(starts, goals, sim.d_safe)
        last_checks = checks
        if checks["min_start_separation_ok"] and checks["min_goal_separation_ok"]:
            meta = {
                "mode": sim.perturbation.mode,
                "max_angle_deg": float(sim.perturbation.max_angle_deg),
                "sphere_radius": float(sim.init_sphere_radius),
                "resample_count": int(attempt - 1),
                **checks,
                "goals_are_negated_starts": True,
            }
            return Scene(
                scene_id=str(scene_id) if scene_id is not None else str(int(seed)),
                seed=int(seed),
                starts=starts,
                goals=goals,
                agent_ids=np.arange(sim.n_agents, dtype=np.int64),
                meta=meta,
            )
    raise InitializationError(
        f"scene seed={seed} failed safety separation after {max_tries} resamples: {last_checks}"
    )


def _scene_arrays(scene: Scene) -> dict[str, np.ndarray]:
    meta_json = json.dumps(scene.meta, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return {
        "schema_version": np.asarray(SCENE_SCHEMA_VERSION, dtype=np.int64),
        "scene_id": np.asarray(scene.scene_id),
        "seed": np.asarray(int(scene.seed), dtype=np.int64),
        "starts": np.ascontiguousarray(scene.starts, dtype=np.float64),
        "goals": np.ascontiguousarray(scene.goals, dtype=np.float64),
        "agent_ids": np.ascontiguousarray(scene.agent_ids, dtype=np.int64),
        "meta_json": np.asarray(meta_json),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_path(scenes_dir: Path, scene_id: str) -> Path:
    return Path(scenes_dir) / f"scene_{scene_id}.npz"


def manifest_path(scenes_dir: Path) -> Path:
    return Path(scenes_dir) / "manifest.json"


def read_manifest(scenes_dir: Path) -> dict[str, Any]:
    p = manifest_path(scenes_dir)
    if not p.exists():
        return {"schema_version": SCENE_SCHEMA_VERSION, "scenes": {}}
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("scenes", {})
    return data


def write_manifest(scenes_dir: Path, manifest: dict[str, Any]) -> None:
    p = manifest_path(scenes_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, allow_nan=False)
    tmp.replace(p)


def save_scene(scene: Scene, scenes_dir: Path) -> Scene:
    """保存场景（npz + manifest 条目），返回带 sha256/path 的副本。"""
    scenes_dir = Path(scenes_dir)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    path = scene_path(scenes_dir, scene.scene_id)
    np.savez(path, **_scene_arrays(scene))
    sha = file_sha256(path)
    scene.sha256 = sha
    scene.path = path
    manifest = read_manifest(scenes_dir)
    manifest["scenes"][scene.scene_id] = {
        "scene_id": scene.scene_id,
        "seed": int(scene.seed),
        "sha256": sha,
        "scene_file": path.name,
        "n_agents": scene.n_agents,
        "min_start_separation": scene.meta.get("min_start_separation"),
        "min_goal_separation": scene.meta.get("min_goal_separation"),
        "min_start_other_goal": scene.meta.get("min_start_other_goal"),
        "cross_pairs_below_safety": scene.meta.get("cross_pairs_below_safety"),
        "resample_count": scene.meta.get("resample_count"),
        "mode": scene.meta.get("mode"),
    }
    write_manifest(scenes_dir, manifest)
    return scene


def load_scene(path: Path, scenes_dir: Path | None = None) -> Scene:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        schema = int(data["schema_version"])
        if schema != SCENE_SCHEMA_VERSION:
            raise InitializationError(f"unsupported scene schema {schema} in {path}")
        scene = Scene(
            scene_id=str(data["scene_id"]),
            seed=int(data["seed"]),
            starts=np.array(data["starts"], dtype=np.float64),
            goals=np.array(data["goals"], dtype=np.float64),
            agent_ids=np.array(data["agent_ids"], dtype=np.int64),
            meta=json.loads(str(data["meta_json"])),
        )
    scene.sha256 = file_sha256(path)
    scene.path = path
    if scene.starts.shape != scene.goals.shape or scene.starts.shape[0] != scene.agent_ids.shape[0]:
        raise InitializationError(f"corrupt scene arrays in {path}")
    return scene


def generate_scenes(
    cfg: Config, profile: str, scenes_dir: Path, *, include_base: bool = True
) -> list[Scene]:
    """按 profile 生成/复用场景文件；返回场景列表（基础场景在前）。"""
    scenes_dir = Path(scenes_dir)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    seeds = profile_seeds(cfg, profile)
    out: list[Scene] = []
    if include_base:
        base = build_base_fibonacci_scene(cfg)
        out.append(_save_or_reuse(base, scenes_dir))
    for seed in seeds:
        scene = build_scene(cfg, int(seed))
        out.append(_save_or_reuse(scene, scenes_dir))
    return out


def _save_or_reuse(scene: Scene, scenes_dir: Path) -> Scene:
    path = scene_path(scenes_dir, scene.scene_id)
    manifest = read_manifest(scenes_dir)
    if path.exists():
        rec = manifest["scenes"].get(scene.scene_id)
        sha = file_sha256(path)
        if rec is not None and rec.get("sha256") == sha:
            try:
                loaded = load_scene(path)
                if np.allclose(loaded.starts, scene.starts, rtol=0.0, atol=0.0) and np.array_equal(
                    loaded.goals, scene.goals
                ):
                    return loaded
            except InitializationError:
                pass
    return save_scene(scene, scenes_dir)


def profile_seeds(cfg: Config, profile: str) -> list[int]:
    """profile → 场景种子列表（不含基础场景）。"""
    sc = cfg.scenes
    if profile in ("full", "formal"):
        return [int(s) for s in sc.formal_seeds]
    if profile == "smoke":
        return [int(s) for s in sc.formal_seeds]
    if profile == "development":
        return [int(s) for s in sc.development_seeds]
    if profile == "sensitivity":
        return [int(s) for s in sc.sensitivity_seeds]
    if profile == "stress":
        return []
    raise ValueError(f"unknown profile: {profile}")


def load_scenes_for_profile(cfg: Config, profile: str, scenes_dir: Path) -> dict[str, Scene]:
    """读取已生成场景（缺失则报错，不静默重生成）。"""
    scenes: dict[str, Scene] = {}
    if profile != "stress":
        paths = [scene_path(scenes_dir, "base_fibonacci")]
        paths += [scene_path(scenes_dir, str(int(s))) for s in profile_seeds(cfg, profile)]
        for p in paths:
            if not p.exists():
                raise InitializationError(f"missing scene file {p}; run generate-scenes first")
            sc = load_scene(p)
            scenes[sc.scene_id] = sc
    else:
        p = scene_path(scenes_dir, "base_fibonacci")
        if not p.exists():
            raise InitializationError(f"missing base scene {p}; run generate-scenes first")
        sc = load_scene(p)
        scenes[sc.scene_id] = sc
    return scenes
