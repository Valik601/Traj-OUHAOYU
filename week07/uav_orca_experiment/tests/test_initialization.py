# OWNER: lead
"""初始化与场景持久化验收（handoff §3、§8 初始化测试）。"""

from __future__ import annotations

import numpy as np
import pytest

from uavbench import geometry
from uavbench.config import load_config
from uavbench.initialization import (
    InitializationError,
    build_base_fibonacci_scene,
    build_scene,
    build_scene_from_directions,
    fibonacci_sphere,
    generate_scenes,
    load_scene,
    read_manifest,
    save_scene,
)


def test_fibonacci_formula_matches_handoff(default_cfg):
    n, radius = 100, 20.0
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    u = np.stack(
        [np.sqrt(1.0 - z * z) * np.cos(phi), np.sqrt(1.0 - z * z) * np.sin(phi), z], axis=-1
    )
    expected = radius * u
    got = fibonacci_sphere(n, radius)
    assert got.shape == (n, 3)
    assert np.array_equal(got, expected)


def test_base_scene_properties(default_cfg):
    scene = build_base_fibonacci_scene(default_cfg)
    assert scene.n_agents == default_cfg.simulation.n_agents == 100
    assert scene.seed == -1
    norms = np.linalg.norm(scene.starts, axis=1)
    assert np.allclose(norms, default_cfg.simulation.init_sphere_radius, rtol=0.0, atol=1e-12)
    assert np.array_equal(scene.goals, -scene.starts)
    assert np.array_equal(scene.agent_ids, np.arange(100))
    assert scene.meta["min_start_separation"] >= default_cfg.simulation.d_safe
    assert scene.meta["min_goal_separation"] >= default_cfg.simulation.d_safe
    # 起点—他机终点交叉邻接必须记录（不能悄悄删除场景）
    assert "min_start_other_goal" in scene.meta
    assert "cross_pairs_below_safety" in scene.meta


@pytest.mark.parametrize("seed", [0, 3, 1000, 1019])
def test_perturbed_scene_bounded_and_safe(default_cfg, seed):
    scene = build_scene(default_cfg, seed)
    base = geometry.fibonacci_sphere_directions(default_cfg.simulation.n_agents)
    dirs = scene.starts / default_cfg.simulation.init_sphere_radius
    cosang = np.clip(np.sum(dirs * base, axis=1), -1.0, 1.0)
    angle = np.arccos(cosang)
    assert float(np.max(angle)) <= np.deg2rad(2.0) + 1e-12
    assert np.array_equal(scene.goals, -scene.starts)
    assert scene.meta["min_start_separation"] >= default_cfg.simulation.d_safe - 1e-12
    assert scene.meta["min_goal_separation"] >= default_cfg.simulation.d_safe - 1e-12
    assert scene.meta["resample_count"] >= 0
    assert scene.meta["goals_are_negated_starts"] is True


def test_scene_generation_deterministic_and_seed_sensitive(default_cfg):
    a = build_scene(default_cfg, 1000)
    b = build_scene(default_cfg, 1000)
    c = build_scene(default_cfg, 1001)
    assert np.array_equal(a.starts, b.starts)
    assert not np.array_equal(a.starts, c.starts)
    # 基础球面不依赖 seed
    base = geometry.fibonacci_sphere_directions(100)
    assert np.allclose(a.starts / 20.0, base, rtol=0.0, atol=np.deg2rad(2.0))


def test_explicit_directions_scene_accepts_tetrahedron(default_cfg):
    dirs = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
    )
    scene = build_scene_from_directions(dirs, 20.0, "tetra4", d_safe=1.0)
    assert scene.n_agents == 4
    assert np.array_equal(scene.goals, -scene.starts)
    assert scene.meta["min_start_separation"] > 1.0


def test_explicit_directions_reject_unsafe_scene():
    dirs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    with pytest.raises(InitializationError):
        build_scene_from_directions(dirs, 20.0, "bad", d_safe=1.0)


def test_save_load_roundtrip_and_manifest(default_cfg, tmp_path):
    scene = save_scene(build_scene(default_cfg, 1000), tmp_path)
    assert scene.path is not None and scene.path.exists()
    assert len(scene.sha256) == 64
    manifest = read_manifest(tmp_path)
    assert manifest["scenes"]["1000"]["sha256"] == scene.sha256
    loaded = load_scene(scene.path)
    assert loaded.scene_id == "1000"
    assert loaded.seed == 1000
    assert np.array_equal(loaded.starts, scene.starts)
    assert np.array_equal(loaded.goals, scene.goals)
    assert loaded.sha256 == scene.sha256
    assert loaded.meta["mode"] == "tangential_gaussian"


def test_generate_scenes_full_profile_and_reuse(default_cfg, tmp_path):
    cfg = load_config()
    scenes = generate_scenes(cfg, "full", tmp_path)
    assert len(scenes) == 21  # base + 20 formal
    assert scenes[0].scene_id == "base_fibonacci"
    hashes = [s.sha256 for s in scenes]
    # 再次生成必须复用同一 hash（不静默重采样）
    scenes2 = generate_scenes(cfg, "full", tmp_path)
    assert [s.sha256 for s in scenes2] == hashes
    assert len(read_manifest(tmp_path)["scenes"]) == 21


def test_generate_scenes_smoke_profile(default_cfg, tmp_path):
    cfg = load_config("configs/smoke.yaml")
    scenes = generate_scenes(cfg, "smoke", tmp_path)
    assert [s.scene_id for s in scenes] == ["base_fibonacci", "0", "1"]
