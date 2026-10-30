"""pytest 公共夹具与开关。OWNER: lead"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--bench",
        action="store_true",
        default=False,
        help="run performance benchmark tests marked with @pytest.mark.bench",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--bench"):
        return
    skip_bench = pytest.mark.skip(reason="performance benchmark; enable with --bench")
    for item in items:
        if "bench" in item.keywords:
            item.add_marker(skip_bench)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def default_cfg():
    from uavbench.config import load_config

    return load_config(ROOT / "configs" / "default.yaml")


@pytest.fixture()
def rng():
    return np.random.default_rng(12345)


@pytest.fixture()
def tiny_scene():
    """4 机严格对称小场景（正四面体），用于解析测试。"""
    from uavbench.initialization import build_scene_from_directions

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
    return build_scene_from_directions(dirs, sphere_radius=20.0, scene_id="tetra4", seed=-1)
