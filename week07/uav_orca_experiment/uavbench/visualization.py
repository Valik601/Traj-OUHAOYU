"""可视化流水线：handoff §10.1 的 9 类必做图 + GIF 动画（CONTRACTS.md §13）。

固定展示案例与"失败/中位案例"选择规则（预先声明，写在代码里，不做事后挑选）
--------------------------------------------------------------------------
- 主展示场景种子固定为 :data:`DISPLAY_SEED` = **1000**（正式 A/B 场景共用同一批种子）。
- 中位/失败案例种子规则（确定性，与最终结果好坏无关）::

      candidates = { 正式实验 experiment_id == "formal" 中全部 scenario == "B" 的 scene；
                     要求 model == "orca_release" 且 metrics.json 的 status.valid 为 true、
                     safe_completion_rate 为有限数值 }
      median_rate = numpy.median([safe_completion_rate(scene) for scene in candidates])
      median_case_seed = argmin_{scene} |safe_completion_rate(scene) - median_rate|
                         （若多个场景与中位数等距，取 **最小 seed**）

  选中 seed 写入 ``out_dir/selection.json``（含候选 rate 表、median_rate 与规则原文）。
  若 ``formal`` 不存在，回退到有序候选实验（formal > smoke > development > 其它按名字排序），
  并在 selection.json 记录回退原因；若没有任何 valid 的 orca_release/B 场景，则跳过中位案例图
  并在 stderr 打印 warning，但其余图照常生成。
- 主展示 seed 与中位 seed 相同时不重复渲染。

公共 API
--------
``render_all(results_dir, out_dir, configs_dir=None) -> list[Path]``
    在完整结果树上渲染全部图（fig01..fig09）、动画与 ``selection.json``；``results_dir`` 是包含
    ``<experiment_id>/<scenario>/<model>/scene_<id>/<config_hash>/`` 的 results 根目录（与 CLI
    ``render-all --results`` 默认值一致）。``configs_dir`` 可选，仅用于在 run 未记录 ``scene_file``
    时回退定位 ``scenes/``（其它参数一律从每个 run 的 config.json 读取）。
``render_scene_seed(scenario, seed, results_dir, out_dir, models=None) -> list[Path]``
    只渲染单个 (scenario, seed) 的场景相关图与动画（CLI ``render`` 使用）。
``render_animation(run_dir, out_path, fps=25, want_mp4=False) -> Path``
    渲染单个 run 的 GIF 动画（最低要求 GIF；``want_mp4=True`` 且找到 ffmpeg 时另存 MP4）。
``interval_min_separation(positions, velocities, dt) -> np.ndarray``
    图 3 使用的逐区间连续最小机间距（全部机对、闭式最近点公式），公开以便单测。

输出文件（全部位于 ``out_dir``）
--------------------------------
``fig01_initial_geometry_{seed|median_seed<N>}.png``
``fig02_trajectories_{seed|median_seed<N>}.png``
``fig03_min_separation_{seed|median_seed<N>}.png``
``fig04_arrival_curves_{seed|median_seed<N>}.png``
``fig05_curriculum_metrics.png`` / ``fig06_safe_completion_paired.png`` /
``fig08_arrival_path_wait.png``（跨场景聚合，不随 seed 变化）
``fig07_center_events_{seed|median_seed<N>}.png``
``fig09_animation_snapshots_{seed|median_seed<N>}.png``
``anim_<orca|orca_release>_B_seed<seed>.gif``（display seed 与 median-case seed 各一套）
``selection.json``（display seed、median-case 规则/候选/选中 seed、run 索引与 warnings）

诚实作图约定（handoff §10.1、§6）
--------------------------------
- 图 3 曲线是**每个控制区间的连续最小值序列**：对全部 N(N-1)/2 机对，用闭式最近点公式
  ``t* = clip(-r·w/|w|², 0, dt)``、``d_min = ||r + t* w||`` 从 ``trajectories.npz`` 的
  位置与区间常速重新计算，**不使用端点距离**；图中注明。
- 图 4/5/6 的 safe-completion 相关量使用**全 120 s 观察窗回溯**的侵犯资格，图中注明它
  不是实时已知量；``cumulative_safe_completions_known_later`` 同样注明。
- 缺失的时间量（如 ``all_arrival_time`` 为 null）不画成 0：图 5 只跳过并标注样本数 n。
- 3D 图等比例（``set_box_aspect((1,1,1))`` + 三轴同范围）。图 1 的球面是 20 m **初始条件
  参考球**，不是机体；动画不画真实半径球体（文字注明机体半径 0.5 m），只画球心标记。
- matplotlib 使用 ``Agg`` 后端；每个 figure 都关闭；不联网；所有写盘只发生在 ``out_dir`` 内。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # noqa: E402  - 必须在 pyplot import 之前（handoff §10.1 环境要求）
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

__all__ = [
    "DISPLAY_SEED",
    "MEDIAN_CASE_RULE",
    "discover_runs",
    "interval_min_separation",
    "render_all",
    "render_animation",
    "render_scene_seed",
]

# ---------------------------------------------------------------------------
# 常量（任务参数；缺失 config.json 时作为回退值）
# ---------------------------------------------------------------------------
DISPLAY_SEED = 1000
D_SAFE_DEFAULT = 1.0  # 2 * r
BODY_RADIUS_DEFAULT = 0.5
CENTER_TOL_DEFAULT = 0.1
INIT_SPHERE_RADIUS_DEFAULT = 20.0
V_MAX_DEFAULT = 2.0
DT_DEFAULT = 0.1
OBSERVATION_TIME_DEFAULT = 120.0

#: 预先声明、写入 selection.json 的中位案例选择规则（见模块 docstring）
MEDIAN_CASE_RULE = (
    "median_case_seed = argmin_seed |safe_completion_rate_B(seed; model=orca_release) - "
    "median_seed(safe_completion_rate_B(orca_release))| over valid formal scenario-B runs; "
    "ties broken by smallest seed. Computed before rendering; see module docstring."
)

#: 场景/模型固定顺序（与 CONTRACTS §3 模型表一致）
MODEL_ORDER = ("direct", "vo", "rvo", "orca", "orca_release", "direct_release")
SCENARIO_MODELS = {
    "A": ("direct", "vo", "rvo", "orca"),
    "B": ("direct", "vo", "rvo", "orca", "orca_release", "direct_release"),
}
MODEL_COLORS = {
    "direct": "#7f7f7f",
    "vo": "#d62728",
    "rvo": "#ff7f0e",
    "orca": "#1f77b4",
    "orca_release": "#2ca02c",
    "direct_release": "#9467bd",
}
SCENARIO_COLORS = {"A": "#1f77b4", "B": "#d62728"}

#: nav 阶段编码（CONTRACTS.md §5 / navigation.py 的冻结常量）
STAGE_WAITING = 0
STAGE_TO_CENTER = 1
STAGE_TO_GOAL = 2
STAGE_ARRIVED = 3
STAGE_COLORS = {
    STAGE_WAITING: "#b3b3b3",
    STAGE_TO_CENTER: "#ff8c00",
    STAGE_TO_GOAL: "#1f77b4",
    STAGE_ARRIVED: "#2ca02c",
}
STAGE_LABELS = {
    STAGE_WAITING: "waiting",
    STAGE_TO_CENTER: "to_center",
    STAGE_TO_GOAL: "to_goal",
    STAGE_ARRIVED: "arrived",
}

#: 动画固定参数
ANIM_MAX_FRAMES = 150
ANIM_DPI = 80
ANIM_ZOOM_RADIUS = 2.5  # m，中心放大视角
ANIM_LABEL_IDS = (0, 11, 22, 33, 44, 55, 66, 77)  # 最多 8 个固定 ID 标签
ANIM_FPS = 25

#: 场景级 bootstrap（与 statistics.py 的固定统计种子保持一致，用于 CSV 缺失时的本地回退）
BOOTSTRAP_SEED = 20261030
BOOTSTRAP_RESAMPLES = 10000

#: 默认绘图参数
FIG_DPI = 150
_FIG2_MODELS = ("vo", "rvo", "orca")
_FIG3_MODELS = {"A": ("vo", "rvo", "orca"), "B": ("vo", "rvo", "orca", "orca_release")}
_FIG4_MODELS = {"A": ("vo", "rvo", "orca"), "B": ("vo", "rvo", "orca", "orca_release")}
_FIG7_MODELS = ("vo", "orca", "orca_release")
_ANIM_MODELS = ("orca", "orca_release")


# ---------------------------------------------------------------------------
# 结果目录发现
# ---------------------------------------------------------------------------
@dataclass
class RunInfo:
    """一次已落盘运行的最小元信息（按 CONTRACTS §12 目录结构发现）。"""

    run_dir: Path
    experiment_id: str
    scenario: str
    model: str
    scene_id: str
    seed: int
    config_hash: str
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        st = self.metrics.get("status") or {}
        if st.get("numerical_error"):
            return "numerical_error"
        if st.get("completed"):
            return "completed"
        return "timeout"

    @property
    def valid(self) -> bool:
        st = self.metrics.get("status") or {}
        return bool(st.get("valid", True))

    @property
    def completed(self) -> bool:
        st = self.metrics.get("status") or {}
        return bool(st.get("completed", False))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.experiment_id, self.scenario, self.model, self.scene_id)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_csv(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _read_npz(path: Path, only: Iterable[str] | None = None) -> dict[str, np.ndarray] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = data.files if only is None else [key for key in data.files if key in set(only)]
            return {key: np.array(data[key]) for key in keys}
    except (OSError, ValueError):
        return None


#: 可视化需要读取的 trajectories.npz 键（跳过体积大的 diag_* 数组，避免缓存膨胀）
_TRAJ_KEYS = (
    "t",
    "positions",
    "velocities",
    "stages",
    "frozen",
    "active",
    "center_entry_times",
    "center_entry_steps",
    "arrival_times",
    "release_planned",
    "release_actual",
    "planning_step_seconds",
    "dt",
)


def _path_component(path: Path, index_from_end: int) -> str:
    parts = Path(path).parts
    try:
        return parts[-index_from_end]
    except IndexError:
        return ""


def _iter_candidate_run_dirs(results_dir: Path) -> Iterable[Path]:
    """遍历 ``results/<exp>/<scenario>/<model>/scene_<id>/<config_hash>`` 叶子目录。"""
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for scenario_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            for model_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
                for scene_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                    if not scene_dir.name.startswith("scene_"):
                        continue
                    for hash_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
                        yield hash_dir


def _run_preference(run: RunInfo) -> tuple[int, str]:
    """去重偏好：valid 优先，其次 config_hash 字典序最小（确定性）。"""
    return (0 if run.valid else 1, run.config_hash)


def discover_runs(results_dir: Path) -> list[RunInfo]:
    """发现所有含 ``metrics.json`` 的运行目录；同一 (exp, scenario, model, scene) 只保留一个。"""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    by_key: dict[tuple[str, str, str, str], RunInfo] = {}
    for run_dir in _iter_candidate_run_dirs(results_dir):
        metrics = _read_json(run_dir / "metrics.json")
        if metrics is None:
            continue
        prov = _read_json(run_dir / "provenance.json") or {}
        exp = str(prov.get("experiment_id") or metrics.get("experiment_id") or _path_component(run_dir, 4))
        scenario = str(prov.get("scenario") or metrics.get("scenario") or _path_component(run_dir, 3))
        model = str(prov.get("model") or metrics.get("model") or _path_component(run_dir, 2))
        scene_id = str(
            prov.get("scene_id") or metrics.get("scene_id") or _path_component(run_dir, 1).removeprefix("scene_")
        )
        seed_raw = prov.get("seed", metrics.get("seed"))
        try:
            seed = int(seed_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            seed = _seed_from_scene_id(scene_id)
        config_hash = str(metrics.get("config_hash") or prov.get("config_hash") or run_dir.name)
        run = RunInfo(
            run_dir=run_dir,
            experiment_id=exp,
            scenario=scenario,
            model=model,
            scene_id=scene_id,
            seed=seed,
            config_hash=config_hash,
            metrics=metrics,
            provenance=prov,
        )
        prev = by_key.get(run.key)
        if prev is None or _run_preference(run) < _run_preference(prev):
            by_key[run.key] = run
    return sorted(
        by_key.values(),
        key=lambda r: (r.experiment_id, r.scenario, r.model, r.seed, r.scene_id),
    )


def _seed_from_scene_id(scene_id: str) -> int:
    try:
        return int(str(scene_id))
    except (TypeError, ValueError):
        return -1


def _ordered_experiments(experiments: Iterable[str]) -> list[str]:
    exps = sorted(set(str(e) for e in experiments))
    preferred = [e for e in ("formal", "smoke", "development") if e in exps]
    return preferred + [e for e in exps if e not in preferred]


def _primary_experiment(runs: Sequence[RunInfo]) -> str | None:
    ordered = _ordered_experiments({r.experiment_id for r in runs})
    return ordered[0] if ordered else None


# ---------------------------------------------------------------------------
# 渲染上下文与惰性资产读取
# ---------------------------------------------------------------------------
@dataclass
class _Context:
    results_dir: Path
    out_dir: Path
    configs_dir: Path | None
    all_runs: list[RunInfo]
    runs: list[RunInfo]  # 主实验的 runs
    experiment_id: str
    selection: dict[str, Any] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    warned: set[str] = field(default_factory=set)


def _warn(message_or_ctx: Any, message: str | None = None) -> None:
    """向 stderr 打印 warning；若传入 context 则同时记录到 selection.json。"""
    if message is None:
        text = str(message_or_ctx)
        ctx = None
    else:
        ctx = message_or_ctx
        text = str(message)
    print(f"[visualization] WARNING: {text}", file=sys.stderr)
    if ctx is not None:
        ctx.warnings.append(text)


def _warn_once(ctx: _Context, key: str, message: str) -> None:
    if key in ctx.warned:
        return
    ctx.warned.add(key)
    _warn(ctx, message)


_LOADERS = {
    "traj": lambda d: _read_npz(d / "trajectories.npz", only=_TRAJ_KEYS),
    "agents": lambda d: _read_csv(d / "agents.csv"),
    "center_events": lambda d: _read_csv(d / "center_events.csv"),
    "arrival_curve": lambda d: _read_csv(d / "arrival_curve.csv"),
    "min_distance_curve": lambda d: _read_csv(d / "min_distance_curve.csv"),
    "violation_events": lambda d: _read_csv(d / "violation_events.csv"),
    "config": lambda d: _read_json(d / "config.json"),
}


def _asset(ctx: _Context, run: RunInfo, name: str) -> Any:
    """惰性读取 run 资产（缺失返回 None，不抛异常）。"""
    if name not in _LOADERS:
        raise KeyError(f"unknown asset: {name}")
    bucket = ctx.assets.get(str(run.run_dir))
    if bucket is None:
        bucket = {}
        ctx.assets[str(run.run_dir)] = bucket
    if name not in bucket:
        try:
            bucket[name] = _LOADERS[name](run.run_dir)
        except (OSError, ValueError, KeyError):
            bucket[name] = None
    return bucket[name]


def _find_run(ctx: _Context, scenario: str, model: str, seed: int) -> RunInfo | None:
    """按 (scenario, model, seed) 查找 run；主实验优先，其次其它实验（带 warning）。"""
    for run in ctx.runs:
        if run.scenario == scenario and run.model == model and run.seed == int(seed):
            return run
    for run in ctx.all_runs:
        if run.scenario == scenario and run.model == model and run.seed == int(seed):
            _warn_once(
                ctx,
                f"fallback-exp:{scenario}:{model}:{seed}",
                f"run {scenario}/{model}/scene_{run.scene_id} not in experiment "
                f"'{ctx.experiment_id}'; using experiment '{run.experiment_id}'",
            )
            return run
    return None


def _runs_for_seed(ctx: _Context, seed: int) -> list[RunInfo]:
    matches = [r for r in ctx.runs if r.seed == int(seed)]
    if matches:
        return matches
    matches = [r for r in ctx.all_runs if r.seed == int(seed)]
    if matches:
        _warn_once(ctx, f"fallback-exp:seed{seed}", f"seed {seed} not in main experiment; using other experiment")
    return matches


def _models_for_scenario(ctx: _Context, scenario: str, seed: int | None = None) -> list[str]:
    """返回该 scenario 中有 run 的模型（固定顺序）；seed 为 None 时看全部 scene。"""
    present = set()
    pool = ctx.runs if seed is None else _runs_for_seed(ctx, seed)
    for run in pool:
        if run.scenario == scenario:
            present.add(run.model)
    return [m for m in SCENARIO_MODELS[scenario] if m in present]


# ---------------------------------------------------------------------------
# 轨迹/指标派生工具
# ---------------------------------------------------------------------------
def interval_min_separation(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
) -> np.ndarray:
    """逐控制区间的**连续**最小机间距序列（全部机对，闭式最近点公式）。

    对每个区间 ``[t_k, t_k+dt]``：相对位置 ``r = p_i - p_j``，相对常速度 ``w = v_i - v_j``，
    ``t* = clip(-r·w / |w|², 0, dt)``，``d_min = ||r + t* w||``；返回每个区间全部机对的最小值。
    这是 handoff §6.1 的区间最小距离定义，**不是端点距离**。

    Parameters
    ----------
    positions : ndarray, shape (K+1, N, 3)
    velocities : ndarray, shape (K, N, 3)  # 区间常速命令速度
    dt : float

    Returns
    -------
    ndarray, shape (K,)  # 第 k 个值对应区间 [t_k, t_k + dt] 的连续最小机间距
    """
    pos = np.asarray(positions, dtype=np.float64)
    vel = np.asarray(velocities, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"positions must have shape (K+1,N,3), got {pos.shape}")
    if vel.ndim != 3 or vel.shape[2] != 3:
        raise ValueError(f"velocities must have shape (K,N,3), got {vel.shape}")
    if vel.shape[0] != pos.shape[0] - 1:
        raise ValueError(
            f"velocities must have K = len(positions)-1 intervals, got {vel.shape[0]} vs {pos.shape[0] - 1}"
        )
    if vel.shape[1] != pos.shape[1]:
        raise ValueError(f"positions and velocities disagree on N: {pos.shape} vs {vel.shape}")
    n_pairs_ok = pos.shape[1] >= 2
    out = np.full(vel.shape[0], np.inf, dtype=np.float64)
    if not n_pairs_ok:
        return out
    for k in range(vel.shape[0]):
        r = pos[k][:, None, :] - pos[k][None, :, :]  # (N,N,3)
        w = vel[k][:, None, :] - vel[k][None, :, :]  # (N,N,3)
        ww = np.einsum("...i,...i->...", w, w)
        rw = np.einsum("...i,...i->...", r, w)
        positive = ww > 0.0
        denom = np.where(positive, ww, 1.0)
        t_star = np.where(positive, np.clip(-rw / denom, 0.0, float(dt)), 0.0)
        closest = r + t_star[..., None] * w
        dist = np.sqrt(np.einsum("...i,...i->...", closest, closest))
        np.fill_diagonal(dist, np.inf)
        out[k] = float(np.min(dist))
    return out


def _dt_of_traj(traj: dict[str, np.ndarray]) -> float:
    t = np.asarray(traj.get("t", np.zeros(0)), dtype=np.float64).ravel()
    if t.size >= 2:
        diffs = np.diff(t)
        dt = float(np.median(diffs))
        if np.isfinite(dt) and dt > 0:
            return dt
    if "dt" in traj:
        try:
            dt = float(np.asarray(traj["dt"]).ravel()[0])
            if np.isfinite(dt) and dt > 0:
                return dt
        except (IndexError, ValueError, TypeError):
            pass
    return DT_DEFAULT


def _traj_derived_velocities(traj: dict[str, np.ndarray], dt: float) -> np.ndarray | None:
    pos = traj.get("positions")
    if pos is None:
        return None
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[0] < 2:
        return None
    return np.diff(pos, axis=0) / float(dt)


def _traj_velocities(traj: dict[str, np.ndarray], dt: float) -> np.ndarray | None:
    vel = traj.get("velocities")
    if vel is not None:
        vel = np.asarray(vel, dtype=np.float64)
        if vel.ndim == 3 and vel.shape[0] == np.asarray(traj.get("positions", np.zeros((0,)))).shape[0] - 1:
            return vel
    # 兼容缺 velocities 的轨迹：由相邻位置差分得到区间常速（对恒速段是精确的）
    return _traj_derived_velocities(traj, dt)


def _sim_params(ctx: _Context, run: RunInfo | None) -> dict[str, float]:
    cfg = _asset(ctx, run, "config") if run is not None else None
    sim = (cfg or {}).get("simulation") or {}
    exp = (cfg or {}).get("experiment") or {}
    params = {
        "n_agents": float(sim.get("n_agents", np.nan)),
        "radius": float(sim.get("radius", BODY_RADIUS_DEFAULT)),
        "d_safe": float(sim.get("d_safe", D_SAFE_DEFAULT)),
        "center_tolerance": float(sim.get("center_tolerance", CENTER_TOL_DEFAULT)),
        "init_sphere_radius": float(sim.get("init_sphere_radius", INIT_SPHERE_RADIUS_DEFAULT)),
        "v_max": float(sim.get("v_max", V_MAX_DEFAULT)),
        "dt": float(exp.get("dt", DT_DEFAULT)),
        "observation_time": float(exp.get("observation_time", OBSERVATION_TIME_DEFAULT)),
    }
    return params


# ---------------------------------------------------------------------------
# 场景文件加载（不依赖其它模块内部实现；只读 scenes/scene_*.npz）
# ---------------------------------------------------------------------------
def _scene_candidates(ctx: _Context, run: RunInfo | None, scene_id: str) -> list[Path]:
    candidates: list[Path] = []
    if run is not None:
        scene_file = str(run.provenance.get("scene_file") or "")
        if scene_file:
            raw = Path(scene_file)
            if raw.is_absolute():
                candidates.append(raw)
            else:
                for root in (ctx.results_dir.parent, Path.cwd(), Path(__file__).resolve().parent.parent):
                    candidates.append(Path(root) / raw)
    roots = [ctx.results_dir.parent, Path.cwd(), Path(__file__).resolve().parent.parent]
    if ctx.configs_dir is not None:
        # configs_dir 是项目根的 configs/ 子目录：其父目录即项目根，可用于定位 scenes/
        roots.insert(0, Path(ctx.configs_dir).resolve().parent)
    for root in roots:
        candidates.append(root / "scenes" / f"scene_{scene_id}.npz")
    return candidates


def _load_scene_arrays(path: Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            starts = np.asarray(data["starts"], dtype=np.float64)
            goals = np.asarray(data["goals"], dtype=np.float64)
            scene_id = str(data["scene_id"]) if "scene_id" in data.files else Path(path).stem.removeprefix("scene_")
            seed = int(np.asarray(data["seed"]).ravel()[0]) if "seed" in data.files else -1
            meta: dict[str, Any] = {}
            if "meta_json" in data.files:
                try:
                    loaded = json.loads(str(data["meta_json"]))
                    if isinstance(loaded, dict):
                        meta = loaded
                except ValueError:
                    meta = {}
    except (OSError, ValueError, KeyError):
        return None
    if starts.ndim != 2 or starts.shape[1] != 3 or starts.shape != goals.shape:
        return None
    return {"scene_id": scene_id, "seed": seed, "starts": starts, "goals": goals, "meta": meta}


def _scene_for_seed(ctx: _Context, seed: int) -> tuple[dict[str, Any], Path] | None:
    runs = _runs_for_seed(ctx, seed)
    if not runs:
        return None
    run = runs[0]
    for candidate in _scene_candidates(ctx, run, run.scene_id):
        if candidate.exists():
            scene = _load_scene_arrays(candidate)
            if scene is not None:
                return scene, candidate
    _warn_once(ctx, f"scene:{seed}", f"scene file for seed {seed} not found; scene-dependent panels skipped")
    return None


# ---------------------------------------------------------------------------
# 小工具：bootstrap、布尔转换、数字列
# ---------------------------------------------------------------------------
def _as_bool_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0.0
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _numeric(df: pd.DataFrame | None, column: str) -> np.ndarray:
    if df is None or column not in df.columns:
        return np.zeros(0, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)


def _finite(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.isfinite(np.asarray(values, dtype=np.float64))]


def _bootstrap_mean_ci(values: Iterable[float]) -> tuple[float, float, float, int]:
    arr = _finite(np.asarray(list(values), dtype=np.float64))
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    if n == 1:
        return (mean, mean, mean, 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    boots = arr[idx].mean(axis=1)
    return (mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n)


def _bootstrap_diff_ci(diffs: Iterable[float]) -> tuple[float, float, float, int]:
    arr = _finite(np.asarray(list(diffs), dtype=np.float64))
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    mean = float(np.mean(arr))
    if n == 1:
        return (mean, mean, mean, 1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    boots = arr[idx].mean(axis=1)
    return (mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n)


def _jitter(count: int, width: float = 0.14) -> np.ndarray:
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    return np.linspace(-width, width, count)


# ---------------------------------------------------------------------------
# 绘图小工具
# ---------------------------------------------------------------------------
def _no_data_axis(ax: Any, message: str = "no data") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="#666666", transform=ax.transAxes)


def _model_color(model: str) -> str:
    return MODEL_COLORS.get(str(model), "#444444")


def _set_equal_3d(ax: Any, lo: float, hi: float) -> None:
    """三轴同范围 + 1:1:1 box aspect（等比例）。"""
    try:
        ax.set_xlim3d(lo, hi)
        ax.set_ylim3d(lo, hi)
        ax.set_zlim3d(lo, hi)
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except (AttributeError, ValueError):
        pass


def _set_equal_2d(ax: Any) -> None:
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    span = max(xhi - xlo, yhi - ylo)
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
    ax.set_ylim(cy - span / 2.0, cy + span / 2.0)
    try:
        ax.set_aspect("equal", adjustable="box")
    except (AttributeError, ValueError):
        pass


def _points_absmax(*arrays: np.ndarray) -> float:
    best = 0.0
    for arr in arrays:
        a = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        a = a[np.all(np.isfinite(a), axis=1)]
        if a.size:
            best = max(best, float(np.max(np.abs(a))))
    return best if best > 0 else 1.0


def _stage_label(int_stage: int) -> str:
    return STAGE_LABELS.get(int(int_stage), str(int_stage))


def _save_fig(_ctx: _Context, fig: Any, path: Path, dpi: int = FIG_DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _emit(ctx: _Context, label: str, fn: Any, *args: Any, **kwargs: Any) -> Path | None:
    """执行一个渲染器；异常只打印 warning（缺失数据优雅降级），不阻塞其它图。"""
    try:
        path = fn(ctx, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 单图失败不得阻塞其余图
        _warn(ctx, f"{label} skipped: {type(exc).__name__}: {exc}")
        try:
            plt.close("all")
        except Exception:  # pragma: no cover - 防御性
            pass
        return None
    if path is not None:
        path = Path(path)
        if path not in ctx.outputs:
            ctx.outputs.append(path)
    return path


def _fig_name(base: str, tag: str) -> Path:
    return Path(f"{base}_{tag}.png")


# ---------------------------------------------------------------------------
# 数据帧构造（图 5/6/8 使用 comparison.csv / metrics.json）
# ---------------------------------------------------------------------------
_COMPARISON_NUMERIC = (
    "arrival_rate",
    "safe_completion_rate",
    "center_crossing_rate",
    "min_separation",
    "violating_pair_count",
    "violation_event_count",
    "all_arrival_time",
    "t90",
    "average_path_length",
)


def _run_metric_row(run: RunInfo) -> dict[str, Any]:
    metrics = run.metrics
    status = metrics.get("status") or {}
    return {
        "experiment_id": run.experiment_id,
        "model": run.model,
        "scenario": run.scenario,
        "scene_id": run.scene_id,
        "seed": run.seed,
        "config_hash": run.config_hash,
        "status": run.status,
        "valid": run.valid,
        "arrival_rate": metrics.get("arrival_rate"),
        "safe_completion_rate": metrics.get("safe_completion_rate"),
        "center_crossing_rate": metrics.get("center_crossing_rate"),
        "min_separation": metrics.get("min_separation"),
        "violating_pair_count": metrics.get("violating_pair_count"),
        "violation_event_count": metrics.get("violation_event_count"),
        "all_arrival_time": metrics.get("all_arrival_time"),
        "t90": metrics.get("t90"),
        "average_path_length": metrics.get("average_path_length"),
        "completed": bool(status.get("completed", False)),
        "team_safe_success": bool(metrics.get("team_safe_success", False)),
    }


def _comparison_frame(ctx: _Context) -> tuple[pd.DataFrame, str]:
    """优先读 ``results/<exp>/comparison.csv``；否则由各 run 的 metrics.json 现场构造。"""
    csv_path = ctx.results_dir / ctx.experiment_id / "comparison.csv"
    df = _read_csv(csv_path)
    if df is not None and len(df) and {"model", "scenario"} <= set(df.columns):
        frame = df.copy()
        frame["model"] = frame["model"].astype(str)
        frame["scenario"] = frame["scenario"].astype(str)
        if "valid" in frame.columns:
            frame["valid"] = _as_bool_series(frame["valid"])
        else:
            frame["valid"] = True
        if "completed" in frame.columns:
            frame["completed"] = _as_bool_series(frame["completed"])
        else:
            frame["completed"] = False
        if "team_safe_success" in frame.columns:
            frame["team_safe_success"] = _as_bool_series(frame["team_safe_success"])
        else:
            frame["team_safe_success"] = False
        return frame, "comparison.csv"
    rows = [_run_metric_row(r) for r in ctx.runs]
    return pd.DataFrame(rows), "metrics.json"


def _summary_frame(ctx: _Context) -> pd.DataFrame | None:
    return _read_csv(ctx.results_dir / ctx.experiment_id / "summary_statistics.csv")


def _paired_frame(ctx: _Context) -> pd.DataFrame | None:
    return _read_csv(ctx.results_dir / ctx.experiment_id / "paired_effects.csv")


def _summary_ci(
    summary: pd.DataFrame | None, scenario: str, model: str, metric: str
) -> tuple[float, float, float, int] | None:
    if summary is None or summary.empty:
        return None
    if not {"scenario", "model"} <= set(summary.columns):
        return None
    subset = summary[(summary["scenario"].astype(str) == str(scenario)) & (summary["model"].astype(str) == str(model))]
    if subset.empty:
        return None
    row = subset.iloc[0]
    mean_col, lo_col, hi_col = f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"
    if mean_col not in subset.columns:
        return None
    mean = pd.to_numeric(pd.Series([row.get(mean_col)]), errors="coerce").iloc[0]
    lo = pd.to_numeric(pd.Series([row.get(lo_col)]), errors="coerce").iloc[0] if lo_col in subset.columns else mean
    hi = pd.to_numeric(pd.Series([row.get(hi_col)]), errors="coerce").iloc[0] if hi_col in subset.columns else mean
    n_col = f"{metric}_n"
    n = int(pd.to_numeric(pd.Series([row.get(n_col)]), errors="coerce").fillna(0).iloc[0]) if n_col in subset.columns else 0
    if not np.isfinite(float(mean)):
        return None
    return (float(mean), float(lo), float(hi), n)


def _scene_pairs(
    df: pd.DataFrame, scenario: str, model_a: str, model_b: str, metric: str
) -> tuple[np.ndarray, int]:
    """返回同一 scene 上 (model_a - model_b) 的逐场景差值与配对数。"""
    if not {"scenario", "model", "scene_id"} <= set(df.columns):
        return np.zeros(0), 0
    valid = df[df["valid"]] if "valid" in df.columns else df
    a = valid[(valid["scenario"].astype(str) == scenario) & (valid["model"].astype(str) == model_a)]
    b = valid[(valid["scenario"].astype(str) == scenario) & (valid["model"].astype(str) == model_b)]
    if a.empty or b.empty:
        return np.zeros(0), 0
    a = a[["scene_id", metric]].copy()
    b = b[["scene_id", metric]].copy()
    a["scene_id"] = a["scene_id"].astype(str)
    b["scene_id"] = b["scene_id"].astype(str)
    a[metric] = pd.to_numeric(a[metric], errors="coerce")
    b[metric] = pd.to_numeric(b[metric], errors="coerce")
    merged = a.merge(b, on="scene_id", how="inner", suffixes=("_a", "_b"))
    va = merged[f"{metric}_a"].to_numpy(dtype=np.float64)
    vb = merged[f"{metric}_b"].to_numpy(dtype=np.float64)
    both = np.isfinite(va) & np.isfinite(vb)
    return (va[both] - vb[both]), int(np.count_nonzero(both))


# ---------------------------------------------------------------------------
# fig01: 初始点/目标点/原点/参考球面
# ---------------------------------------------------------------------------
def _render_fig01(ctx: _Context, seed: int, tag: str) -> Path | None:
    loaded = _scene_for_seed(ctx, seed)
    if loaded is None:
        return None
    scene, scene_path = loaded
    run = next(iter(_runs_for_seed(ctx, seed)), None)
    params = _sim_params(ctx, run)
    radius = float(scene["meta"].get("sphere_radius", params["init_sphere_radius"])) or params["init_sphere_radius"]
    starts = np.asarray(scene["starts"], dtype=np.float64)
    goals = np.asarray(scene["goals"], dtype=np.float64)
    d_safe = params["d_safe"]
    body_radius = params["radius"]

    fig = plt.figure(figsize=(7.6, 6.6))
    try:
        ax = fig.add_subplot(111, projection="3d")
        u = np.linspace(0.0, 2.0 * np.pi, 36)
        v = np.linspace(0.0, np.pi, 18)
        xs = radius * np.outer(np.cos(u), np.sin(v))
        ys = radius * np.outer(np.sin(u), np.sin(v))
        zs = radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="#b0b0b0", linewidth=0.35, alpha=0.35, rstride=2, cstride=2)
        ax.scatter(
            starts[:, 0], starts[:, 1], starts[:, 2],
            s=10, c="#1f77b4", depthshade=False, label="starts p_i(0)",
        )
        ax.scatter(
            goals[:, 0], goals[:, 1], goals[:, 2],
            s=10, c="#d62728", depthshade=False, label="goals g_i = -p_i(0)",
        )
        ax.scatter([0.0], [0.0], [0.0], marker="*", s=150, c="black", label="origin / center region")
        lim = radius * 1.08
        _set_equal_3d(ax, -lim, lim)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=20, azim=35)
        ax.legend(loc="upper right", fontsize=7.5)
        fig.suptitle(
            f"Initial geometry — scene {scene['scene_id']} (seed {seed})",
            fontsize=12,
        )
        ax.set_title(
            f"N = {len(starts)}, R = {radius:.1f} m reference sphere (initial-condition surface, not a body); "
            f"body radius r = {body_radius:.1f} m, d_safe = {d_safe:.1f} m",
            fontsize=8,
        )
        fig.text(
            0.5, 0.005,
            "Blue = start positions, red = antipodal goals, black star = origin. "
            "The grey wireframe is the 20 m initial-condition sphere (schematic reference, not an obstacle).",
            ha="center", fontsize=7,
        )
        return _save_fig(ctx, fig, ctx.out_dir / _fig_name("fig01_initial_geometry", tag))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig02: 3D 全轨迹 + 中心放大
# ---------------------------------------------------------------------------
def _render_fig02(
    ctx: _Context,
    seed: int,
    tag: str,
    scenarios: Sequence[str] = ("A", "B"),
    models: Sequence[str] | None = None,
) -> Path | None:
    rows = [s for s in scenarios if _models_for_scenario(ctx, s, seed)]
    if not rows:
        _warn(ctx, f"fig02 skipped: no runs for seed {seed} in scenarios {list(scenarios)}")
        return None
    model_cols = [m for m in (models or _FIG2_MODELS) if m in MODEL_ORDER][:3]
    if not model_cols:
        model_cols = list(_FIG2_MODELS)
    fig = plt.figure(figsize=(15.5, 1.4 + 4.35 * len(rows)))
    try:
        for i, scenario in enumerate(rows):
            for j, model in enumerate(model_cols):
                ax = fig.add_subplot(len(rows), 4, i * 4 + j + 1, projection="3d")
                run = _find_run(ctx, scenario, model, seed)
                traj = _asset(ctx, run, "traj") if run is not None else None
                pos = np.asarray(traj["positions"], dtype=np.float64) if traj is not None and "positions" in traj else None
                if pos is None:
                    _no_data_axis(ax, f"no {scenario}/{model} run")
                    continue
                stride = max(1, int(math.ceil(pos.shape[0] / 240)))
                for agent in range(pos.shape[1]):
                    ax.plot(
                        pos[::stride, agent, 0], pos[::stride, agent, 1], pos[::stride, agent, 2],
                        color=_model_color(model), linewidth=0.45, alpha=0.45,
                    )
                lim = _points_absmax(pos) * 1.05
                _set_equal_3d(ax, -lim, lim)
                ax.set_xlabel("x [m]", fontsize=7)
                ax.set_ylabel("y [m]", fontsize=7)
                ax.set_zlabel("z [m]", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_title(f"{scenario} — {model} (seed {seed})", fontsize=9)
            # 中心放大列
            zoom_ax = fig.add_subplot(len(rows), 4, i * 4 + 4, projection="3d")
            zoom_model = "orca" if _find_run(ctx, scenario, "orca", seed) is not None else model_cols[0]
            run = _find_run(ctx, scenario, zoom_model, seed)
            traj = _asset(ctx, run, "traj") if run is not None else None
            pos = np.asarray(traj["positions"], dtype=np.float64) if traj is not None and "positions" in traj else None
            if pos is None:
                _no_data_axis(zoom_ax, "no data")
                continue
            stride = max(1, int(math.ceil(pos.shape[0] / 240)))
            zi = 2.5
            inside = np.all(np.abs(pos) <= zi, axis=2)  # (K+1, N)：只画中心放大区域内的轨迹片段
            for agent in range(pos.shape[1]):
                px = np.array(pos[::stride, agent, 0], dtype=np.float64)
                py = np.array(pos[::stride, agent, 1], dtype=np.float64)
                pz = np.array(pos[::stride, agent, 2], dtype=np.float64)
                keep = inside[::stride, agent]
                px[~keep] = np.nan
                py[~keep] = np.nan
                pz[~keep] = np.nan
                zoom_ax.plot(px, py, pz, color=_model_color(zoom_model), linewidth=0.55, alpha=0.55)
            zoom_ax.scatter([0.0], [0.0], [0.0], marker="+", s=70, c="black")
            _set_equal_3d(zoom_ax, -zi, zi)
            zoom_ax.set_xlabel("x [m]", fontsize=7)
            zoom_ax.set_ylabel("y [m]", fontsize=7)
            zoom_ax.set_zlabel("z [m]", fontsize=7)
            zoom_ax.tick_params(labelsize=6)
            zoom_ax.set_title(f"{scenario} — {zoom_model} — center zoom ±{zi:.1f} m", fontsize=9)
        fig.suptitle(
            f"All-agent 3D trajectories — seed {seed} — scenarios {', '.join(rows)} — "
            f"models {', '.join(model_cols)} — equal axes",
            fontsize=12,
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.055, wspace=0.04, hspace=0.10)
        fig.text(
            0.5, 0.005,
            "Each thin line = one agent (positions from trajectories.npz). Centre zoom = trajectories clipped to "
            "|x|,|y|,|z| <= 2.5 m around the origin; the black + marks the origin/center region.",
            ha="center", fontsize=7,
        )
        return _save_fig(ctx, fig, ctx.out_dir / _fig_name("fig02_trajectories", tag))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig03: 逐区间连续最小机间距（含 1 m 阈值）
# ---------------------------------------------------------------------------
def _min_sep_curve(ctx: _Context, run: RunInfo) -> tuple[np.ndarray, np.ndarray, str] | None:
    """返回 (t, per-interval continuous min separation, source)。优先从轨迹重算。"""
    traj = _asset(ctx, run, "traj")
    if traj is not None and "positions" in traj:
        dt = _dt_of_traj(traj)
        vel = _traj_velocities(traj, dt)
        pos = np.asarray(traj["positions"], dtype=np.float64)
        if vel is not None and pos.ndim == 3 and pos.shape[0] == vel.shape[0] + 1:
            values = interval_min_separation(pos, vel, dt)
            t = np.asarray(traj.get("t", np.arange(values.size) * dt), dtype=np.float64).ravel()
            if t.size >= values.size:
                t = t[: values.size]
            else:
                t = np.arange(values.size) * dt
            return t, values, "recomputed from positions+velocities (closed-form interval minimum)"
    csv = _asset(ctx, run, "min_distance_curve")
    if csv is not None and {"t", "min_distance"} <= set(csv.columns):
        t = _numeric(csv, "t")
        values = _numeric(csv, "min_distance")
        mask = np.isfinite(t) & np.isfinite(values)
        if np.any(mask):
            return t[mask], values[mask], "min_distance_curve.csv (metrics module continuous output)"
    return None


def _render_fig03(
    ctx: _Context,
    seed: int,
    tag: str,
    scenarios: Sequence[str] = ("A", "B"),
    models: Sequence[str] | None = None,
) -> Path | None:
    rows = [s for s in scenarios if _models_for_scenario(ctx, s, seed)]
    if not rows:
        _warn(ctx, f"fig03 skipped: no runs for seed {seed}")
        return None
    fig, axes = plt.subplots(1, len(rows), figsize=(6.4 * len(rows), 4.8), squeeze=False)
    try:
        for col, scenario in enumerate(rows):
            ax = axes[0][col]
            model_list = [m for m in (models or _FIG3_MODELS[scenario]) if m in MODEL_ORDER]
            plotted = 0
            source = "trajectory recompute"
            for model in model_list:
                run = _find_run(ctx, scenario, model, seed)
                if run is None:
                    continue
                curve = _min_sep_curve(ctx, run)
                if curve is None:
                    continue
                t, values, source = curve
                label = f"{model} (min {float(np.min(values)):.3f} m)"
                ax.plot(t, values, color=_model_color(model), linewidth=1.2, label=label)
                plotted += 1
            ax.axhline(
                D_SAFE_DEFAULT, color="crimson", linestyle="--", linewidth=1.0,
                label=f"d_safe = 2r = {D_SAFE_DEFAULT:.1f} m threshold",
            )
            ax.set_xlabel("time [s]")
            ax.set_ylabel("min inter-agent centre distance [m]")
            ax.set_title(f"Scenario {scenario} — seed {seed}", fontsize=10)
            ax.grid(alpha=0.3)
            if plotted:
                ax.legend(fontsize=7.5, loc="best")
            else:
                _no_data_axis(ax, "no trajectory/min-distance data")
        fig.suptitle(
            f"Continuous minimum inter-agent separation — per-interval minimum sequence — seed {seed}",
            fontsize=12,
        )
        fig.text(
            0.5, 0.005,
            "Curve = per-control-interval CONTINUOUS minimum over all N(N-1)/2 agent pairs, recomputed from "
            "trajectories.npz positions and constant interval velocities with the closed-form closest-approach "
            "formula t* = clip(-r·w/|w|², 0, dt), d_min = ||r + t*w||. Endpoint-only distances are NOT used. "
            "Each point corresponds to the interval [t_k, t_k + dt].",
            ha="center", fontsize=7,
        )
        return _save_fig(ctx, fig, ctx.out_dir / _fig_name("fig03_min_separation", tag))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig04: 到达数 / 安全完成数累计曲线
# ---------------------------------------------------------------------------
def _arrival_curves(
    ctx: _Context, run: RunInfo
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str] | None:
    """返回 (t, cumulative arrivals, cumulative safe completions or None, cumulative center visits or None, source)."""
    csv = _asset(ctx, run, "arrival_curve")
    if csv is not None and {"t", "cumulative_arrivals"} <= set(csv.columns):
        t = _numeric(csv, "t")
        arr = _numeric(csv, "cumulative_arrivals")
        mask = np.isfinite(t) & np.isfinite(arr)
        if np.any(mask):
            t, arr = t[mask], arr[mask]
            safe = _numeric(csv, "cumulative_safe_completions_known_later") if (
                "cumulative_safe_completions_known_later" in csv.columns
            ) else None
            center = _numeric(csv, "cumulative_center_visits") if "cumulative_center_visits" in csv.columns else None
            if safe is not None and safe.size == t.size:
                safe = safe[mask] if safe.size == mask.size else None
            if center is not None and center.size == t.size:
                center = center[mask] if center.size == mask.size else None
            return t, arr, safe, center, "arrival_curve.csv"
    traj = _asset(ctx, run, "traj")
    t_grid = None
    arrival_times = None
    safe_times = None
    if traj is not None and "t" in traj:
        t_grid = np.asarray(traj["t"], dtype=np.float64).ravel()
    agents = _asset(ctx, run, "agents")
    if agents is not None and "arrival_time" in agents.columns:
        arrival_times = _numeric(agents, "arrival_time")
        arrival_times = arrival_times[np.isfinite(arrival_times)]
        if "safe_completed" in agents.columns:
            flags = _as_bool_series(agents["safe_completed"]).to_numpy()
            times = _numeric(agents, "arrival_time")
            keep = flags & np.isfinite(times)
            safe_times = np.sort(times[keep])
    elif traj is not None and "arrival_times" in traj:
        arrival_times = _finite(np.asarray(traj["arrival_times"], dtype=np.float64))
    if t_grid is None or t_grid.size == 0:
        if arrival_times is None:
            return None
        t_grid = np.arange(0.0, max(float(np.max(arrival_times)) if arrival_times.size else 0.0) + 1e-9, DT_DEFAULT)
    if arrival_times is None:
        return None
    t_grid = np.sort(np.unique(t_grid))
    arrivals = np.searchsorted(np.sort(arrival_times), t_grid, side="right").astype(np.float64)
    safe = None
    if safe_times is not None:
        safe = np.searchsorted(np.sort(safe_times), t_grid, side="right").astype(np.float64)
    center = None
    center_events = _asset(ctx, run, "center_events")
    center_times = np.zeros(0, dtype=np.float64)
    if center_events is not None and "center_entry_time" in center_events.columns:
        center_times = _finite(_numeric(center_events, "center_entry_time"))
    elif traj is not None and "center_entry_times" in traj:
        center_times = _finite(np.asarray(traj["center_entry_times"], dtype=np.float64))
    if center_times.size:
        center = np.searchsorted(np.sort(center_times), t_grid, side="right").astype(np.float64)
    return t_grid, arrivals, safe, center, "agents.csv / trajectories.npz (derived)"


def _render_fig04(
    ctx: _Context,
    seed: int,
    tag: str,
    scenarios: Sequence[str] = ("A", "B"),
    models: Sequence[str] | None = None,
) -> Path | None:
    rows = [s for s in scenarios if _models_for_scenario(ctx, s, seed)]
    if not rows:
        _warn(ctx, f"fig04 skipped: no runs for seed {seed}")
        return None
    fig, axes = plt.subplots(1, len(rows), figsize=(6.4 * len(rows), 4.8), squeeze=False)
    try:
        for col, scenario in enumerate(rows):
            ax = axes[0][col]
            model_list = [m for m in (models or _FIG4_MODELS[scenario]) if m in MODEL_ORDER]
            n_agents = None
            plotted = 0
            for model in model_list:
                run = _find_run(ctx, scenario, model, seed)
                if run is None:
                    continue
                curves = _arrival_curves(ctx, run)
                if curves is None:
                    continue
                t, arrivals, safe, _center, _source = curves
                color = _model_color(model)
                ax.plot(t, arrivals, color=color, linewidth=1.2, label=f"{model} arrivals (final {int(arrivals[-1])})")
                if safe is not None:
                    ax.plot(t, safe, color=color, linewidth=1.1, linestyle="--",
                            label=f"{model} safe completions (final {int(safe[-1])})")
                if n_agents is None:
                    agents = _asset(ctx, run, "agents")
                    if agents is not None and "agent_id" in agents.columns:
                        n_agents = int(len(agents))
                    else:
                        traj = _asset(ctx, run, "traj")
                        if traj is not None and "positions" in traj:
                            n_agents = int(np.asarray(traj["positions"]).shape[1])
                plotted += 1
            ax.set_xlabel("time [s]")
            ax.set_ylabel("cumulative agent count")
            if n_agents:
                ax.set_ylim(0, n_agents * 1.05)
            ax.set_title(f"Scenario {scenario} — seed {seed}", fontsize=10)
            ax.grid(alpha=0.3)
            if plotted:
                ax.legend(fontsize=7, loc="lower right")
            else:
                _no_data_axis(ax, "no arrival data")
        fig.suptitle(
            f"Arrival and safe-completion progress — seed {seed} (cumulative counts, not rates)",
            fontsize=12,
        )
        fig.text(
            0.5, 0.005,
            "Solid = cumulative arrivals; dashed = cumulative SAFE completions. Safety qualification uses the "
            "retrospective full 120 s observation window (a later violation removes eligibility), so the dashed "
            "curve is NOT a real-time quantity. Missing arrival/center times are simply not counted (never plotted as 0).",
            ha="center", fontsize=7,
        )
        return _save_fig(ctx, fig, ctx.out_dir / _fig_name("fig04_arrival_curves", tag))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig05: 五项课程指标对比
# ---------------------------------------------------------------------------
_CURRICULUM = (
    ("arrival_rate", "arrival rate"),
    ("min_separation", "global min separation [m]"),
    ("violating_pair_count", "violating pair count"),
    ("all_arrival_time", "all-arrival time [s]\n(conditional)"),
    ("average_path_length", "mean path length [m]"),
)


def _render_fig05(ctx: _Context) -> Path | None:
    df, source = _comparison_frame(ctx)
    if df.empty:
        _warn(ctx, "fig05 skipped: no comparison data")
        return None
    df = df[df["valid"]].copy()
    if df.empty:
        _warn(ctx, "fig05 skipped: no valid runs")
        return None
    scenarios = [s for s in ("A", "B") if s in set(df["scenario"].astype(str))]
    if not scenarios:
        _warn(ctx, "fig05 skipped: unknown scenarios")
        return None
    models = [m for m in MODEL_ORDER if m in set(df["model"].astype(str))]
    if not models:
        models = sorted(set(df["model"].astype(str)))
    n_all_runs = len(df)
    n_invalid = len(ctx.runs) - len(df)
    fig, axes = plt.subplots(1, len(_CURRICULUM), figsize=(18.5, 5.8), squeeze=False)
    width = 0.8 / max(len(scenarios), 1)
    try:
        for col, (metric, display) in enumerate(_CURRICULUM):
            ax = axes[0][col]
            for si, scenario in enumerate(scenarios):
                for mi, model in enumerate(models):
                    group = df[(df["scenario"].astype(str) == scenario) & (df["model"].astype(str) == model)]
                    if group.empty:
                        continue
                    values = _numeric(group, metric)
                    n_runs = int(len(group))
                    n_completed = int(_as_bool_series(group["completed"]).sum()) if "completed" in group.columns else 0
                    x = mi - 0.5 * (len(scenarios) - 1) * width + si * width
                    if metric == "all_arrival_time":
                        finite = values[np.isfinite(values)]
                        n_defined = int(finite.size)
                        if n_defined:
                            mean = float(np.mean(finite))
                            sd = float(np.std(finite, ddof=1)) if n_defined > 1 else 0.0
                            ax.bar(x, mean, width=width * 0.92, color=SCENARIO_COLORS[scenario],
                                   alpha=0.85, zorder=2)
                            ax.errorbar([x], [mean], yerr=[sd], fmt="none", ecolor="#333333",
                                        capsize=2, linewidth=0.8, zorder=3)
                            ax.text(x, mean + sd, f"n{n_defined}/{n_runs}", ha="center", va="bottom",
                                    fontsize=5.4, rotation=90, zorder=4)
                        else:
                            ax.text(x, 0.02, "n=0", ha="center", va="bottom", fontsize=5.4, rotation=90,
                                    color="#666666")
                    else:
                        finite = values[np.isfinite(values)]
                        if not finite.size:
                            continue
                        mean = float(np.mean(finite))
                        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                        ax.bar(x, mean, width=width * 0.92, color=SCENARIO_COLORS[scenario], alpha=0.85, zorder=2)
                        ax.errorbar([x], [mean], yerr=[sd], fmt="none", ecolor="#333333",
                                    capsize=2, linewidth=0.8, zorder=3)
                        ax.text(x, mean, f"v{n_runs}/c{n_completed}", ha="center",
                                va="bottom", fontsize=5.2, rotation=90, zorder=4)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
            ax.set_title(display, fontsize=9.5)
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0.0, color="#999999", linewidth=0.6, zorder=1)
        handles = [Patch(facecolor=SCENARIO_COLORS[s], label=f"scenario {s}") for s in scenarios]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.935),
                   fontsize=8, ncol=len(scenarios))
        fig.suptitle(
            f"Five curriculum indicators — experiment '{ctx.experiment_id}' — scene-level mean over valid runs "
            f"(error bars: ±1 SD across scenes)",
            fontsize=12, y=0.975,
        )
        fig.subplots_adjust(left=0.045, right=0.99, top=0.82, bottom=0.28, wspace=0.28)
        fig.text(
            0.5, 0.075,
            f"Source: {source}. v = valid runs in the bar mean; c = fully-completed runs (all agents arrived). "
            f"Invalid runs excluded ({n_invalid} invalid of {len(ctx.runs)}; {n_all_runs} valid).",
            ha="center", fontsize=6.8,
        )
        fig.text(
            0.5, 0.022,
            "all-arrival time is conditional on EVERY agent arriving: runs where it is null are omitted and "
            "annotated n=0 (never drawn as 0). Arrival rate may include violating agents; safety is in fig06.",
            ha="center", fontsize=6.8,
        )
        return _save_fig(ctx, fig, ctx.out_dir / "fig05_curriculum_metrics.png")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig06: 安全完成率 + 全队安全成功 + 场景配对差
# ---------------------------------------------------------------------------
def _local_pairs_for_fig06() -> list[tuple[str, str]]:
    return [
        ("orca_release", "orca"),
        ("orca", "vo"),
        ("orca", "rvo"),
        ("orca_release", "direct_release"),
        ("orca", "direct_release"),
        ("direct_release", "direct"),
    ]


def _render_fig06(ctx: _Context) -> Path | None:
    df, source = _comparison_frame(ctx)
    if df.empty:
        _warn(ctx, "fig06 skipped: no comparison data")
        return None
    df = df.copy()
    if "valid" in df.columns:
        df = df[df["valid"]]
    if df.empty:
        _warn(ctx, "fig06 skipped: no valid runs")
        return None
    scenario = "B" if (df["scenario"].astype(str) == "B").any() else ("A" if (df["scenario"].astype(str) == "A").any() else None)
    if scenario is None:
        _warn(ctx, "fig06 skipped: no scenario A/B data")
        return None
    summary = _summary_frame(ctx)
    paired = _paired_frame(ctx)
    models = [m for m in SCENARIO_MODELS[scenario] if m in set(df["model"].astype(str))]
    if not models:
        models = sorted(set(df["model"].astype(str)))
    sub = df[df["scenario"].astype(str) == scenario]

    fig = plt.figure(figsize=(18.4, 5.2))
    try:
        ax_a = fig.add_axes([0.045, 0.20, 0.255, 0.64])
        ax_b = fig.add_axes([0.355, 0.20, 0.245, 0.64])
        ax_c = fig.add_axes([0.725, 0.20, 0.26, 0.64])
        ci_from_summary = False
        for j, model in enumerate(models):
            rates = _numeric(sub[sub["model"].astype(str) == model], "safe_completion_rate")
            rates = rates[np.isfinite(rates)]
            if not rates.size:
                continue
            color = _model_color(model)
            x = j + _jitter(rates.size)
            ax_a.scatter(x, rates, s=16, color=color, alpha=0.6, edgecolors="none", zorder=2)
            ci = _summary_ci(summary, scenario, model, "safe_completion_rate")
            if ci is None:
                ci = _bootstrap_mean_ci(rates)
            else:
                ci_from_summary = True
            mean, lo, hi, _n = ci
            ax_a.errorbar([j], [mean], yerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                          fmt="D", color="black", ecolor="black", capsize=3, markersize=5, zorder=3)
        ax_a.set_xticks(range(len(models)))
        ax_a.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax_a.set_ylim(-0.03, 1.03)
        ax_a.set_ylabel("safe completion rate")
        ax_a.set_title("Safe completion rate\n(points = scenes; ◆ mean + 95% CI)", fontsize=9.5)
        ax_a.grid(alpha=0.3)

        for j, model in enumerate(models):
            group = sub[sub["model"].astype(str) == model]
            if group.empty:
                continue
            successes = _as_bool_series(group["team_safe_success"]).to_numpy(dtype=bool)
            color = _model_color(model)
            x = j + _jitter(successes.size)
            ax_b.scatter(x, successes.astype(float), s=14, color=color, alpha=0.55, edgecolors="none", zorder=2)
            mean = float(np.mean(successes)) if successes.size else float("nan")
            ax_b.scatter([j], [mean], marker="D", s=42, color="black", zorder=3)
            if successes.size:
                ax_b.text(j, min(mean + 0.05, 1.02), f"{mean:.2f}\n(n={int(successes.size)})",
                          ha="center", va="bottom", fontsize=6.5)
        ax_b.set_xticks(range(len(models)))
        ax_b.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax_b.set_ylim(-0.05, 1.15)
        ax_b.set_ylabel("team safe success (1 = all completed, zero violations)")
        ax_b.set_title("Team safe success rate\n(points = scenes; ◆ scene mean)", fontsize=9.5)
        ax_b.grid(alpha=0.3)

        rows: list[dict[str, Any]] = []
        if paired is not None and not paired.empty and {"scenario", "metric", "pair"} <= set(paired.columns):
            filt = paired[
                (paired["scenario"].astype(str) == scenario)
                & (paired["metric"].astype(str) == "safe_completion_rate")
            ]
            for _, row in filt.iterrows():
                mean_diff = pd.to_numeric(pd.Series([row.get("mean_diff")]), errors="coerce").iloc[0]
                lo = pd.to_numeric(pd.Series([row.get("ci_low")]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([row.get("ci_high")]), errors="coerce").iloc[0]
                if not np.isfinite(float(mean_diff)):
                    continue
                rows.append(
                    {
                        "pair": str(row.get("pair")),
                        "mean_diff": float(mean_diff),
                        "ci_low": float(lo) if np.isfinite(float(lo)) else float(mean_diff),
                        "ci_high": float(hi) if np.isfinite(float(hi)) else float(mean_diff),
                        "n_pairs": int(pd.to_numeric(pd.Series([row.get("n_pairs")]), errors="coerce").fillna(0).iloc[0]),
                        "source": "paired_effects.csv",
                    }
                )
        pairs_source = "paired_effects.csv" if rows else "local recompute (paired_effects.csv unavailable)"
        if not rows:
            for model_a, model_b in _local_pairs_for_fig06():
                if model_a not in models or model_b not in models:
                    continue
                diffs, n_pairs = _scene_pairs(sub, scenario, model_a, model_b, "safe_completion_rate")
                if not diffs.size:
                    continue
                mean, lo, hi, n = _bootstrap_diff_ci(diffs)
                rows.append(
                    {
                        "pair": f"{model_a} - {model_b}",
                        "mean_diff": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_pairs": n,
                        "source": "local",
                    }
                )
        if rows:
            ys = np.arange(len(rows))[::-1]
            all_lo = [float(row["ci_low"]) for row in rows] + [0.0]
            all_hi = [float(row["ci_high"]) for row in rows] + [0.0]
            for y, row in zip(ys, rows):
                mean = float(row["mean_diff"])
                lo = float(row["ci_low"])
                hi = float(row["ci_high"])
                ax_c.errorbar([mean], [y], xerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                              fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3, markersize=5, zorder=3)
                ax_c.text(mean, y + 0.22, f"n={row['n_pairs']}", ha="center", va="bottom", fontsize=6.5)
            span = max(float(max(all_hi) - min(all_lo)), 1e-6)
            ax_c.set_xlim(min(all_lo) - 0.08 * span, max(all_hi) + 0.08 * span)
            ax_c.set_yticks(ys)
            ax_c.set_yticklabels([str(row["pair"]) for row in rows], fontsize=7.5)
            ax_c.axvline(0.0, color="#999999", linewidth=0.8, linestyle=":")
        else:
            _no_data_axis(ax_c, "no paired contrast data")
        ax_c.set_xlabel("paired difference (model_a - model_b)")
        ax_c.set_title(
            f"Paired differences ({scenario}, safe completion rate)\n95% percentile bootstrap CI",
            fontsize=9.5,
        )
        ax_c.grid(alpha=0.3)

        ci_source = (
            "summary_statistics.csv / paired_effects.csv (scene-level bootstrap, 10,000 resamples, fixed seed)"
            if (ci_from_summary or any(row["source"] == "paired_effects.csv" for row in rows))
            else "local scene-level bootstrap 95% percentile interval (fixed seed, 10,000 resamples; CSV aggregates unavailable)"
        )
        fig.suptitle(
            f"Safety summary — experiment '{ctx.experiment_id}' — scenario {scenario} — "
            f"per-scene points and paired contrasts (independent unit = scene)",
            fontsize=12,
        )
        fig.text(
            0.5, 0.02,
            f"CI: {ci_source}. Pair panel source: {pairs_source}. Safe completion uses the retrospective full-window "
            "violation status; paired differences share identical scenes.",
            ha="center", fontsize=7,
        )
        return _save_fig(ctx, fig, ctx.out_dir / "fig06_safe_completion_paired.png")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig07: 中心首次进入事件栅格 + 累计访问 + 计划/实际放行
# ---------------------------------------------------------------------------
def _center_entries(ctx: _Context, run: RunInfo) -> tuple[np.ndarray, np.ndarray]:
    df = _asset(ctx, run, "center_events")
    ids: np.ndarray
    times: np.ndarray
    if df is not None and {"agent_id", "center_entry_time"} <= set(df.columns):
        ids = _numeric(df, "agent_id").astype(np.int64)
        times = _numeric(df, "center_entry_time")
        mask = np.isfinite(times)
        return ids[mask], times[mask]
    traj = _asset(ctx, run, "traj")
    if traj is not None and "center_entry_times" in traj:
        times = np.asarray(traj["center_entry_times"], dtype=np.float64)
        ids = np.arange(times.size, dtype=np.int64)
        mask = np.isfinite(times)
        return ids[mask], times[mask]
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)


def _render_fig07(
    ctx: _Context,
    seed: int,
    tag: str,
    models: Sequence[str] | None = None,
) -> Path | None:
    model_list = [m for m in (models or _FIG7_MODELS) if m in MODEL_ORDER]
    available = [(m, _find_run(ctx, "B", m, seed)) for m in model_list]
    available = [(m, r) for m, r in available if r is not None]
    if not available:
        _warn(ctx, f"fig07 skipped: no scenario-B centre-event runs for seed {seed}")
        return None
    fig = plt.figure(figsize=(17.0, 4.9))
    try:
        ax_raster = fig.add_axes([0.05, 0.17, 0.30, 0.66])
        ax_cum = fig.add_axes([0.40, 0.17, 0.27, 0.66])
        ax_rel = fig.add_axes([0.73, 0.17, 0.24, 0.66])
        n_agents = 0
        for model, run in available:
            ids, times = _center_entries(ctx, run)
            n_agents = max(n_agents, int(ids.max()) + 1 if ids.size else 0)
            traj = _asset(ctx, run, "traj")
            if traj is not None and "positions" in traj:
                n_agents = max(n_agents, int(np.asarray(traj["positions"]).shape[1]))
            ax_raster.scatter(times, ids, marker="o", s=11, color=_model_color(model),
                              alpha=0.85, label=f"{model} ({ids.size} agents)")
            curves = _arrival_curves(ctx, run)
            if curves is not None:
                t, _arrivals, _safe, center, _src = curves
                if center is not None:
                    ax_cum.plot(t, center, color=_model_color(model), linewidth=1.3, label=model)
        ax_raster.set_xlabel("first centre-entry time [s]")
        ax_raster.set_ylabel("agent id")
        ax_raster.set_title(f"Centre first-entry raster — B seed {seed}", fontsize=9.5)
        ax_raster.legend(fontsize=6.5, loc="lower right")
        ax_raster.grid(alpha=0.25)

        ax_cum.set_xlabel("time [s]")
        ax_cum.set_ylabel("cumulative agents that visited centre")
        ax_cum.set_title("Cumulative centre visits (first visit only)", fontsize=9.5)
        if n_agents:
            ax_cum.set_ylim(0, n_agents * 1.05)
        ax_cum.legend(fontsize=7.5)
        ax_cum.grid(alpha=0.25)

        release_run = _find_run(ctx, "B", "orca_release", seed) or _find_run(ctx, "B", "direct_release", seed)
        if release_run is None:
            _no_data_axis(ax_rel, "no release-model run")
        else:
            agents = _asset(ctx, release_run, "agents")
            planned = _numeric(agents, "release_planned") if agents is not None else np.zeros(0)
            actual = _numeric(agents, "release_actual") if agents is not None else np.zeros(0)
            mask = np.isfinite(planned) & np.isfinite(actual)
            planned, actual = planned[mask], actual[mask]
            if planned.size:
                ax_rel.scatter(planned, actual, s=14, color=_model_color(release_run.model), alpha=0.8)
                lo = float(min(planned.min(), actual.min()))
                hi = float(max(planned.max(), actual.max()))
                pad = max(1.0, 0.05 * (hi - lo))
                ax_rel.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999999",
                            linestyle="--", linewidth=0.8, label="actual = planned")
                ax_rel.set_xlim(lo - pad, hi + pad)
                ax_rel.set_ylim(lo - pad, hi + pad)
                ax_rel.legend(fontsize=7, loc="upper left")
                ax_rel.text(
                    0.97, 0.05,
                    f"{release_run.model}\nlast actual = {float(actual.max()):.1f} s\n"
                    f"quantised to dt grid",
                    transform=ax_rel.transAxes, ha="right", va="bottom", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
                )
            else:
                _no_data_axis(ax_rel, "release columns missing")
            ax_rel.set_xlabel("planned release time [s]")
            ax_rel.set_ylabel("actual release time [s]")
            ax_rel.set_title("Planned vs actual release (staggered)", fontsize=9.5)
            ax_rel.grid(alpha=0.25)
        fig.suptitle(
            f"Centre-region access and staggered release — scenario B — seed {seed}",
            fontsize=12,
        )
        fig.text(
            0.5, 0.02,
            "Centre visit = first in-step entry into ||p|| <= 0.1 m; only the first visit is recorded (later re-entries "
            "do not reset). Release times are quantised by release_tick = ceil(release_planned/dt), actual = "
            "release_tick*dt (dt = 0.1 s), so actual >= planned and the last slot is 74.3 s for 100 agents.",
            ha="center", fontsize=7,
        )
        return _save_fig(ctx, fig, ctx.out_dir / _fig_name("fig07_center_events", tag))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig08: 到达率-平均路程散点 + 等待/放行后耗时分解
# ---------------------------------------------------------------------------
def _render_fig08(ctx: _Context) -> Path | None:
    df, source = _comparison_frame(ctx)
    if df.empty:
        _warn(ctx, "fig08 skipped: no comparison data")
        return None
    df = df[df["valid"]].copy()
    if df.empty:
        _warn(ctx, "fig08 skipped: no valid runs")
        return None
    fig, (ax_scatter, ax_wait) = plt.subplots(1, 2, figsize=(14.2, 5.0))
    try:
        scenarios = [s for s in ("A", "B") if s in set(df["scenario"].astype(str))]
        for scenario in scenarios:
            for model in MODEL_ORDER:
                sub = df[(df["scenario"].astype(str) == scenario) & (df["model"].astype(str) == model)]
                if sub.empty:
                    continue
                rates = _numeric(sub, "arrival_rate")
                paths = _numeric(sub, "average_path_length")
                mask = np.isfinite(rates) & np.isfinite(paths)
                if not np.any(mask):
                    continue
                ax_scatter.scatter(
                    paths[mask], rates[mask],
                    marker="o" if scenario == "A" else "s",
                    s=26, color=_model_color(model),
                    edgecolors=SCENARIO_COLORS[scenario], linewidths=0.6, alpha=0.75,
                )
        handles = [Line2D([], [], marker="o", linestyle="", color=_model_color(m), label=m, markersize=6)
                   for m in MODEL_ORDER if (df["model"].astype(str) == m).any()]
        handles += [
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=SCENARIO_COLORS[s], label=f"scenario {s}", markersize=6)
            for s in scenarios
        ]
        ax_scatter.legend(handles=handles, fontsize=6.5, loc="lower right", ncol=2)
        ax_scatter.set_xlabel("mean path length [m]")
        ax_scatter.set_ylabel("arrival rate")
        ax_scatter.set_title("Arrival rate vs mean path length\n(each point = one valid scene run)", fontsize=9.5)
        ax_scatter.set_ylim(-0.03, 1.03)
        ax_scatter.grid(alpha=0.3)

        release_models = [m for m in ("orca_release", "direct_release") if m in set(df["model"].astype(str))]
        b_runs = [r for r in ctx.runs if r.scenario == "B" and r.valid and r.model in release_models]
        labels: list[str] = []
        waits: list[float] = []
        travels: list[float] = []
        n_scenes_list: list[int] = []
        for model in release_models:
            wait_means: list[float] = []
            travel_means: list[float] = []
            for run in b_runs:
                if run.model != model:
                    continue
                agents = _asset(ctx, run, "agents")
                if agents is None:
                    continue
                wait_vals = _finite(_numeric(agents, "wait_duration"))
                travel_vals = _finite(_numeric(agents, "post_release_travel_time"))
                if wait_vals.size:
                    wait_means.append(float(np.mean(wait_vals)))
                if travel_vals.size:
                    travel_means.append(float(np.mean(travel_vals)))
            if not wait_means and not travel_means:
                continue
            labels.append(model)
            waits.append(float(np.mean(wait_means)) if wait_means else 0.0)
            travels.append(float(np.mean(travel_means)) if travel_means else 0.0)
            n_scenes_list.append(max(len(wait_means), len(travel_means)))
        if labels:
            x = np.arange(len(labels))
            ax_wait.bar(x, waits, width=0.55, color="#ff9f43", label="mean wait duration (pre-release)")
            ax_wait.bar(x, travels, width=0.55, bottom=waits, color="#54a0ff",
                        label="mean post-release travel time")
            for i, (w, tr, ns) in enumerate(zip(waits, travels, n_scenes_list)):
                ax_wait.text(i, w + 0.5 * tr, f"{w:.1f} + {tr:.1f} s\n(n={ns} scenes)",
                             ha="center", va="center", fontsize=7.5, color="white",
                             bbox={"facecolor": "#00000033", "edgecolor": "none", "pad": 1.0})
            ax_wait.set_xlim(-0.6, len(labels) - 0.4)
            ax_wait.set_xticks(x)
            ax_wait.set_xticklabels(labels, fontsize=8)
            ax_wait.legend(fontsize=8, loc="upper left", framealpha=0.95)
        else:
            _no_data_axis(ax_wait, "no scenario-B release-model runs found")
        ax_wait.set_ylabel("time [s] (mean over agents and scenes)")
        ax_wait.set_title("Wait vs post-release travel decomposition\n(staggered-release models, scenario B)", fontsize=9.5)
        ax_wait.grid(axis="y", alpha=0.3)
        fig.suptitle(
            f"Path/safety trade-off and staggered-release cost — experiment '{ctx.experiment_id}'",
            fontsize=12,
        )
        fig.text(
            0.5, 0.01,
            f"Source: {source} for the scatter; agents.csv wait_duration/post_release_travel_time for the bars. "
            "wait_duration = time before the actual release (includes the scheduled stagger wait); "
            "post_release_travel_time = travel time from actual release to arrival. Missing (unfinished) post-release "
            "times are skipped, never counted as 0. Scatter uses valid runs only.",
            ha="center", fontsize=7,
        )
        return _save_fig(ctx, fig, ctx.out_dir / "fig08_arrival_path_wait.png")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# fig09: 动画静态快照（2D 俯视，A/B 主展示的静态补充）
# ---------------------------------------------------------------------------
def _render_fig09(
    ctx: _Context,
    seed: int,
    tag: str,
    models: Sequence[str] | None = None,
) -> Path | None:
    model_list = [m for m in (models or _ANIM_MODELS) if m in MODEL_ORDER]
    runs = [(m, _find_run(ctx, "B", m, seed)) for m in model_list]
    runs = [(m, r) for m, r in runs if r is not None]
    if not runs:
        _warn(ctx, f"fig09 skipped: no scenario-B orca/orca_release runs for seed {seed}")
        return None
    fig, axes = plt.subplots(len(runs), 4, figsize=(16.0, 3.3 * len(runs) + 1.5), squeeze=False)
    try:
        for row, (model, run) in enumerate(runs):
            traj = _asset(ctx, run, "traj")
            pos = np.asarray(traj["positions"], dtype=np.float64) if traj is not None and "positions" in traj else None
            stages = np.asarray(traj["stages"]) if traj is not None and "stages" in traj else None
            t = np.asarray(traj["t"], dtype=np.float64).ravel() if traj is not None and "t" in traj else None
            if pos is None or stages is None or t is None:
                for col in range(4):
                    _no_data_axis(axes[row][col], f"no trajectory for {model}")
                continue
            n_steps = pos.shape[0] - 1
            agents = _asset(ctx, run, "agents")
            violators: set[int] = set()
            if agents is not None and {"agent_id", "violated"} <= set(agents.columns):
                flags = _as_bool_series(agents["violated"]).to_numpy()
                ids = _numeric(agents, "agent_id").astype(np.int64)
                violators = {int(i) for i, flag in zip(ids, flags) if flag}
            snapshots = [0, n_steps // 3, (2 * n_steps) // 3, n_steps]
            lim = _points_absmax(pos) * 1.08
            for col, k in enumerate(snapshots):
                ax = axes[row][col]
                for agent in range(pos.shape[1]):
                    ax.plot(pos[: k + 1, agent, 0], pos[: k + 1, agent, 1],
                            color="#c8c8c8", linewidth=0.25, alpha=0.35)
                stage_k = stages[k]
                colors = [STAGE_COLORS.get(int(s), "#000000") for s in stage_k]
                sizes = [22 if a in violators else 10 for a in range(pos.shape[1])]
                edges = ["red" if a in violators else "none" for a in range(pos.shape[1])]
                lengths = [0.9 if a in violators else 0.0 for a in range(pos.shape[1])]
                ax.scatter(pos[k, :, 0], pos[k, :, 1], s=sizes, c=colors, edgecolors=edges,
                           linewidths=lengths, zorder=3)
                n_arrived = int(np.sum(np.asarray(stage_k) == STAGE_ARRIVED))
                dmin = _instant_min_separation(pos[k])
                ax.text(
                    0.03, 0.97,
                    f"t = {float(t[k]):.1f} s\narrived = {n_arrived}\nmin sep = {dmin:.3f} m",
                    transform=ax.transAxes, va="top", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
                )
                _set_equal_2d(ax)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal", adjustable="box")
                ax.set_title(f"{model} — t={float(t[k]):.1f} s", fontsize=9)
                ax.grid(alpha=0.2)
                ax.set_xlabel("x [m]", fontsize=7)
                ax.set_ylabel("y [m]", fontsize=7)
        fig.suptitle(
            f"Animation snapshots (top view x–y) — scenario B — seed {seed} — ORCA vs ORCA+Release",
            fontsize=12,
        )
        fig.subplots_adjust(left=0.055, right=0.99, top=0.875, bottom=0.14, wspace=0.34, hspace=0.30)
        handles = [Line2D([], [], marker="o", linestyle="", color=STAGE_COLORS[s], label=f"stage: {STAGE_LABELS[s]}",
                          markersize=6) for s in sorted(STAGE_COLORS)]
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                              markeredgecolor="red", label="ever in violation", markersize=6))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045),
                   ncol=5, fontsize=8, frameon=False)
        fig.text(
            0.5, 0.012,
            "Markers = agent centres (2D top-view projection); no true-radius body spheres drawn (body radius "
            "r = 0.5 m, d_safe = 1.0 m). Static companion of anim_orca_B_seed*.gif / anim_orca_release_B_seed*.gif.",
            ha="center", fontsize=6.8,
        )
        return _save_fig(ctx, fig, ctx.out_dir / _fig_name("fig09_animation_snapshots", tag))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 动画（GIF 最低要求；可选 MP4）
# ---------------------------------------------------------------------------
def _instant_min_separation(position: np.ndarray) -> float:
    p = np.asarray(position, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return float("nan")
    r = p[:, None, :] - p[None, :, :]
    d = np.sqrt(np.einsum("...i,...i->...", r, r))
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


def _ffmpeg_available() -> bool:
    try:
        return bool(animation.FFMpegWriter.isAvailable())
    except (AttributeError, RuntimeError):  # pragma: no cover - 环境相关
        return False


def _load_animation_bundle(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    traj = _read_npz(run_dir / "trajectories.npz", only=_TRAJ_KEYS)
    if traj is None:
        raise FileNotFoundError(f"trajectories.npz not found/readable in {run_dir}")
    metrics = _read_json(run_dir / "metrics.json") or {}
    prov = _read_json(run_dir / "provenance.json") or {}
    cfg = _read_json(run_dir / "config.json") or {}
    model = str(prov.get("model") or metrics.get("model") or run_dir.parent.parent.name)
    scenario = str(prov.get("scenario") or metrics.get("scenario") or run_dir.parent.parent.parent.name)
    try:
        seed = int(prov.get("seed", metrics.get("seed", -1)))
    except (TypeError, ValueError):
        seed = -1
    sim = (cfg.get("simulation") or {}) if isinstance(cfg, dict) else {}
    return {
        "run_dir": run_dir,
        "traj": traj,
        "agents": _read_csv(run_dir / "agents.csv"),
        "center_events": _read_csv(run_dir / "center_events.csv"),
        "metrics": metrics,
        "provenance": prov,
        "model": model,
        "scenario": scenario,
        "seed": seed,
        "dt": float((cfg.get("experiment") or {}).get("dt", DT_DEFAULT)) if isinstance(cfg, dict) else DT_DEFAULT,
        "body_radius": float(sim.get("radius", BODY_RADIUS_DEFAULT)),
        "d_safe": float(sim.get("d_safe", D_SAFE_DEFAULT)),
    }


def _animate_bundle(bundle: dict[str, Any], out_path: Path, fps: int = 25, want_mp4: bool = False) -> Path:
    traj = bundle["traj"]
    pos = np.asarray(traj["positions"], dtype=np.float64)
    stages = np.asarray(traj["stages"])
    t = np.asarray(traj["t"], dtype=np.float64).ravel()
    if pos.ndim != 3 or stages.shape[0] != pos.shape[0]:
        raise ValueError(f"inconsistent trajectory arrays in {bundle['run_dir']}")
    n_agents = int(pos.shape[1])
    n_steps = pos.shape[0] - 1
    stride = max(1, int(math.ceil(n_steps / float(ANIM_MAX_FRAMES))))
    frame_steps = list(range(0, n_steps + 1, stride))
    if frame_steps[-1] != n_steps:
        frame_steps.append(n_steps)

    agents = bundle.get("agents")
    violators: set[int] = set()
    if agents is not None and {"agent_id", "violated"} <= set(agents.columns):
        flags = _as_bool_series(agents["violated"]).to_numpy()
        ids = _numeric(agents, "agent_id").astype(np.int64)
        violators = {int(i) for i, flag in zip(ids, flags) if flag}

    center_times = np.zeros(0, dtype=np.float64)
    center_events = bundle.get("center_events")
    if center_events is not None and "center_entry_time" in center_events.columns:
        center_times = _finite(_numeric(center_events, "center_entry_time"))
    elif "center_entry_times" in traj:
        center_times = _finite(np.asarray(traj["center_entry_times"], dtype=np.float64))

    label_ids = [int(i) for i in ANIM_LABEL_IDS if int(i) < n_agents]
    body_radius = float(bundle.get("body_radius", BODY_RADIUS_DEFAULT))
    d_safe = float(bundle.get("d_safe", D_SAFE_DEFAULT))
    model = str(bundle.get("model", ""))
    scenario = str(bundle.get("scenario", ""))
    seed = int(bundle.get("seed", -1))

    out_path = Path(out_path)
    gif_path = out_path if out_path.suffix.lower() == ".gif" else out_path.with_suffix(".gif")
    mp4_path = out_path if out_path.suffix.lower() == ".mp4" else gif_path.with_suffix(".mp4")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    playback_seconds = len(frame_steps) / max(float(fps), 1.0)
    real_seconds = float(t[-1]) if t.size else 0.0
    speed_factor = real_seconds / playback_seconds if playback_seconds > 0 else float("nan")

    fig = plt.figure(figsize=(11.6, 5.8))
    try:
        ax_main = fig.add_axes([0.035, 0.235, 0.44, 0.60], projection="3d")
        ax_zoom = fig.add_axes([0.525, 0.235, 0.44, 0.60], projection="3d")

        line_stride = max(1, int(math.ceil(pos.shape[0] / 160)))
        for agent in range(n_agents):
            if agent in violators:
                color, lw, alpha = "#d62728", 0.8, 0.75
            else:
                color, lw, alpha = "#9aa0a6", 0.3, 0.35
            ax_main.plot(pos[::line_stride, agent, 0], pos[::line_stride, agent, 1],
                         pos[::line_stride, agent, 2], color=color, linewidth=lw, alpha=alpha)
        lim = _points_absmax(pos) * 1.05
        _set_equal_3d(ax_main, -lim, lim)
        ax_main.set_title("full view", fontsize=9)
        ax_main.set_xlabel("x [m]", fontsize=7)
        ax_main.set_ylabel("y [m]", fontsize=7)
        ax_main.set_zlabel("z [m]", fontsize=7)
        ax_main.tick_params(labelsize=6)
        ax_main.view_init(elev=18, azim=35)

        _set_equal_3d(ax_zoom, -ANIM_ZOOM_RADIUS, ANIM_ZOOM_RADIUS)
        ax_zoom.scatter([0.0], [0.0], [0.0], marker="+", s=70, c="black", zorder=4)
        ax_zoom.set_title(f"center zoom |p| <= {ANIM_ZOOM_RADIUS:.1f} m", fontsize=9)
        ax_zoom.set_xlabel("x [m]", fontsize=7)
        ax_zoom.set_ylabel("y [m]", fontsize=7)
        ax_zoom.set_zlabel("z [m]", fontsize=7)
        ax_zoom.tick_params(labelsize=6)
        ax_zoom.view_init(elev=24, azim=35)

        fig.suptitle(
            f"{model} — scenario {scenario} — seed {seed} — ORCA vs ORCA+Release comparison animation",
            fontsize=11,
        )
        annotation = fig.text(0.035, 0.895, "", fontsize=10, family="monospace")
        fig.text(
            0.5, 0.165,
            f"Stage colours: waiting=gray / to_center=orange / to_goal=blue / arrived=green. Red = agents ever in "
            f"violation. Markers = agent centres; NO true-radius body spheres drawn (body radius r = "
            f"{body_radius:.2f} m; d_safe = {d_safe:.2f} m).",
            ha="center", va="top", fontsize=6.8,
        )
        fig.text(
            0.5, 0.118,
            f"Playback: {len(frame_steps)} frames at {fps} fps (~{speed_factor:.0f}× real time; "
            f"every {stride} control step(s)). Left = full 3D view, right = centre zoom |p| <= "
            f"{ANIM_ZOOM_RADIUS:.1f} m.",
            ha="center", va="top", fontsize=6.8,
        )
        handles = [
            Line2D([], [], marker="o", linestyle="", color=STAGE_COLORS[s],
                   label=STAGE_LABELS[s], markersize=6)
            for s in sorted(STAGE_COLORS)
        ]
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                              markeredgecolor="red", label="ever in violation", markersize=6))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=5, fontsize=7.5, frameon=False)

        dynamic: list[Any] = []

        def update(frame_index: int) -> tuple[()]:
            nonlocal dynamic
            for artist in dynamic:
                try:
                    artist.remove()
                except (ValueError, NotImplementedError):  # pragma: no cover - 防御性
                    pass
            dynamic = []
            k = frame_steps[frame_index]
            p = pos[k]
            stage_k = stages[k]
            colors = [STAGE_COLORS.get(int(s), "#000000") for s in stage_k]
            sizes = [30 if agent in violators else 14 for agent in range(n_agents)]
            edges = ["red" if agent in violators else "none" for agent in range(n_agents)]
            widths = [1.0 if agent in violators else 0.0 for agent in range(n_agents)]
            dynamic.append(
                ax_main.scatter(p[:, 0], p[:, 1], p[:, 2], s=sizes, c=colors,
                                edgecolors=edges, linewidths=widths, depthshade=False)
            )
            for agent in label_ids:
                dynamic.append(
                    ax_main.text(p[agent, 0], p[agent, 1], p[agent, 2], f"#{agent}",
                                 fontsize=6, color="#222222")
                )
            inner = np.flatnonzero(np.all(np.abs(p) <= ANIM_ZOOM_RADIUS, axis=1))
            if inner.size:
                dynamic.append(
                    ax_zoom.scatter(p[inner, 0], p[inner, 1], p[inner, 2],
                                    s=[30 if int(i) in violators else 16 for i in inner],
                                    c=[colors[int(i)] for i in inner],
                                    edgecolors=["red" if int(i) in violators else "none" for i in inner],
                                    linewidths=[1.0 if int(i) in violators else 0.0 for i in inner],
                                    depthshade=False)
                )
                for agent in inner:
                    if int(agent) in label_ids:
                        dynamic.append(
                            ax_zoom.text(p[agent, 0], p[agent, 1], p[agent, 2], f"#{int(agent)}",
                                         fontsize=6, color="#222222")
                        )
            else:
                dynamic.append(
                    ax_zoom.text(0.0, 0.0, ANIM_ZOOM_RADIUS, "no agents within zoom region",
                                 fontsize=7, color="#666666")
                )
            n_arrived = int(np.sum(np.asarray(stage_k) == STAGE_ARRIVED))
            n_center = int(np.count_nonzero(center_times <= float(t[k]))) if center_times.size else 0
            dmin = _instant_min_separation(p)
            dmin_text = f"{dmin:.3f} m" if np.isfinite(dmin) else "n/a"
            annotation.set_text(
                f"t = {float(t[k]):6.1f} s    arrived = {n_arrived:3d}    "
                f"center visits = {n_center:3d}    current min separation = {dmin_text}"
            )
            return ()

        anim = animation.FuncAnimation(fig, update, frames=len(frame_steps),
                                       interval=1000.0 / max(float(fps), 1.0), blit=False)
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=ANIM_DPI)
        if want_mp4:
            if not _ffmpeg_available():
                print("[visualization] WARNING: ffmpeg not available; MP4 skipped (GIF kept)",
                      file=sys.stderr)
            else:
                try:
                    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
                    anim.save(mp4_path, writer=writer, dpi=ANIM_DPI)
                except Exception as exc:  # noqa: BLE001 - MP4 失败不得阻塞静态图/GIF
                    print(f"[visualization] WARNING: MP4 encoding failed ({type(exc).__name__}: {exc}); GIF kept",
                          file=sys.stderr)
        del anim
        return gif_path
    finally:
        plt.close(fig)


def render_animation(run_dir: Path, out_path: Path, fps: int = 25, want_mp4: bool = False) -> Path:
    """渲染单个 run 的 GIF 动画（最低要求 GIF；ffmpeg 可用且 ``want_mp4=True`` 时另存 MP4）。

    Parameters
    ----------
    run_dir : Path
        ``results/<exp>/<scenario>/<model>/scene_<id>/<config_hash>`` 运行目录。
    out_path : Path
        输出 GIF 路径（``.gif``）；若给 ``.mp4`` 则 GIF 写到同名的 ``.gif``，MP4 写到该路径。
    fps : int
        帧率。
    want_mp4 : bool
        是否额外尝试编码 MP4（ffmpeg 缺失时只打印 warning，GIF 仍然产出）。

    Returns
    -------
    Path
        已写入的 GIF 路径。
    """
    bundle = _load_animation_bundle(Path(run_dir))
    gif = _animate_bundle(bundle, Path(out_path), fps=fps, want_mp4=bool(want_mp4))
    if not gif.exists() or gif.stat().st_size <= 0:
        raise RuntimeError(f"GIF animation was not written: {gif}")
    return gif


# ---------------------------------------------------------------------------
# 渲染编排：render_all / render_scene_seed
# ---------------------------------------------------------------------------
def _tag_for_seed(seed: int, tag: str) -> str:
    return f"seed{int(seed)}" if tag == "display" else f"median_seed{int(seed)}"


def _selected_display_seed(ctx: _Context) -> tuple[int | None, str | None]:
    seeds = sorted({r.seed for r in ctx.runs if r.seed >= 0})
    if not seeds:
        return None, "no non-negative scene seeds found"
    if DISPLAY_SEED in seeds:
        return DISPLAY_SEED, None
    fallback = seeds[0]
    return fallback, (
        f"display seed {DISPLAY_SEED} not present in experiment '{ctx.experiment_id}'; "
        f"fell back to smallest available seed {fallback}"
    )


def _median_case_seed(ctx: _Context) -> tuple[int | None, dict[str, Any]]:
    candidates: dict[int, float] = {}
    for run in ctx.runs:
        if run.scenario != "B" or run.model != "orca_release" or not run.valid:
            continue
        rate = run.metrics.get("safe_completion_rate")
        try:
            value = float(rate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        candidates[int(run.seed)] = value
    info: dict[str, Any] = {
        "rule": MEDIAN_CASE_RULE,
        "candidates": {str(seed): float(rate) for seed, rate in sorted(candidates.items())},
        "median_rate": None,
        "selected_seed": None,
        "selection_distance": None,
        "note": "",
    }
    if not candidates:
        info["note"] = "no valid scenario-B orca_release run with finite safe_completion_rate"
        return None, info
    seeds = np.array(sorted(candidates), dtype=np.int64)
    rates = np.array([candidates[int(s)] for s in seeds], dtype=np.float64)
    median_rate = float(np.median(rates))
    distances = np.abs(rates - median_rate)
    best = float(np.min(distances))
    tie = np.flatnonzero(distances <= best + 1e-12)
    chosen = int(seeds[int(tie[0])])  # ties broken by smallest seed
    info.update(
        {
            "median_rate": median_rate,
            "selected_seed": chosen,
            "selection_distance": float(np.min(distances)),
            "tie_break": "smallest seed" if tie.size > 1 else "unique nearest",
        }
    )
    return chosen, info


def _execute_scene_figures(
    ctx: _Context,
    seed: int,
    tag: str,
    *,
    scenarios: Sequence[str] = ("A", "B"),
    models: Sequence[str] | None = None,
    include_aggregates: bool = False,
    include_animation: bool = True,
) -> None:
    """渲染一个 (seed, tag) 的全部场景相关图（fig01/02/03/04/07/09 + 动画）。"""
    scene_tag = _tag_for_seed(seed, tag)
    _emit(ctx, f"fig01[{scene_tag}]", _render_fig01, seed, scene_tag)
    _emit(ctx, f"fig02[{scene_tag}]", _render_fig02, seed, scene_tag, scenarios, models)
    _emit(ctx, f"fig03[{scene_tag}]", _render_fig03, seed, scene_tag, scenarios, models)
    _emit(ctx, f"fig04[{scene_tag}]", _render_fig04, seed, scene_tag, scenarios, models)
    if "B" in scenarios:
        _emit(ctx, f"fig07[{scene_tag}]", _render_fig07, seed, scene_tag, models)
        _emit(ctx, f"fig09[{scene_tag}]", _render_fig09, seed, scene_tag, models)
    if include_animation and "B" in scenarios:
        anim_models = [m for m in _ANIM_MODELS if m in MODEL_ORDER]
        if models is not None:
            anim_models = [m for m in anim_models if m in set(models)]
        for model in anim_models:
            run = _find_run(ctx, "B", model, seed)
            if run is None:
                continue
            out = ctx.out_dir / f"anim_{model}_B_seed{int(seed)}.gif"
            path = _emit(
                ctx,
                f"anim[{model},{scene_tag}]",
                _render_animation_file,
                run.run_dir,
                out,
            )
            if path is not None:
                _ = path
    if include_aggregates:
        _emit(ctx, "fig05", _render_fig05)
        _emit(ctx, "fig06", _render_fig06)
        _emit(ctx, "fig08", _render_fig08)


def _render_animation_file(ctx: _Context, run_dir: Path, out_path: Path) -> Path:
    path = render_animation(Path(run_dir), Path(out_path), fps=ANIM_FPS, want_mp4=False)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"animation file empty: {path}")
    return path


def _write_selection_json(ctx: _Context, median_info: dict[str, Any], display_note: str | None) -> Path:
    path = ctx.out_dir / "selection.json"
    relative_outputs = []
    for output in ctx.outputs:
        try:
            relative_outputs.append(str(output.relative_to(ctx.out_dir)))
        except ValueError:
            relative_outputs.append(str(output))
    payload = {
        "generated_by": "uavbench.visualization.render_all",
        "display_seed_fixed": DISPLAY_SEED,
        "display_seed_used": ctx.selection.get("display_seed"),
        "display_seed_note": display_note,
        "median_case": median_info,
        "median_case_note": median_info.get("note", ""),
        "experiment_id": ctx.experiment_id,
        "experiments_present": sorted({r.experiment_id for r in ctx.all_runs}),
        "n_runs_discovered": len(ctx.all_runs),
        "n_runs_in_main_experiment": len(ctx.runs),
        "selection_rules": {
            "display_seed": f"fixed display seed {DISPLAY_SEED} (handoff §10.1)",
            "median_case": MEDIAN_CASE_RULE,
        },
        "warnings": list(ctx.warnings),
        "outputs": relative_outputs,
        "runs": {
            f"{run.experiment_id}:{run.scenario}:{run.model}:scene_{run.scene_id}": str(
                _relative_or_absolute(run.run_dir, ctx.results_dir)
            )
            for run in ctx.runs
        },
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def render_all(results_dir: Path, out_dir: Path, configs_dir: Path | None = None) -> list[Path]:
    """在完整结果树上渲染全部必做图（fig01–fig09）、动画与 ``selection.json``。

    缺失的实验/模型/scene 会让对应图打印 warning 并跳过，其余图照常产出。
    返回写出的文件路径列表（全部位于 ``out_dir`` 内）。
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    if not results_dir.exists():
        _warn(f"results_dir does not exist: {results_dir}; nothing to render")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    all_runs = discover_runs(results_dir)
    if not all_runs:
        _warn(f"no runs with metrics.json discovered under {results_dir}; nothing to render")
        return []
    experiment_id = _primary_experiment(all_runs) or ""
    runs = [r for r in all_runs if r.experiment_id == experiment_id]
    ctx = _Context(
        results_dir=results_dir,
        out_dir=out_dir,
        configs_dir=Path(configs_dir) if configs_dir is not None else None,
        all_runs=all_runs,
        runs=runs,
        experiment_id=experiment_id,
    )
    if experiment_id != "formal":
        _warn(ctx, f"experiment 'formal' not found; using '{experiment_id}' for aggregate figures")
    display_seed, display_note = _selected_display_seed(ctx)
    median_seed, median_info = _median_case_seed(ctx)
    ctx.selection["display_seed"] = display_seed
    if display_note:
        _warn(ctx, display_note)
    if median_seed is None:
        _warn(ctx, f"median-case figures skipped: {median_info.get('note', 'no candidate')}")
    elif median_seed == display_seed:
        median_info["note"] = (
            f"median-case seed equals display seed {display_seed}; median-case duplicates not rendered"
        )

    _emit(ctx, "fig05", _render_fig05)
    _emit(ctx, "fig06", _render_fig06)
    _emit(ctx, "fig08", _render_fig08)

    seeds_to_render: list[tuple[str, int]] = []
    if display_seed is not None:
        seeds_to_render.append(("display", int(display_seed)))
    if median_seed is not None and median_seed != display_seed:
        seeds_to_render.append(("median", int(median_seed)))
    for tag, seed in seeds_to_render:
        _execute_scene_figures(ctx, seed, tag, include_aggregates=False, include_animation=True)

    selection_path = _write_selection_json(ctx, median_info, display_note)
    if selection_path not in ctx.outputs:
        ctx.outputs.append(selection_path)
    return ctx.outputs


def render_scene_seed(
    scenario: str,
    seed: int,
    results_dir: Path,
    out_dir: Path,
    models: Sequence[str] | None = None,
) -> list[Path]:
    """只渲染单个 (scenario, seed) 的场景相关图与动画（CLI ``render`` 使用）。

    聚合图（fig05/06/08）不在此函数中渲染；完整渲染请用 :func:`render_all`。
    """
    scenario = str(scenario).upper()
    if scenario not in ("A", "B"):
        _warn(f"unknown scenario {scenario!r}; expected 'A' or 'B'")
        return []
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    if not results_dir.exists():
        _warn(f"results_dir does not exist: {results_dir}; nothing to render")
        return []
    all_runs = discover_runs(results_dir)
    seed_runs = [r for r in all_runs if r.seed == int(seed)]
    if not seed_runs:
        _warn(f"no runs found for seed {seed} under {results_dir}; nothing to render")
        return []
    experiment_id = _ordered_experiments({r.experiment_id for r in seed_runs})[0]
    runs = [r for r in all_runs if r.experiment_id == experiment_id]
    ctx = _Context(
        results_dir=results_dir,
        out_dir=out_dir,
        configs_dir=None,
        all_runs=all_runs,
        runs=runs,
        experiment_id=experiment_id,
    )
    ctx.selection["display_seed"] = int(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_list = [m for m in (models or SCENARIO_MODELS[scenario]) if m in MODEL_ORDER]
    if not model_list:
        model_list = list(SCENARIO_MODELS[scenario])
    tag = f"seed{int(seed)}"
    _emit(ctx, "fig01", _render_fig01, int(seed), tag)
    for scenario_i in (scenario,):
        _emit(ctx, "fig02", _render_fig02, int(seed), tag, (scenario_i,), model_list)
        _emit(ctx, "fig03", _render_fig03, int(seed), tag, (scenario_i,), model_list)
        _emit(ctx, "fig04", _render_fig04, int(seed), tag, (scenario_i,), model_list)
    if scenario == "B":
        _emit(ctx, "fig07", _render_fig07, int(seed), tag, model_list)
        _emit(ctx, "fig09", _render_fig09, int(seed), tag, model_list)
        for model in _ANIM_MODELS:
            if model not in model_list:
                continue
            run = _find_run(ctx, "B", model, int(seed))
            if run is None:
                continue
            _emit(
                ctx,
                f"anim[{model}]",
                _render_animation_file,
                run.run_dir,
                ctx.out_dir / f"anim_{model}_B_seed{int(seed)}.gif",
            )
    return ctx.outputs
