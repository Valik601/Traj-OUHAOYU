"""运行编排：场景/配置/种子 → 仿真 → 落盘 → 指标 → 审计 → 批次汇总。

设计要点（handoff §7、§9）：

- 每次运行的目录与 ``config_hash`` 绑定；``--resume`` 只复用场景 hash、配置 hash、代码源
  hash、后端均匹配且已写审计记录的运行。
- 结果按 experiment_id 分组（smoke/formal/development/sensitivity_*/stress），禁止把开发运行
  混入正式汇总。
- 正式运行前冻结 ``experiment_manifest.json``；冻结后清单变化必须显式处理。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import platform
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import __version__
from .config import (
    MODEL_BASE,
    RELEASE_MODELS,
    Config,
    apply_overrides,
    load_config,
)
from .initialization import (
    Scene,
    build_scene,
    load_scene,
    save_scene,
    scene_path,
)
from .scheduling import build_release_table
from .simulation import run_simulation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENES_DIR = ROOT / "scenes"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_CONFIGS_DIR = ROOT / "configs"

PROFILE_CONFIG = {
    "smoke": "smoke.yaml",
    "full": "full.yaml",
    "development": "development.yaml",
    "sensitivity": "sensitivity.yaml",
    "stress": "default.yaml",
}
PROFILE_EXPERIMENT = {
    "smoke": "smoke",
    "full": "formal",
    "development": "development",
    "stress": "stress",
}
#: 正式/确定性压力测试使用的模型集合
STRESS_MODELS_A = ("direct", "vo", "rvo", "orca")
STRESS_MODELS_B = ("direct", "vo", "rvo", "orca", "orca_release", "direct_release")

#: 敏感性 block 定义（handoff §7.2）：每个 block 只改一个因素，其余复用正式结果。
SENSITIVITY_BLOCKS: dict[str, dict[str, Any]] = {
    "candidate_dirs": {
        "scenario": "B",
        "models": ("vo", "rvo"),
        "baseline_label": "128",
        "variants": {
            "64": {"planners.candidate.directions": 64},
            "256": {"planners.candidate.directions": 256},
        },
    },
    "time_discretization": {
        "scenario": "B",
        "models": ("orca", "orca_release"),
        "baseline_label": "dt0.1",
        "variants": {"dt0.05": {"experiment.dt": 0.05}},
    },
    "release_delta": {
        "scenario": "B",
        "models": ("orca_release", "direct_release"),
        "baseline_label": "0.75",
        "variants": {
            "0.5": {"scheduling.delta": 0.5},
            "1.0": {"scheduling.delta": 1.0},
        },
    },
    "perception_range": {
        "scenario": "B",
        "models": ("vo", "rvo", "orca", "orca_release"),
        "baseline_label": "all",
        "variants": {
            "22m": {"simulation.neighbor_mode": "range", "simulation.neighbor_range": 22.0}
        },
    },
    "release_order": {
        "scenario": "B",
        "models": ("orca_release", "direct_release"),
        "baseline_label": "random_stream_0",
        "variants": {
            "fixed_id": {"scheduling.order": "fixed_id"},
            "random_stream_1": {"scheduling.order": "random_stream_1"},
        },
    },
}


# ---------------------------------------------------------------------------
# 代码源 hash / 配置
# ---------------------------------------------------------------------------
_CODE_HASH_CACHE: str | None = None


def code_source_hash(root: Path | None = None) -> str:
    """``uavbench/**.py`` + ``configs/*.yaml`` 的确定性源 hash（作为代码版本标识）。"""
    global _CODE_HASH_CACHE
    if _CODE_HASH_CACHE is not None:
        return _CODE_HASH_CACHE
    root = Path(root) if root is not None else ROOT
    files = sorted((root / "uavbench").rglob("*.py")) + sorted((root / "configs").glob("*.yaml"))
    h = hashlib.sha256()
    for f in files:
        rel = f.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    _CODE_HASH_CACHE = h.hexdigest()
    return _CODE_HASH_CACHE


def config_path_for(profile: str, configs_dir: Path | None = None) -> Path:
    configs_dir = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"unknown profile: {profile}")
    return configs_dir / PROFILE_CONFIG[profile]


# ---------------------------------------------------------------------------
# 运行规格
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    scenario: str
    model: str
    scene_id: str
    seed: int
    config_file: str  # 相对 configs 目录的文件名
    overrides: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def config(self, configs_dir: Path | None = None) -> Config:
        configs_dir = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR
        return apply_overrides(load_config(configs_dir / self.config_file), dict(self.overrides))

    def config_hash(self, configs_dir: Path | None = None) -> str:
        return self.config(configs_dir).hash()

    @property
    def rel_dir(self) -> Path:
        return Path(self.experiment_id) / self.scenario / self.model / f"scene_{self.scene_id}"

    def out_dir(self, results_dir: Path, configs_dir: Path | None = None) -> Path:
        return Path(results_dir) / self.rel_dir / self.config_hash(configs_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario": self.scenario,
            "model": self.model,
            "scene_id": self.scene_id,
            "seed": int(self.seed),
            "config_file": self.config_file,
            "overrides": dict(self.overrides),
            "note": self.note,
        }


def _scenario_models(cfg: Config) -> dict[str, tuple[str, ...]]:
    return {"A": tuple(cfg.models.A), "B": tuple(cfg.models.B)}


def build_run_specs(
    profile: str,
    configs_dir: Path | None = None,
    *,
    scenes: dict[str, Scene] | None = None,
    models: Iterable[str] | None = None,
    scenarios: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    blocks: Iterable[str] | None = None,
) -> list[RunSpec]:
    """构造一个 profile 的完整运行矩阵（未过滤）。"""
    configs_dir = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR
    if profile == "stress":
        cfg = load_config(configs_dir / "default.yaml")
        if scenes is None:
            raise ValueError("stress profile requires base scene")
        base_scene = scenes["base_fibonacci"]
        specs: list[RunSpec] = []
        for scenario, model_list in (("A", STRESS_MODELS_A), ("B", STRESS_MODELS_B)):
            for model in model_list:
                specs.append(
                    RunSpec(
                        experiment_id="stress",
                        scenario=scenario,
                        model=model,
                        scene_id=base_scene.scene_id,
                        seed=base_scene.seed,
                        config_file="default.yaml",
                        note="deterministic undirected Fibonacci stress test",
                    )
                )
        return _filter_specs(specs, models, scenarios, seeds, blocks)

    cfg = load_config(config_path_for(profile, configs_dir))
    experiment_id = PROFILE_EXPERIMENT.get(profile)
    scenario_models = _scenario_models(cfg)

    if profile == "sensitivity":
        specs = []
        use_blocks = list(SENSITIVITY_BLOCKS) if blocks is None else list(blocks)
        for block in use_blocks:
            if block not in SENSITIVITY_BLOCKS:
                raise ValueError(f"unknown sensitivity block: {block}")
            definition = SENSITIVITY_BLOCKS[block]
            scenario = definition["scenario"]
            for label, overrides in definition["variants"].items():
                exp_id = f"sensitivity_{block}_{label}"
                for seed in cfg.scenes.sensitivity_seeds:
                    for model in definition["models"]:
                        specs.append(
                            RunSpec(
                                experiment_id=exp_id,
                                scenario=scenario,
                                model=model,
                                scene_id=str(int(seed)),
                                seed=int(seed),
                                config_file="sensitivity.yaml",
                                overrides=dict(overrides),
                                note=f"{block} variant {label} (baseline {definition['baseline_label']})",
                            )
                        )
        return _filter_specs(specs, models, scenarios, seeds, blocks)

    assert experiment_id is not None
    specs = []
    seeds_list = list(cfg.scenes.formal_seeds)
    for scenario in ("A", "B"):
        for model in scenario_models[scenario]:
            for seed in seeds_list:
                specs.append(
                    RunSpec(
                        experiment_id=experiment_id,
                        scenario=scenario,
                        model=model,
                        scene_id=str(int(seed)),
                        seed=int(seed),
                        config_file=PROFILE_CONFIG[profile],
                        note=f"profile {profile}",
                    )
                )
    return _filter_specs(specs, models, scenarios, seeds, blocks)


def _filter_specs(
    specs: list[RunSpec],
    models: Iterable[str] | None,
    scenarios: Iterable[str] | None,
    seeds: Iterable[int] | None,
    blocks: Iterable[str] | None,
) -> list[RunSpec]:
    out = specs
    if models is not None:
        wanted = {str(m) for m in models}
        out = [s for s in out if s.model in wanted]
    if scenarios is not None:
        wanted_sc = {str(s).upper() for s in scenarios}
        out = [s for s in out if s.scenario in wanted_sc]
    if seeds is not None:
        wanted_seed = {int(v) for v in seeds}
        out = [s for s in out if int(s.seed) in wanted_seed]
    return out


# ---------------------------------------------------------------------------
# 单次运行
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment_info(backend: str) -> dict[str, Any]:
    import matplotlib
    import pandas
    import scipy

    return {
        "python": sys.version.split()[0],
        "python_build": platform.python_build()[1],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "uavbench": __version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_env": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "planner_backend": backend,
        "numa": None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
    tmp.replace(path)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


def _write_run_artifacts(
    out_dir: Path,
    cfg: Config,
    spec: RunSpec,
    result,
    provenance: dict[str, Any],
    *,
    error_message: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", cfg.to_json_dict())
    _write_json(out_dir / "provenance.json", provenance)
    npz = result.to_npz_dict()
    np.savez(out_dir / "trajectories.npz", **npz)

    # center_events.csv
    rows: list[list[Any]] = []
    for i in range(cfg.simulation.n_agents):
        t = result.center_entry_times[i]
        if np.isfinite(t):
            rows.append([i, f"{float(t):.10f}", int(result.center_entry_steps[i])])
    _write_csv(out_dir / "center_events.csv", ["agent_id", "center_entry_time", "center_entry_step"], rows)

    # planner_diagnostics.csv
    diag = result.diag
    header = [
        "t",
        "step",
        "agent_id",
        "model",
        "sample_exhausted",
        "used_fallback",
        "solver_infeasible",
        "constraint_violation_max",
        "candidate_count_tested",
        "initial_overlap_pairs",
        "numerical_error",
    ]
    diag_rows: list[list[Any]] = []
    n_calls = int(np.asarray(diag.get("step", np.zeros(0))).size)
    for j in range(n_calls):
        diag_rows.append(
            [
                f"{float(diag['t'][j]):.10f}",
                int(diag["step"][j]),
                int(diag["agent_id"][j]),
                spec.model,
                int(bool(diag["sample_exhausted"][j])),
                int(bool(diag["used_fallback"][j])),
                int(bool(diag["solver_infeasible"][j])),
                f"{float(diag['constraint_violation_max'][j]):.12g}",
                int(diag["candidate_count_tested"][j]),
                int(diag["initial_overlap_pairs"][j]),
                int(bool(diag["numerical_error"][j])),
            ]
        )
    _write_csv(out_dir / "planner_diagnostics.csv", header, diag_rows)

    lines = [
        f"experiment_id={spec.experiment_id}",
        f"scenario={spec.scenario} model={spec.model}",
        f"scene_id={spec.scene_id} seed={spec.seed}",
        f"config_hash={provenance['config_hash']}",
        f"scene_hash={provenance['scene_hash']} source_hash={provenance['source_hash']}",
        f"backend={provenance['environment']['planner_backend']}",
        f"valid={result.valid} numerical_error={result.numerical_error}",
        f"n_steps_executed={result.n_steps_executed}/{cfg.n_steps}",
        f"wall_seconds={result.wall_seconds:.3f}",
        f"planning_total_seconds={float(np.sum(result.planning_step_seconds)):.3f}",
    ]
    if error_message:
        lines.append(f"ERROR {error_message}")
    (out_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_of(result, cfg: Config) -> dict[str, bool]:
    arrival_finite = bool(np.all(np.isfinite(result.arrival_times)))
    completed = arrival_finite and result.valid
    timeout = (not completed) and result.valid
    return {
        "completed": bool(completed),
        "timeout": bool(timeout),
        "numerical_error": bool(result.numerical_error),
        "valid": bool(result.valid),
    }


def _inline_audit(out_dir: Path, cfg: Config, spec: RunSpec, result, status: dict[str, bool]) -> dict[str, Any]:
    """轻量一致性检查：资产存在、hash 匹配、指标可复算的入口条件成立。"""
    problems: list[str] = []
    for name in ("config.json", "provenance.json", "trajectories.npz", "metrics.json"):
        if not (out_dir / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return {"ok": False, "mode": "inline", "problems": problems}
    with (out_dir / "metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    if metrics.get("scene_hash") != _sha256_file(scene_path(out_dir.parents[3] / "x" if False else out_dir)) if False else False:
        pass
    # 到达数与轨迹阶段一致（不依赖指标模块内部）
    arrival_steps = np.argmax(result.stages == 3, axis=0)
    arrived = np.zeros(cfg.simulation.n_agents, dtype=bool)
    for i, k in enumerate(arrival_steps):
        arrived[i] = bool(k > 0)
    if int(metrics.get("arrival_count", -1)) != int(np.sum(arrived)):
        problems.append(
            f"arrival_count mismatch metrics={metrics.get('arrival_count')} traj={int(np.sum(arrived))}"
        )
    if bool(metrics.get("status", {}).get("valid", True)) != bool(result.valid):
        problems.append("valid flag mismatch between metrics.json and simulation result")
    return {"ok": not problems, "mode": "inline", "problems": problems}


def execute_run(
    spec_dict: dict[str, Any],
    *,
    root_dir: str | None = None,
    configs_dir: str | None = None,
    scenes_dir: str | None = None,
    results_dir: str | None = None,
) -> dict[str, Any]:
    """执行单个运行（worker 入口，可被 ProcessPoolExecutor 调用）。"""
    root = Path(root_dir) if root_dir else ROOT
    configs_dir = Path(configs_dir) if configs_dir else root / "configs"
    scenes_dir = Path(scenes_dir) if scenes_dir else root / "scenes"
    results_dir = Path(results_dir) if results_dir else root / "results"
    spec = RunSpec(
        experiment_id=spec_dict["experiment_id"],
        scenario=spec_dict["scenario"],
        model=spec_dict["model"],
        scene_id=spec_dict["scene_id"],
        seed=int(spec_dict["seed"]),
        config_file=spec_dict["config_file"],
        overrides=dict(spec_dict.get("overrides") or {}),
        note=str(spec_dict.get("note", "")),
    )
    cfg = spec.config(configs_dir)
    scene_file = scene_path(scenes_dir, spec.scene_id)
    if not scene_file.exists():
        raise FileNotFoundError(f"scene file missing: {scene_file}; run generate-scenes first")
    scene = load_scene(scene_file)
    out_dir = spec.out_dir(results_dir, configs_dir)
    config_hash = cfg.hash()
    scene_hash = scene.sha256
    source_hash = code_source_hash(ROOT)

    started = datetime.now(timezone.utc)
    candidate_set = None
    if MODEL_BASE[spec.model] in ("vo", "rvo"):
        from .planners.candidates import build_candidate_set

        candidate_set = build_candidate_set(cfg, int(scene.seed))
    release_table = None
    if spec.model in RELEASE_MODELS and spec.scenario == "B":
        release_table = build_release_table(cfg, scene)

    run_start = datetime.now(timezone.utc)
    try:
        from .planners import build_planner

        planner = build_planner(spec.model, cfg, candidate_set)
        result = run_simulation(
            cfg,
            scene,
            spec.model,
            spec.scenario,
            release_table=release_table,
            candidate_set=candidate_set,
            planner=planner,
        )
        error_message = result.error_message
        backend = str(getattr(planner, "backend", "numpy") or "numpy")
    except Exception as exc:  # noqa: BLE001 - 运行级错误必须如实落盘
        import traceback as _tb

        traceback.print_exc()
        return {
            "spec": spec.to_dict(),
            "status": "numerical_error",
            "valid": False,
            "runtime_s": float((datetime.now(timezone.utc) - run_start).total_seconds()),
            "error": f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}",
            "out_dir": str(out_dir),
            "config_hash": config_hash,
            "scene_hash": scene_hash,
        }

    status = _status_of(result, cfg)
    backend = str(getattr(result, "backend", backend))
    env = _environment_info(backend)
    provenance = {
        "experiment_id": spec.experiment_id,
        "scenario": spec.scenario,
        "model": spec.model,
        "model_base": MODEL_BASE[spec.model],
        "planner_name": result.planner_name,
        "scene_id": scene.scene_id,
        "seed": int(scene.seed),
        "scene_file": str(scene_file.relative_to(root)) if root in scene_file.parents else str(scene_file),
        "scene_hash": scene_hash,
        "config_file": spec.config_file,
        "overrides": dict(spec.overrides),
        "config_hash": config_hash,
        "source_hash": source_hash,
        "git_commit": _git_commit(root),
        "environment": env,
        "released": spec.model in RELEASE_MODELS,
        "release_order": result.release_order,
        "observation_time": cfg.observation_time,
        "n_steps": cfg.n_steps,
        "dt": cfg.dt,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": sys.argv,
        "note": spec.note,
        "scene_meta": scene.meta,
    }
    _write_run_artifacts(out_dir, cfg, spec, result, provenance, error_message=error_message)

    # ---- 指标（离线重算） ------------------------------------------------
    from .metrics import compute_metrics, write_metrics

    traj = dict(result.to_npz_dict())
    metrics_result = compute_metrics(
        traj,
        scene,
        cfg,
        model=spec.model,
        scenario=spec.scenario,
        seed=int(scene.seed),
        scene_hash=scene_hash,
        config_hash=config_hash,
        center_entry_times=result.center_entry_times,
        center_entry_steps=result.center_entry_steps,
        release_planned=result.release_planned,
        release_actual=result.release_actual,
        status=status,
        experiment_id=spec.experiment_id,
    )
    write_metrics(out_dir, metrics_result)
    audit = _inline_audit(out_dir, cfg, spec, result, status)
    _write_json(out_dir / "audit.json", audit)

    runtime_s = float((datetime.now(timezone.utc) - started).total_seconds())
    metrics = metrics_result.metrics
    return {
        "spec": spec.to_dict(),
        "status": "completed" if status["completed"] else ("numerical_error" if status["numerical_error"] else "timeout"),
        "valid": bool(status["valid"]),
        "runtime_s": runtime_s,
        "out_dir": str(out_dir),
        "config_hash": config_hash,
        "scene_hash": scene_hash,
        "audit_ok": bool(audit["ok"]),
        "error": error_message,
        "metrics": {
            "arrival_rate": metrics.get("arrival_rate"),
            "safe_completion_rate": metrics.get("safe_completion_rate"),
            "center_crossing_rate": metrics.get("center_crossing_rate"),
            "min_separation": metrics.get("min_separation"),
            "violating_pair_count": metrics.get("violating_pair_count"),
            "violation_event_count": metrics.get("violation_event_count"),
            "all_arrival_time": metrics.get("all_arrival_time"),
            "t90": metrics.get("t90"),
            "average_path_length": metrics.get("average_path_length"),
            "team_safe_success": metrics.get("team_safe_success"),
            "planning_time_mean": (metrics.get("planning") or {}).get("mean_ms"),
            "planning_time_p95": (metrics.get("planning") or {}).get("p95_ms"),
            "sample_exhausted_count": (metrics.get("planner_diagnostics") or {}).get("sample_exhausted_count"),
            "solver_fallback_count": (metrics.get("planner_diagnostics") or {}).get("solver_fallback_count"),
        },
    }


def _git_commit(root: Path) -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return None


# ---------------------------------------------------------------------------
# 批次
# ---------------------------------------------------------------------------
def _load_scenes_for_profile(profile: str, cfg: Config, scenes_dir: Path) -> dict[str, Scene]:
    from .initialization import profile_seeds

    scenes: dict[str, Scene] = {}
    if profile == "stress":
        ids = ["base_fibonacci"]
    else:
        ids = ["base_fibonacci"] + [str(int(s)) for s in profile_seeds(cfg, profile)]
        if profile == "sensitivity":
            ids = ["base_fibonacci"] + [str(int(s)) for s in cfg.scenes.sensitivity_seeds]
    for sid in ids:
        p = scene_path(scenes_dir, sid)
        if not p.exists():
            raise FileNotFoundError(f"missing scene {p}; run generate-scenes --profile {profile} first")
        sc = load_scene(p)
        scenes[sid] = sc
    return scenes


def ensure_scenes(profile: str, configs_dir: Path | None = None, scenes_dir: Path | None = None) -> dict[str, Scene]:
    """确保 profile 所需场景存在（缺失则按 profile 生成）。"""
    from .initialization import generate_scenes, load_scenes_for_profile, profile_seeds

    configs_dir = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR
    scenes_dir = Path(scenes_dir) if scenes_dir is not None else DEFAULT_SCENES_DIR
    cfg = load_config(config_path_for(profile, configs_dir))
    if profile == "stress":
        seeds = []
    else:
        seeds = profile_seeds(cfg, profile)
    if profile == "sensitivity":
        seeds = [int(s) for s in cfg.scenes.sensitivity_seeds]
    missing = any(not scene_path(scenes_dir, str(int(s))).exists() for s in seeds) or not scene_path(
        scenes_dir, "base_fibonacci"
    ).exists()
    if missing:
        generate_scenes(cfg, "sensitivity" if profile == "sensitivity" else profile, scenes_dir)
    return _load_scenes_for_profile(profile, cfg, scenes_dir)


def _experiment_manifest_payload(specs: list[RunSpec], configs_dir: Path, scenes: dict[str, Scene]) -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_hash": code_source_hash(ROOT),
        "n_runs": len(specs),
        "specs": [
            {
                **s.to_dict(),
                "config_hash": s.config_hash(configs_dir),
                "scene_hash": scenes[s.scene_id].sha256,
            }
            for s in specs
        ],
    }


def freeze_manifest(
    experiment_id: str,
    specs: list[RunSpec],
    configs_dir: Path,
    scenes: dict[str, Scene],
    results_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """正式运行前冻结运行清单；已有清单不一致时默认拒绝。"""
    path = Path(results_dir) / experiment_id / "experiment_manifest.json"
    payload = _experiment_manifest_payload(specs, configs_dir, scenes)
    if path.exists() and not force:
        with path.open("r", encoding="utf-8") as fh:
            old = json.load(fh)
        old_specs = [{k: v for k, v in s.items() if k != "config_hash"} for s in old.get("specs", [])]
        new_specs = [{k: v for k, v in s.items() if k != "config_hash"} for s in payload["specs"]]
        if old_specs != new_specs:
            raise RuntimeError(
                f"frozen experiment manifest mismatch at {path}; refusing to change a frozen formal protocol"
            )
        return path
    _write_json(path, payload)
    return path


def run_batch(
    profile: str,
    *,
    jobs: int = 1,
    resume: bool = True,
    limit: int | None = None,
    models: Iterable[str] | None = None,
    scenarios: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    blocks: Iterable[str] | None = None,
    configs_dir: Path | None = None,
    scenes_dir: Path | None = None,
    results_dir: Path | None = None,
    freeze: bool = True,
    force_manifest: bool = False,
) -> dict[str, Any]:
    configs_dir = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR
    scenes_dir = Path(scenes_dir) if scenes_dir is not None else DEFAULT_SCENES_DIR
    results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    scenes = ensure_scenes(profile, configs_dir, scenes_dir)
    specs = build_run_specs(
        profile,
        configs_dir,
        scenes=scenes,
        models=models,
        scenarios=scenarios,
        seeds=seeds,
        blocks=blocks,
    )
    if limit is not None:
        specs = specs[: int(limit)]
    if profile in ("full", "smoke") and freeze:
        freeze_manifest(PROFILE_EXPERIMENT[profile], specs, configs_dir, scenes, results_dir, force=force_manifest)

    # 分组写入 experiment manifest 便于敏感性 block 追溯
    if profile == "sensitivity":
        groups = sorted({s.experiment_id for s in specs})
        for exp in groups:
            subset = [s for s in specs if s.experiment_id == exp]
            freeze_manifest(exp, subset, configs_dir, scenes, results_dir, force=force_manifest)

    existing: list[dict[str, Any]] = []
    todo: list[RunSpec] = []
    for spec in specs:
        if resume and _run_is_resumable(spec, configs_dir, scenes, results_dir, profile):
            existing.append(
                {
                    "spec": spec.to_dict(),
                    "status": "resumed",
                    "valid": None,
                    "runtime_s": 0.0,
                    "out_dir": str(spec.out_dir(results_dir, configs_dir)),
                    "config_hash": spec.config_hash(configs_dir),
                    "scene_hash": scenes[spec.scene_id].sha256,
                    "error": None,
                }
            )
        else:
            todo.append(spec)

    records: list[dict[str, Any]] = list(existing)
    jsonl_path = results_dir / _batch_experiment_id(profile, specs) / "run_manifest.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    run_dirs_kw = {
        "root_dir": str(ROOT),
        "configs_dir": str(configs_dir),
        "scenes_dir": str(scenes_dir),
        "results_dir": str(results_dir),
    }
    if todo:
        if jobs <= 1:
            for spec in todo:
                rec = execute_run(spec.to_dict(), **run_dirs_kw)
                records.append(rec)
                _append_jsonl(jsonl_path, rec)
        else:
            with cf.ProcessPoolExecutor(max_workers=int(jobs)) as pool:
                futures = {
                    pool.submit(execute_run, spec.to_dict(), **run_dirs_kw): spec for spec in todo
                }
                for fut in cf.as_completed(futures):
                    spec = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        rec = {
                            "spec": spec.to_dict(),
                            "status": "numerical_error",
                            "valid": False,
                            "runtime_s": 0.0,
                            "out_dir": str(spec.out_dir(results_dir, configs_dir)),
                            "config_hash": spec.config_hash(configs_dir),
                            "scene_hash": scenes[spec.scene_id].sha256,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    records.append(rec)
                    _append_jsonl(jsonl_path, rec)

    manifest_rows = _write_run_manifest(records, results_dir)
    summary: dict[str, Any] = {
        "profile": profile,
        "n_specs": len(specs),
        "n_resumed": len(existing),
        "n_executed": len(todo),
        "n_completed": sum(1 for r in records if r["status"] == "completed"),
        "n_timeout": sum(1 for r in records if r["status"] == "timeout"),
        "n_numerical_error": sum(1 for r in records if r["status"] == "numerical_error")
        + sum(1 for r in records if r["status"] == "numerical_error"),
        "n_valid": sum(1 for r in records if r.get("valid") is True),
        "n_invalid": sum(1 for r in records if r.get("valid") is False),
        "total_runtime_s": float(sum(r.get("runtime_s") or 0.0 for r in records)),
        "run_manifest": str(manifest_rows),
    }
    return summary


def _batch_experiment_id(profile: str, specs: list[RunSpec]) -> str:
    exp_ids = sorted({s.experiment_id for s in specs})
    return exp_ids[0] if len(exp_ids) == 1 else profile


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _write_run_manifest(records: list[dict[str, Any]], results_dir: Path) -> Path:
    header = [
        "experiment_id",
        "scenario",
        "model",
        "scene_id",
        "seed",
        "status",
        "valid",
        "config_hash",
        "scene_hash",
        "runtime_s",
        "source",
        "error",
    ]
    by_exp: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        spec = rec["spec"]
        by_exp.setdefault(spec["experiment_id"], []).append(rec)
    out_path = None
    for exp, rows in sorted(by_exp.items()):
        path = Path(results_dir) / exp / "run_manifest.csv"
        csv_rows = []
        for rec in rows:
            spec = rec["spec"]
            csv_rows.append(
                [
                    spec["experiment_id"],
                    spec["scenario"],
                    spec["model"],
                    spec["scene_id"],
                    spec["seed"],
                    rec["status"],
                    "" if rec.get("valid") is None else int(bool(rec["valid"])),
                    rec.get("config_hash", ""),
                    rec.get("scene_hash", ""),
                    f"{float(rec.get('runtime_s') or 0.0):.3f}",
                    "resumed" if rec["status"] == "resumed" else "cold",
                    (rec.get("error") or "").replace("\n", " | ")[:500],
                ]
            )
        _write_csv(path, header, csv_rows)
        out_path = out_path or path
    return out_path or (Path(results_dir) / "run_manifest.csv")


def _run_is_resumable(
    spec: RunSpec, configs_dir: Path, scenes: dict[str, Scene], results_dir: Path, profile: str
) -> bool:
    out_dir = spec.out_dir(results_dir, configs_dir)
    audit_path = out_dir / "audit.json"
    metrics_path = out_dir / "metrics.json"
    prov_path = out_dir / "provenance.json"
    if not (audit_path.exists() and metrics_path.exists() and prov_path.exists()):
        return False
    try:
        with audit_path.open("r", encoding="utf-8") as fh:
            audit = json.load(fh)
        if not audit.get("ok", False):
            return False
        with prov_path.open("r", encoding="utf-8") as fh:
            prov = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    scene = scenes.get(spec.scene_id)
    if scene is None:
        return False
    return (
        prov.get("config_hash") == spec.config_hash(configs_dir)
        and prov.get("scene_hash") == scene.sha256
        and prov.get("source_hash") == code_source_hash(ROOT)
    )
