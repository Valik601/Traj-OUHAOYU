"""配置模型：YAML ↔ 冻结 dataclass ↔ canonical JSON/hash。

配置哈希用于结果目录与 ``--resume`` 判据（CONTRACTS.md §3）。
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"

MODEL_NAMES = ("direct", "vo", "rvo", "orca", "orca_release", "direct_release")
MODEL_BASE = {
    "direct": "direct",
    "vo": "vo",
    "rvo": "rvo",
    "orca": "orca",
    "orca_release": "orca",
    "direct_release": "direct",
}
RELEASE_MODELS = frozenset({"orca_release", "direct_release"})
SCENARIO_MODELS = {
    "A": ("direct", "vo", "rvo", "orca"),
    "B": ("direct", "vo", "rvo", "orca", "orca_release", "direct_release"),
}


class ConfigError(ValueError):
    """配置非法或无法解析。"""


@dataclass(frozen=True)
class PerturbationConfig:
    mode: str = "tangential_gaussian"
    max_angle_deg: float = 2.0
    resample_max: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "uav_antipodal_v2"
    observation_time: float = 120.0
    dt: float = 0.1


@dataclass(frozen=True)
class SimulationConfig:
    n_agents: int = 100
    radius: float = 0.5
    d_safe: float = 1.0
    d_plan: float = 1.0
    init_sphere_radius: float = 20.0
    v_max: float = 2.0
    goal_tolerance: float = 0.3
    center_tolerance: float = 0.1
    initial_velocity: str = "zero"
    neighbor_mode: str = "all"
    neighbor_range: float | None = None
    perturbation: PerturbationConfig = PerturbationConfig()


@dataclass(frozen=True)
class CandidateConfig:
    directions: int = 128
    speed_layers: tuple[float, ...] = (0.5, 1.0, 2.0)
    include_zero: bool = True
    include_current: bool = True
    include_pref: bool = True


@dataclass(frozen=True)
class TimeHorizonConfig:
    tau: float = 5.0


@dataclass(frozen=True)
class PlannerConfig:
    candidate: CandidateConfig = CandidateConfig()
    vo: TimeHorizonConfig = TimeHorizonConfig()
    rvo: TimeHorizonConfig = TimeHorizonConfig()
    orca: TimeHorizonConfig = TimeHorizonConfig()
    backend: str = "auto"


@dataclass(frozen=True)
class SchedulingConfig:
    delta: float = 0.75
    order: str = "random_stream_0"


@dataclass(frozen=True)
class MetricsConfig:
    eps_distance: float = 1.0e-8
    stall_window: float = 5.0
    stall_path_threshold: float = 0.1
    stall_waypoint_threshold: float = 0.1
    bootstrap_resamples: int = 10000
    bootstrap_seed: int = 20261030


@dataclass(frozen=True)
class ScenesConfig:
    base_seed: int = 1000
    n_formal_scenes: int = 20
    development_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    formal_seeds: tuple[int, ...] = tuple(range(1000, 1020))
    sensitivity_seeds: tuple[int, ...] = (1000, 1001, 1002, 1003, 1004)


@dataclass(frozen=True)
class ModelsConfig:
    A: tuple[str, ...] = SCENARIO_MODELS["A"]
    B: tuple[str, ...] = SCENARIO_MODELS["B"]


@dataclass(frozen=True)
class Config:
    schema_version: int = 1
    experiment: ExperimentConfig = ExperimentConfig()
    simulation: SimulationConfig = SimulationConfig()
    planners: PlannerConfig = PlannerConfig()
    scheduling: SchedulingConfig = SchedulingConfig()
    metrics: MetricsConfig = MetricsConfig()
    scenes: ScenesConfig = ScenesConfig()
    models: ModelsConfig = ModelsConfig()

    # ---- 派生量 -------------------------------------------------------------
    @property
    def n_agents(self) -> int:
        return self.simulation.n_agents

    @property
    def dt(self) -> float:
        return self.experiment.dt

    @property
    def observation_time(self) -> float:
        return self.experiment.observation_time

    @property
    def n_steps(self) -> int:
        return int(round(self.experiment.observation_time / self.experiment.dt))

    @property
    def d_eval(self) -> float:
        """正式数值事件阈值 ``d_safe - eps_distance``（handoff §2.2）。"""
        return self.simulation.d_safe - self.metrics.eps_distance

    @property
    def neighbor_mode(self) -> str:
        return self.simulation.neighbor_mode

    # ---- 序列化 -------------------------------------------------------------
    def to_json_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_json_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    # ---- 反序列化 -----------------------------------------------------------
    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "Config":
        try:
            sim_d = dict(d.get("simulation", {}))
            pert = PerturbationConfig(**sim_d.pop("perturbation", {}))
            planners_d = dict(d.get("planners", {}))
            scenes_d = dict(d.get("scenes", {}))
            for key in ("development_seeds", "formal_seeds", "sensitivity_seeds"):
                if key in scenes_d:
                    scenes_d[key] = tuple(int(v) for v in scenes_d[key])
            models_d = dict(d.get("models", {}))
            for key in ("A", "B"):
                if key in models_d:
                    models_d[key] = tuple(str(v) for v in models_d[key])
            cand_d = dict(planners_d.get("candidate", {})) if "candidate" in planners_d else {}
            if "speed_layers" in cand_d:
                cand_d["speed_layers"] = tuple(float(v) for v in cand_d["speed_layers"])
            cfg = cls(
                schema_version=int(d.get("schema_version", 1)),
                experiment=ExperimentConfig(**d.get("experiment", {})),
                simulation=SimulationConfig(**sim_d, perturbation=pert),
                planners=PlannerConfig(
                    candidate=CandidateConfig(**cand_d),
                    vo=TimeHorizonConfig(**planners_d.get("vo", {})),
                    rvo=TimeHorizonConfig(**planners_d.get("rvo", {})),
                    orca=TimeHorizonConfig(**planners_d.get("orca", {})),
                    backend=planners_d.get("backend", "auto"),
                ),
                scheduling=SchedulingConfig(**d.get("scheduling", {})),
                metrics=MetricsConfig(**d.get("metrics", {})),
                scenes=ScenesConfig(**scenes_d),
                models=ModelsConfig(**models_d),
            )
        except TypeError as exc:  # 未知键/类型错误
            raise ConfigError(f"invalid config structure: {exc}") from exc
        return cfg


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """读取 YAML；支持顶层 ``extends: <相对路径>`` 继承（用于 profile 配置）。"""
    p = Path(path).resolve()
    if p in seen:
        raise ConfigError(f"config extends cycle at {p}")
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {p}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = p.parent / parent_path
    base = _load_raw(parent_path, seen | {p})
    return _deep_merge(base, raw)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    cfg = Config.from_json_dict(_load_raw(Path(path)))
    errors = validate_config(cfg)
    if errors:
        raise ConfigError("; ".join(errors))
    return cfg


def _set_dotted(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = d
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ConfigError(f"unknown override path: {path}")
        node = node[part]
    if parts[-1] not in node:
        raise ConfigError(f"unknown override path: {path}")
    node[parts[-1]] = value


def apply_overrides(cfg: Config, overrides: dict[str, Any] | None) -> Config:
    """按点号路径覆盖配置（敏感性实验用），返回新对象。"""
    if not overrides:
        return cfg
    d = copy.deepcopy(cfg.to_json_dict())
    for path, value in overrides.items():
        if "." not in path:
            raise ConfigError(f"override must be dotted path: {path}")
        _set_dotted(d, path, value)
    new_cfg = Config.from_json_dict(d)
    errors = validate_config(new_cfg)
    if errors:
        raise ConfigError("; ".join(errors))
    return new_cfg


def validate_config(cfg: Config) -> list[str]:
    """返回错误信息列表；空列表表示合法。"""
    errs: list[str] = []
    exp, sim, pl, sch, met, sc, mdl = (
        cfg.experiment,
        cfg.simulation,
        cfg.planners,
        cfg.scheduling,
        cfg.metrics,
        cfg.scenes,
        cfg.models,
    )
    if cfg.schema_version != 1:
        errs.append(f"unsupported schema_version: {cfg.schema_version}")
    if exp.observation_time <= 0:
        errs.append("observation_time must be positive")
    if exp.dt <= 0:
        errs.append("dt must be positive")
    ratio = exp.observation_time / exp.dt
    if abs(ratio - round(ratio)) > 1e-9:
        errs.append("observation_time / dt must be an integer number of steps")
    if sim.n_agents < 2:
        errs.append("n_agents must be >= 2")
    if sim.radius <= 0:
        errs.append("radius must be positive")
    if abs(sim.d_safe - 2.0 * sim.radius) > 1e-12:
        errs.append("d_safe must equal 2*radius (sphere sum)")
    if sim.d_plan <= 0:
        errs.append("d_plan must be positive")
    if sim.d_plan < 0.5 * sim.d_safe:
        errs.append("d_plan must be >= 0.5*d_safe")
    if sim.init_sphere_radius <= 0:
        errs.append("init_sphere_radius must be positive")
    if sim.v_max <= 0:
        errs.append("v_max must be positive")
    if sim.goal_tolerance <= 0:
        errs.append("goal_tolerance must be positive")
    if sim.center_tolerance <= 0:
        errs.append("center_tolerance must be positive")
    if sim.center_tolerance >= 0.5 * sim.d_safe:
        errs.append("center_tolerance must be < d_safe/2 (single-agent capacity argument)")
    if sim.initial_velocity != "zero":
        errs.append("only initial_velocity=zero is supported in the main protocol")
    if sim.neighbor_mode not in ("all", "range"):
        errs.append("neighbor_mode must be 'all' or 'range'")
    if sim.neighbor_mode == "range":
        if sim.neighbor_range is None or sim.neighbor_range <= 0:
            errs.append("neighbor_range must be positive in range mode")
    elif sim.neighbor_range is not None:
        errs.append("neighbor_range must be null in all mode")
    if sim.perturbation.mode not in ("none", "tangential_gaussian"):
        errs.append("perturbation.mode must be none or tangential_gaussian")
    if not (0.0 <= sim.perturbation.max_angle_deg <= 45.0):
        errs.append("perturbation.max_angle_deg out of [0,45]")
    if sim.perturbation.resample_max < 1:
        errs.append("perturbation.resample_max must be >= 1")

    if pl.candidate.directions <= 0:
        errs.append("candidate.directions must be positive")
    if not pl.candidate.speed_layers:
        errs.append("candidate.speed_layers must be non-empty")
    for s in pl.candidate.speed_layers:
        if s <= 0:
            errs.append("candidate.speed_layers must be positive")
    for name, tau_cfg in (("vo", pl.vo), ("rvo", pl.rvo), ("orca", pl.orca)):
        if tau_cfg.tau <= 0:
            errs.append(f"planners.{name}.tau must be positive")
    if pl.backend not in ("auto", "python", "numba"):
        errs.append("planners.backend must be auto|python|numba")

    if sch.delta < 0:
        errs.append("scheduling.delta must be non-negative")
    if sch.order not in ("fixed_id", "random_stream_0", "random_stream_1"):
        errs.append("scheduling.order must be fixed_id|random_stream_0|random_stream_1")

    if not (0.0 < met.eps_distance < 1e-3):
        errs.append("metrics.eps_distance must be in (0,1e-3)")
    if met.stall_window <= exp.dt:
        errs.append("metrics.stall_window must exceed dt")
    if met.bootstrap_resamples < 100:
        errs.append("metrics.bootstrap_resamples must be >= 100")
    if met.bootstrap_seed < 0:
        errs.append("metrics.bootstrap_seed must be non-negative")

    if sc.n_formal_scenes != len(sc.formal_seeds):
        errs.append("n_formal_scenes must equal len(formal_seeds)")
    for name in ("development_seeds", "formal_seeds", "sensitivity_seeds"):
        seeds = getattr(sc, name)
        if len(set(seeds)) != len(seeds):
            errs.append(f"{name} must be unique")
    if len(sc.formal_seeds) < 1:
        errs.append("formal_seeds must be non-empty")
    if len(sc.sensitivity_seeds) < 1:
        errs.append("sensitivity_seeds must be non-empty")

    for scen, models in (("A", mdl.A), ("B", mdl.B)):
        for m in models:
            if m not in MODEL_NAMES:
                errs.append(f"unknown model {m!r} in models.{scen}")
        if scen == "A" and any(m in RELEASE_MODELS for m in models):
            errs.append("Scenario A must not include release models")
        if scen == "B" and set(models) != set(SCENARIO_MODELS["B"]):
            errs.append("Scenario B must include exactly the 6 protocol models")
    return errs
