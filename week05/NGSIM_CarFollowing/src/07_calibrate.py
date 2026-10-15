"""Optional in-sample parameter calibration for the car-following models.

This stage deliberately reuses :mod:`05_simulate` for the external measured
leader and numerical integration.  Consequently, calibration and the main
comparison pipeline have exactly the same collision and numerical-failure
semantics.

The default input is the one ``trajectory_clean.csv`` selected and prepared by
stages 02--03.  Optimising and reporting error on that same event is an
*in-sample demonstration*, not an independent test of generalisation.  The
warning is recorded in both output artifacts and emitted to the console.
Stage 07 remains optional: ``run_pipeline.py`` excludes it from the default
step range and runs it only when requested with ``--start-step 7``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import differential_evolution, least_squares
import yaml


LOGGER = logging.getLogger("ngsim.calibrate")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_NAMES = ("OVM", "FVDM", "IDM")

IN_SAMPLE_NOTICE = (
    "IN-SAMPLE DEMONSTRATION ONLY: parameters are calibrated and evaluated on "
    "the current selected case (outputs/04_prepared/trajectory_clean.csv). "
    "These errors are not independent held-out or generalization results."
)

PARAMETER_ORDER: dict[str, tuple[str, ...]] = {
    "OVM": ("kappa", "v_max", "s_c", "ell"),
    "FVDM": ("kappa", "lambda", "v_max", "s_c", "ell"),
    "IDM": ("v0", "T", "s0", "a", "b", "delta"),
}

# Physically interpretable search ranges, deliberately narrower than the broad
# equation-level validation limits in 04_car_following_models.py.  Every bound
# can be overridden under calibration.parameter_bounds (or calibration.bounds).
DEFAULT_PARAMETER_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "OVM": {
        "kappa": (0.05, 3.0),       # 1/s
        "v_max": (5.0, 50.0),       # m/s
        "s_c": (0.0, 100.0),        # m
        "ell": (0.5, 50.0),         # m
    },
    "FVDM": {
        "kappa": (0.05, 3.0),       # 1/s
        "lambda": (0.0, 3.0),       # 1/s
        "v_max": (5.0, 50.0),       # m/s
        "s_c": (0.0, 100.0),        # m
        "ell": (0.5, 50.0),         # m
    },
    "IDM": {
        "v0": (5.0, 50.0),          # m/s
        "T": (0.1, 5.0),            # s
        "s0": (0.0, 10.0),          # m
        "a": (0.1, 5.0),            # m/s^2
        "b": (0.1, 8.0),            # m/s^2
        "delta": (1.0, 8.0),        # dimensionless
    },
}

DEFAULT_DE_OPTIONS: dict[str, Any] = {
    # These intentionally small defaults make a missing calibration section a
    # runnable demonstration.  Increase them for a serious calibration run.
    "maxiter": 8,
    "popsize": 4,
    "tol": 0.01,
    "atol": 0.0,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "polish": False,
    "updating": "immediate",
    "workers": 1,
}

DEFAULT_LS_OPTIONS: dict[str, Any] = {
    "max_nfev": 200,
    "ftol": 1.0e-8,
    "xtol": 1.0e-8,
    "gtol": 1.0e-8,
    "method": "trf",
    "loss": "linear",
    "verbose": 0,
}


def _load_simulation_module() -> ModuleType:
    """Load the numbered simulation script independently of the working directory."""

    module_path = SCRIPT_DIR / "05_simulate.py"
    module_name = "ngsim_car_following_simulation"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load simulation functions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SIMULATION = _load_simulation_module()
MODELS = SIMULATION.MODELS


def _case_insensitive_get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    folded = key.casefold()
    for candidate, value in mapping.items():
        if str(candidate).casefold() == folded:
            return value
    return default


def _as_mapping(value: Any, *, name: str, allow_none: bool = False) -> Mapping[str, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not boolean")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real number")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not boolean")
    number = _finite_float(value, name=name)
    result = int(number)
    minimum = 0 if allow_zero else 1
    if result != number or result < minimum:
        comparison = ">= 0" if allow_zero else ">= 1"
        raise ValueError(f"{name} must be an integer {comparison}")
    return result


def _plain_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_config_path(config_path: str | Path | None) -> Path:
    candidate = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def load_config(config_path: str | Path | None) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_config_path(config_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration file must contain a YAML mapping")
    return payload, resolved


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _normalise_optimizer_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("calibration optimizer must be a non-empty string")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "de": "differential_evolution",
        "differential_evolution": "differential_evolution",
        "ls": "least_squares",
        "least_square": "least_squares",
        "least_squares": "least_squares",
    }
    if normalized not in aliases:
        raise ValueError(
            "calibration optimizer must be 'differential_evolution' or 'least_squares'"
        )
    return aliases[normalized]


def _first_not_none(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def build_calibration_settings(
    config: Mapping[str, Any],
    *,
    optimizer_override: str | None = None,
) -> dict[str, Any]:
    """Return validated settings, including useful defaults when absent."""

    calibration = _as_mapping(
        _case_insensitive_get(config, "calibration", {}),
        name="calibration",
        allow_none=True,
    )
    objective = _as_mapping(
        _case_insensitive_get(calibration, "objective", {}),
        name="calibration.objective",
        allow_none=True,
    )
    weights = _as_mapping(
        _case_insensitive_get(objective, "weights", _case_insensitive_get(calibration, "weights", {})),
        name="calibration objective weights",
        allow_none=True,
    )
    penalties = _as_mapping(
        _case_insensitive_get(objective, "penalties", _case_insensitive_get(calibration, "penalties", {})),
        name="calibration objective penalties",
        allow_none=True,
    )
    normalization = _as_mapping(
        _case_insensitive_get(objective, "normalization", _case_insensitive_get(calibration, "normalization", {})),
        name="calibration normalization",
        allow_none=True,
    )

    optimizer_value = _case_insensitive_get(calibration, "optimizer", None)
    inline_optimizer_options: Mapping[str, Any] = {}
    if isinstance(optimizer_value, Mapping):
        inline_optimizer_options = optimizer_value
        optimizer_value = _first_not_none(
            _case_insensitive_get(inline_optimizer_options, "name", None),
            _case_insensitive_get(inline_optimizer_options, "optimizer", None),
            _case_insensitive_get(inline_optimizer_options, "method", None),
        )
    optimizer_name = _normalise_optimizer_name(
        _first_not_none(
            optimizer_override,
            optimizer_value,
            _case_insensitive_get(calibration, "method", None),
            default="differential_evolution",
        )
    )

    position_weight = _finite_float(
        _first_not_none(
            _case_insensitive_get(objective, "position_weight", None),
            _case_insensitive_get(weights, "position", None),
            _case_insensitive_get(calibration, "position_weight", None),
            default=0.5,
        ),
        name="calibration position weight",
    )
    speed_weight = _finite_float(
        _first_not_none(
            _case_insensitive_get(objective, "speed_weight", None),
            _case_insensitive_get(weights, "speed", None),
            _case_insensitive_get(calibration, "speed_weight", None),
            default=0.5,
        ),
        name="calibration speed weight",
    )
    if position_weight < 0.0 or speed_weight < 0.0 or position_weight + speed_weight <= 0.0:
        raise ValueError("position and speed weights must be non-negative and not both zero")
    requested_weights = {"position": position_weight, "speed": speed_weight}
    weight_sum = position_weight + speed_weight
    effective_weights = {
        "position": position_weight / weight_sum,
        "speed": speed_weight / weight_sum,
    }

    def penalty(name: str, aliases: tuple[str, ...], default: float) -> float:
        candidates = [_case_insensitive_get(penalties, key, None) for key in aliases]
        candidates += [_case_insensitive_get(objective, key, None) for key in aliases]
        candidates += [_case_insensitive_get(calibration, key, None) for key in aliases]
        result = _finite_float(
            _first_not_none(*candidates, default=default),
            name=f"calibration {name}",
        )
        if result < 0.0:
            raise ValueError(f"calibration {name} must be non-negative")
        return result

    failure_penalty = penalty("failure penalty", ("failure", "failure_penalty"), 5.0)
    collision_penalty = penalty("collision penalty", ("collision", "collision_penalty"), 5.0)
    missing_penalty = penalty(
        "missing-fraction penalty",
        ("missing_fraction", "missing_fraction_penalty", "coverage_penalty"),
        5.0,
    )
    invalid_penalty = penalty(
        "invalid-candidate penalty",
        ("invalid_candidate", "invalid_candidate_penalty"),
        50.0,
    )

    def optional_scale(name: str) -> float | None:
        raw = _case_insensitive_get(normalization, name, None)
        if raw is None or (isinstance(raw, str) and raw.strip().casefold() == "auto"):
            return None
        result = _finite_float(raw, name=f"calibration.normalization.{name}")
        if result <= 0.0:
            raise ValueError(f"calibration.normalization.{name} must be positive or 'auto'")
        return result

    minimum_position_scale = _finite_float(
        _case_insensitive_get(normalization, "minimum_position_scale_m", 1.0),
        name="calibration.normalization.minimum_position_scale_m",
    )
    minimum_speed_scale = _finite_float(
        _case_insensitive_get(normalization, "minimum_speed_scale_m_s", 1.0),
        name="calibration.normalization.minimum_speed_scale_m_s",
    )
    if minimum_position_scale <= 0.0 or minimum_speed_scale <= 0.0:
        raise ValueError("minimum normalization scales must be positive")

    project = _as_mapping(
        _case_insensitive_get(config, "project", {}), name="project", allow_none=True
    )
    seed = _positive_int(
        _first_not_none(
            _case_insensitive_get(calibration, "random_seed", None),
            _case_insensitive_get(calibration, "seed", None),
            _case_insensitive_get(project, "random_seed", None),
            default=20261015,
        ),
        name="calibration random seed",
        allow_zero=True,
    )

    section_options = _as_mapping(
        _case_insensitive_get(calibration, optimizer_name, {}),
        name=f"calibration.{optimizer_name}",
        allow_none=True,
    )
    generic_options = _as_mapping(
        _case_insensitive_get(calibration, "optimizer_options", {}),
        name="calibration.optimizer_options",
        allow_none=True,
    )
    nested_inline_options = _as_mapping(
        _case_insensitive_get(inline_optimizer_options, "options", {}),
        name="calibration.optimizer.options",
        allow_none=True,
    )

    if optimizer_name == "differential_evolution":
        options = dict(DEFAULT_DE_OPTIONS)
        allowed = set(options)
    else:
        options = dict(DEFAULT_LS_OPTIONS)
        allowed = set(options)
    for source in (calibration, generic_options, inline_optimizer_options, nested_inline_options, section_options):
        for key in allowed:
            value = _case_insensitive_get(source, key, None)
            if value is not None:
                options[key] = value

    if optimizer_name == "differential_evolution":
        options["maxiter"] = _positive_int(options["maxiter"], name="differential_evolution.maxiter", allow_zero=True)
        options["popsize"] = _positive_int(options["popsize"], name="differential_evolution.popsize")
        options["tol"] = _finite_float(options["tol"], name="differential_evolution.tol")
        options["atol"] = _finite_float(options["atol"], name="differential_evolution.atol")
        options["recombination"] = _finite_float(
            options["recombination"], name="differential_evolution.recombination"
        )
        options["polish"] = _plain_bool(options["polish"], name="differential_evolution.polish")
        options["workers"] = _positive_int(options["workers"], name="differential_evolution.workers")
        if options["workers"] != 1:
            raise ValueError(
                "differential_evolution.workers must be 1; deterministic calibration uses an in-process objective"
            )
        if options["updating"] not in {"immediate", "deferred"}:
            raise ValueError("differential_evolution.updating must be 'immediate' or 'deferred'")
        if isinstance(options["mutation"], list):
            options["mutation"] = tuple(options["mutation"])
        if options["tol"] < 0.0 or options["atol"] < 0.0:
            raise ValueError("differential_evolution tolerances must be non-negative")
        if not 0.0 <= options["recombination"] <= 1.0:
            raise ValueError("differential_evolution.recombination must be in [0, 1]")
    else:
        options["max_nfev"] = _positive_int(options["max_nfev"], name="least_squares.max_nfev")
        for key in ("ftol", "xtol", "gtol"):
            options[key] = _finite_float(options[key], name=f"least_squares.{key}")
            if options[key] <= 0.0:
                raise ValueError(f"least_squares.{key} must be positive")
        options["verbose"] = _positive_int(
            options["verbose"], name="least_squares.verbose", allow_zero=True
        )
        if options["verbose"] > 2:
            raise ValueError("least_squares.verbose must be 0, 1, or 2")
        if options["method"] not in {"trf", "dogbox"}:
            raise ValueError("least_squares.method must be 'trf' or 'dogbox' for bounded calibration")

    return {
        "optimizer": optimizer_name,
        "optimizer_options": options,
        "random_seed": seed,
        "requested_weights": requested_weights,
        "weights": effective_weights,
        "penalties": {
            "failure": failure_penalty,
            "collision": collision_penalty,
            "missing_fraction": missing_penalty,
            "invalid_candidate": invalid_penalty,
        },
        "normalization": {
            "position_scale_m": optional_scale("position_scale_m"),
            "speed_scale_m_s": optional_scale("speed_scale_m_s"),
            "minimum_position_scale_m": minimum_position_scale,
            "minimum_speed_scale_m_s": minimum_speed_scale,
            "strategy": "observed range with a positive floor when scale is auto",
        },
    }


def _configured_model_parameters(config: Mapping[str, Any], model_name: str) -> dict[str, float]:
    models = _as_mapping(_case_insensitive_get(config, "models", None), name="models")
    supplied = _case_insensitive_get(models, model_name, None)
    if not isinstance(supplied, Mapping):
        raise ValueError(f"configuration is missing models.{model_name}")
    return MODELS.validate_model_parameters(model_name, supplied)


def _parse_bound(value: Any, *, name: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        lower = _first_not_none(
            _case_insensitive_get(value, "lower", None),
            _case_insensitive_get(value, "min", None),
        )
        upper = _first_not_none(
            _case_insensitive_get(value, "upper", None),
            _case_insensitive_get(value, "max", None),
        )
        if lower is None or upper is None:
            raise ValueError(f"{name} mapping must provide lower/min and upper/max")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        lower, upper = value
    else:
        raise ValueError(f"{name} must be [lower, upper] or a lower/upper mapping")
    lower_value = _finite_float(lower, name=f"{name}.lower")
    upper_value = _finite_float(upper, name=f"{name}.upper")
    if lower_value >= upper_value:
        raise ValueError(f"{name} lower bound must be less than its upper bound")
    return lower_value, upper_value


def get_parameter_bounds(
    config: Mapping[str, Any], model_name: str
) -> tuple[list[tuple[float, float]], dict[str, list[float]]]:
    """Return bounds in optimizer order and a JSON-friendly named mapping."""

    calibration = _as_mapping(
        _case_insensitive_get(config, "calibration", {}),
        name="calibration",
        allow_none=True,
    )
    configured = _first_not_none(
        _case_insensitive_get(calibration, "parameter_bounds", None),
        _case_insensitive_get(calibration, "bounds", None),
        default={},
    )
    configured = _as_mapping(configured, name="calibration parameter bounds", allow_none=True)
    model_bounds = _case_insensitive_get(configured, model_name, None)
    if model_bounds is None and any(
        _case_insensitive_get(configured, parameter, None) is not None
        for parameter in PARAMETER_ORDER[model_name]
    ):
        model_bounds = configured
    model_bounds = _as_mapping(
        model_bounds, name=f"calibration bounds for {model_name}", allow_none=True
    )

    unknown = sorted(
        str(key)
        for key in model_bounds
        if str(key).casefold() not in {name.casefold() for name in PARAMETER_ORDER[model_name]}
    )
    if unknown:
        raise ValueError(f"unknown {model_name} calibration bound(s): {', '.join(unknown)}")

    ordered: list[tuple[float, float]] = []
    named: dict[str, list[float]] = {}
    for parameter in PARAMETER_ORDER[model_name]:
        raw = _case_insensitive_get(model_bounds, parameter, DEFAULT_PARAMETER_BOUNDS[model_name][parameter])
        lower, upper = _parse_bound(raw, name=f"{model_name}.{parameter}")
        # Equation-level validation is the final authority on legal endpoints.
        MODELS._validate_bound(parameter, lower)
        MODELS._validate_bound(parameter, upper)
        ordered.append((lower, upper))
        named[parameter] = [lower, upper]
    return ordered, named


def _simulation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    simulation = _as_mapping(
        _case_insensitive_get(config, "simulation", {}),
        name="simulation",
        allow_none=True,
    )
    data = _as_mapping(
        _case_insensitive_get(config, "data", {}), name="data", allow_none=True
    )
    return {
        "time_step_s": _finite_float(
            _case_insensitive_get(simulation, "time_step_s", 0.1),
            name="simulation.time_step_s",
        ),
        "collision_gap_m": _finite_float(
            _case_insensitive_get(simulation, "collision_gap_m", 0.0),
            name="simulation.collision_gap_m",
        ),
        "maximum_abs_acceleration_m_s2": _finite_float(
            _case_insensitive_get(simulation, "maximum_abs_acceleration_m_s2", 20.0),
            name="simulation.maximum_abs_acceleration_m_s2",
        ),
        "stop_on_unreasonable_acceleration": _plain_bool(
            _case_insensitive_get(simulation, "stop_on_unreasonable_acceleration", True),
            name="simulation.stop_on_unreasonable_acceleration",
        ),
        "time_tolerance_s": _finite_float(
            _case_insensitive_get(data, "time_grid_tolerance_s", 1.0e-6),
            name="data.time_grid_tolerance_s",
        ),
    }


def _normalization_scales(
    trajectory: pd.DataFrame, settings: Mapping[str, Any]
) -> dict[str, float]:
    normalization = settings["normalization"]
    observed_position_range = float(np.ptp(trajectory["follower_x_true"].to_numpy(dtype=float)))
    observed_speed_range = float(np.ptp(trajectory["follower_v_true"].to_numpy(dtype=float)))
    position_scale = normalization["position_scale_m"]
    speed_scale = normalization["speed_scale_m_s"]
    if position_scale is None:
        position_scale = max(observed_position_range, normalization["minimum_position_scale_m"])
    if speed_scale is None:
        speed_scale = max(observed_speed_range, normalization["minimum_speed_scale_m_s"])
    return {
        "position_scale_m": float(position_scale),
        "speed_scale_m_s": float(speed_scale),
        "observed_position_range_m": observed_position_range,
        "observed_speed_range_m_s": observed_speed_range,
    }


def _parameters_from_vector(
    model_name: str,
    values: Sequence[float],
    base_parameters: Mapping[str, float],
) -> dict[str, float]:
    names = PARAMETER_ORDER[model_name]
    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(names),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{model_name} candidate must contain {len(names)} finite parameters")
    candidate = {name: float(value) for name, value in zip(names, vector)}
    # Preserve IDM numerical-safety settings; they are implementation guards,
    # not behavioural parameters in this calibration.
    for optional_name in ("minimum_gap_epsilon_m", "maximum_braking_m_s2"):
        if optional_name in base_parameters:
            candidate[optional_name] = float(base_parameters[optional_name])
    return MODELS.validate_model_parameters(model_name, candidate)


def evaluate_parameters(
    trajectory: pd.DataFrame,
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    scales: Mapping[str, float],
    simulation_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Simulate one candidate and construct its fixed-length residual vector."""

    simulation, status = SIMULATION.simulate_model(
        trajectory,
        model_name,
        parameters,
        **simulation_settings,
    )
    sample_count = len(trajectory)
    written = min(len(simulation), sample_count)
    if written and not np.allclose(
        simulation["t"].to_numpy(dtype=float)[:written],
        trajectory["t"].to_numpy(dtype=float)[:written],
        rtol=0.0,
        atol=float(simulation_settings["time_tolerance_s"]),
    ):
        raise ValueError("simulation timestamps do not align with the prepared trajectory")

    valid = np.zeros(sample_count, dtype=bool)
    if written:
        local_valid = simulation["valid"].to_numpy(dtype=bool)[:written]
        local_valid &= np.isfinite(
            simulation.loc[: written - 1, ["x_sim", "v_sim"]].to_numpy(dtype=float)
        ).all(axis=1)
        # Only the common contiguous valid prefix is eligible for fit errors.
        invalid_indices = np.flatnonzero(~local_valid)
        if invalid_indices.size:
            local_valid[int(invalid_indices[0]) :] = False
        valid[:written] = local_valid

    valid_count = int(valid.sum())
    coverage = valid_count / sample_count
    position_errors = np.zeros(sample_count, dtype=float)
    speed_errors = np.zeros(sample_count, dtype=float)
    if valid_count:
        indices = np.flatnonzero(valid)
        position_errors[indices] = (
            simulation["x_sim"].to_numpy(dtype=float)[indices]
            - trajectory["follower_x_true"].to_numpy(dtype=float)[indices]
        )
        speed_errors[indices] = (
            simulation["v_sim"].to_numpy(dtype=float)[indices]
            - trajectory["follower_v_true"].to_numpy(dtype=float)[indices]
        )
        position_rmse = float(np.sqrt(np.mean(np.square(position_errors[indices]))))
        speed_rmse = float(np.sqrt(np.mean(np.square(speed_errors[indices]))))
        normalized_position_rmse = position_rmse / scales["position_scale_m"]
        normalized_speed_rmse = speed_rmse / scales["speed_scale_m_s"]
    else:
        position_rmse = speed_rmse = math.nan
        normalized_position_rmse = normalized_speed_rmse = math.nan

    weights = settings["weights"]
    denominator = math.sqrt(max(valid_count, 1))
    position_residuals = (
        math.sqrt(weights["position"])
        * position_errors
        / scales["position_scale_m"]
        / denominator
    )
    speed_residuals = (
        math.sqrt(weights["speed"])
        * speed_errors
        / scales["speed_scale_m_s"]
        / denominator
    )

    completed = bool(status.get("completed", False)) and valid_count == sample_count
    status_name = str(status.get("status", "unknown"))
    penalties = settings["penalties"]
    failure_term = 0.0 if completed else float(penalties["failure"])
    collision_term = float(penalties["collision"]) if status_name == "collision" else 0.0
    missing_term = float(penalties["missing_fraction"]) * (1.0 - coverage)
    residuals = np.concatenate(
        [
            position_residuals,
            speed_residuals,
            np.asarray([failure_term, collision_term, missing_term], dtype=float),
        ]
    )
    objective_value = float(np.linalg.norm(residuals))
    fit_objective = (
        float(
            math.sqrt(
                weights["position"] * normalized_position_rmse**2
                + weights["speed"] * normalized_speed_rmse**2
            )
        )
        if valid_count
        else math.nan
    )

    metrics = {
        "objective_value": objective_value,
        "fit_objective_on_valid_prefix": fit_objective,
        "position_rmse_m": position_rmse,
        "speed_rmse_m_s": speed_rmse,
        "normalized_position_rmse": normalized_position_rmse,
        "normalized_speed_rmse": normalized_speed_rmse,
        "valid_samples": valid_count,
        "requested_samples": sample_count,
        "coverage_fraction": coverage,
        "completed": completed,
        "status": status_name,
        "failure_reason": status.get("failure_reason"),
        "failure_time_s": status.get("failure_time_s"),
        "minimum_recorded_gap_m": status.get("minimum_recorded_gap_m"),
        "failure_penalty_component": failure_term,
        "collision_penalty_component": collision_term,
        "missing_fraction_penalty_component": missing_term,
    }
    return {
        "residuals": residuals,
        "metrics": metrics,
        "status": status,
        "simulation": simulation,
    }


class CalibrationObjective:
    """Stateful SciPy objective that records evaluation and invalid-candidate counts."""

    def __init__(
        self,
        trajectory: pd.DataFrame,
        model_name: str,
        base_parameters: Mapping[str, float],
        settings: Mapping[str, Any],
        scales: Mapping[str, float],
        simulation_settings: Mapping[str, Any],
    ) -> None:
        self.trajectory = trajectory
        self.model_name = model_name
        self.base_parameters = dict(base_parameters)
        self.settings = settings
        self.scales = scales
        self.simulation_settings = simulation_settings
        self.evaluation_count = 0
        self.invalid_candidate_count = 0
        self.last_invalid_reason: str | None = None

    @property
    def residual_length(self) -> int:
        return 2 * len(self.trajectory) + 3

    def residuals(self, values: Sequence[float]) -> np.ndarray:
        self.evaluation_count += 1
        try:
            parameters = _parameters_from_vector(
                self.model_name, values, self.base_parameters
            )
            return evaluate_parameters(
                self.trajectory,
                self.model_name,
                parameters,
                settings=self.settings,
                scales=self.scales,
                simulation_settings=self.simulation_settings,
            )["residuals"]
        except (FloatingPointError, OverflowError, TypeError, ValueError) as exc:
            self.invalid_candidate_count += 1
            self.last_invalid_reason = f"{type(exc).__name__}: {exc}"
            residuals = np.zeros(self.residual_length, dtype=float)
            residuals[-3:] = float(self.settings["penalties"]["invalid_candidate"])
            return residuals

    def scalar(self, values: Sequence[float]) -> float:
        return float(np.linalg.norm(self.residuals(values)))


def calibrate_model(
    trajectory: pd.DataFrame,
    model_name: str,
    config: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    scales: Mapping[str, float],
    simulation_settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate one model and return its detailed record and metrics row."""

    model_name = MODELS._canonical_model_name(model_name)
    base_parameters = _configured_model_parameters(config, model_name)
    bounds, named_bounds = get_parameter_bounds(config, model_name)
    parameter_names = PARAMETER_ORDER[model_name]
    configured_vector = np.asarray([base_parameters[name] for name in parameter_names], dtype=float)
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    start_vector = np.clip(configured_vector, lower, upper)

    baseline = evaluate_parameters(
        trajectory,
        model_name,
        base_parameters,
        settings=settings,
        scales=scales,
        simulation_settings=simulation_settings,
    )
    objective = CalibrationObjective(
        trajectory,
        model_name,
        base_parameters,
        settings,
        scales,
        simulation_settings,
    )

    started = time.perf_counter()
    optimizer_name = settings["optimizer"]
    optimizer_options = dict(settings["optimizer_options"])
    if optimizer_name == "differential_evolution":
        result = differential_evolution(
            objective.scalar,
            bounds,
            seed=int(settings["random_seed"]),
            x0=start_vector,
            **optimizer_options,
        )
    else:
        result = least_squares(
            objective.residuals,
            start_vector,
            bounds=(lower, upper),
            **optimizer_options,
        )
    elapsed = time.perf_counter() - started

    calibrated_parameters = _parameters_from_vector(
        model_name, result.x, base_parameters
    )
    final_evaluation = evaluate_parameters(
        trajectory,
        model_name,
        calibrated_parameters,
        settings=settings,
        scales=scales,
        simulation_settings=simulation_settings,
    )
    metrics = final_evaluation["metrics"]
    optimizer_record = {
        "method": optimizer_name,
        "options": optimizer_options,
        "success": bool(result.success),
        "message": str(result.message),
        "reported_objective": float(
            result.fun if np.ndim(result.fun) == 0 else np.linalg.norm(result.fun)
        ),
        "function_evaluations_reported_by_scipy": int(result.nfev),
        "objective_calls_observed": int(objective.evaluation_count),
        "invalid_candidate_calls": int(objective.invalid_candidate_count),
        "last_invalid_candidate_reason": objective.last_invalid_reason,
        "iterations": int(result.nit) if hasattr(result, "nit") else None,
        "elapsed_seconds": elapsed,
    }
    record = {
        "parameter_order": list(parameter_names),
        "parameter_bounds": named_bounds,
        "configured_initial_parameters": base_parameters,
        "optimizer_start_parameters": {
            name: float(value) for name, value in zip(parameter_names, start_vector)
        },
        "calibrated_parameters": calibrated_parameters,
        "baseline_metrics": baseline["metrics"],
        "calibrated_metrics": metrics,
        "optimizer": optimizer_record,
        "simulation_status": final_evaluation["status"],
    }
    row = {
        "model": model_name,
        "optimizer": optimizer_name,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "objective_value": metrics["objective_value"],
        "baseline_objective_value": baseline["metrics"]["objective_value"],
        "fit_objective_on_valid_prefix": metrics["fit_objective_on_valid_prefix"],
        "position_rmse_m": metrics["position_rmse_m"],
        "speed_rmse_m_s": metrics["speed_rmse_m_s"],
        "normalized_position_rmse": metrics["normalized_position_rmse"],
        "normalized_speed_rmse": metrics["normalized_speed_rmse"],
        "coverage_fraction": metrics["coverage_fraction"],
        "valid_samples": metrics["valid_samples"],
        "requested_samples": metrics["requested_samples"],
        "completed": metrics["completed"],
        "status": metrics["status"],
        "failure_reason": metrics["failure_reason"],
        "failure_time_s": metrics["failure_time_s"],
        "minimum_recorded_gap_m": metrics["minimum_recorded_gap_m"],
        "function_evaluations": int(result.nfev),
        "elapsed_seconds": elapsed,
        "calibration_scope": "in_sample_current_selected_case",
        "in_sample": True,
        "independent_test_case": False,
        "generalization_result": False,
        "scope_warning": IN_SAMPLE_NOTICE,
        "calibrated_parameters_json": json.dumps(
            calibrated_parameters, ensure_ascii=False, sort_keys=True
        ),
    }
    return record, row


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _replace_file_resiliently(temporary_name: str, destination: Path) -> None:
    """Rename a staged temporary file onto its destination.

    Stage 07 keeps its own copy of this helper because it deliberately avoids
    importing ``common.py``.  Windows antivirus/indexer agents can briefly lock a
    freshly written temporary file, which makes ``os.replace`` fail with
    ``ERROR_ACCESS_DENIED`` (5) or ``ERROR_SHARING_VIOLATION`` (32); the rename is
    retried with backoff and degrades to an in-place copy if the lock persists.
    """

    source = Path(temporary_name)
    delay = 0.05
    last_error: OSError | None = None
    for attempt in range(1, 13):
        try:
            os.replace(source, destination)
            if attempt > 1:
                LOGGER.info(
                    "Renamed %s after %d attempts (transient file lock)",
                    destination.name,
                    attempt,
                )
            return
        except OSError as error:
            retryable = getattr(error, "winerror", None) in (5, 32) or isinstance(
                error, PermissionError
            )
            if not retryable or isinstance(error, FileNotFoundError):
                raise
            last_error = error
            if attempt < 12:
                time.sleep(delay)
                delay = min(delay * 2.0, 1.0)
    LOGGER.warning(
        "Atomic rename of %s failed after 12 attempts (%s); overwriting in place instead",
        destination.name,
        last_error,
    )
    try:
        with source.open("rb") as reader, destination.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
    finally:
        try:
            source.unlink()
        except FileNotFoundError:
            pass


def _write_json_atomic(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_safe(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        _replace_file_resiliently(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False, float_format="%.10g")
        _replace_file_resiliently(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def run_calibration(
    config_path: str | Path | None = None,
    model: str = "all",
    *,
    optimizer_override: str | None = None,
) -> dict[str, Any]:
    """Run calibration and write JSON/CSV artifacts for one or all models."""

    config, resolved_config = load_config(config_path)
    canonical_choice = model if model == "all" else MODELS._canonical_model_name(model)
    selected_models = MODEL_NAMES if canonical_choice == "all" else (canonical_choice,)
    settings = build_calibration_settings(config, optimizer_override=optimizer_override)
    simulation_settings = _simulation_settings(config)
    project_root = resolved_config.parent

    paths = _as_mapping(
        _case_insensitive_get(config, "paths", {}), name="paths", allow_none=True
    )
    prepared_dir = _resolve_project_path(
        _case_insensitive_get(paths, "prepared_dir", "outputs/04_prepared"), project_root
    )
    calibration = _as_mapping(
        _case_insensitive_get(config, "calibration", {}),
        name="calibration",
        allow_none=True,
    )
    output_value = _first_not_none(
        _case_insensitive_get(paths, "calibration_dir", None),
        _case_insensitive_get(calibration, "output_dir", None),
        default="outputs/07_calibration",
    )
    output_dir = _resolve_project_path(output_value, project_root)
    trajectory_path = prepared_dir / "trajectory_clean.csv"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"prepared trajectory not found: {trajectory_path}")
    raw_trajectory = pd.read_csv(trajectory_path)
    trajectory = SIMULATION.validate_trajectory(
        raw_trajectory,
        time_step_s=simulation_settings["time_step_s"],
        time_tolerance_s=simulation_settings["time_tolerance_s"],
    )
    scales = _normalization_scales(trajectory, settings)

    cases_dir = _resolve_project_path(
        _case_insensitive_get(paths, "cases_dir", "outputs/03_cases"), project_root
    )
    selected_case_path = cases_dir / "selected_case.json"
    selected_case = _read_optional_json(selected_case_path)

    LOGGER.warning(IN_SAMPLE_NOTICE)
    LOGGER.info(
        "Calibration is optional stage 07 and is not part of the default run_pipeline.py stages 0-6."
    )
    LOGGER.info(
        "Optimizer=%s; models=%s; samples=%d",
        settings["optimizer"],
        ",".join(selected_models),
        len(trajectory),
    )

    model_records: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for model_name in selected_models:
        LOGGER.info("Calibrating %s ...", model_name)
        record, row = calibrate_model(
            trajectory,
            model_name,
            config,
            settings=settings,
            scales=scales,
            simulation_settings=simulation_settings,
        )
        model_records[model_name] = record
        metric_rows.append(row)
        LOGGER.info(
            "%s objective %.6g -> %.6g; status=%s; optimizer_success=%s",
            model_name,
            row["baseline_objective_value"],
            row["objective_value"],
            row["status"],
            row["optimizer_success"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = output_dir / "calibrated_parameters.json"
    metrics_path = output_dir / "calibration_metrics.csv"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_scope": "in_sample_current_selected_case",
            "in_sample": True,
            "independent_test_case": False,
            "generalization_result": False,
            "scope_warning": IN_SAMPLE_NOTICE,
            "pipeline_stage": 7,
            "pipeline_default_inclusion": False,
            "pipeline_note": (
                "optional stage; run_pipeline.py excludes step 7 by default and runs it "
                "only when requested with --start-step 7 / --end-step 7"
            ),
            "config_path": str(resolved_config),
            "config_sha256": _sha256(resolved_config),
            "source_trajectory": str(trajectory_path),
            "source_trajectory_sha256": _sha256(trajectory_path),
            "selected_case_path": str(selected_case_path) if selected_case_path.is_file() else None,
            "selected_case_sha256": _sha256(selected_case_path) if selected_case_path.is_file() else None,
            "selected_case": selected_case,
            "sample_count": int(len(trajectory)),
            "scipy_version": scipy.__version__,
        },
        "objective": {
            "definition": (
                "Euclidean norm of weighted position/speed errors normalized by observed "
                "ranges, plus explicit incomplete-run, collision, and missing-coverage terms"
            ),
            "validity_rule": "contiguous common valid prefix from t=0",
            "requested_weights": settings["requested_weights"],
            "effective_weights_normalized_to_sum_one": settings["weights"],
            "normalization": {**settings["normalization"], **scales},
            "penalties": settings["penalties"],
        },
        "optimizer": {
            "method": settings["optimizer"],
            "random_seed": settings["random_seed"],
            "options": settings["optimizer_options"],
        },
        "simulation": simulation_settings,
        "models": model_records,
        "outputs": {
            "calibrated_parameters": str(parameters_path),
            "calibration_metrics": str(metrics_path),
        },
    }
    _write_json_atomic(payload, parameters_path)
    _write_csv_atomic(pd.DataFrame(metric_rows), metrics_path)
    LOGGER.warning(IN_SAMPLE_NOTICE)
    LOGGER.info("Wrote calibrated parameters: %s", parameters_path)
    LOGGER.info("Wrote calibration metrics: %s", metrics_path)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally calibrate OVM/FVDM/IDM on the current prepared following case. "
            "Results are in-sample demonstrations, not held-out generalization estimates."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML configuration path; relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--model",
        choices=("OVM", "FVDM", "IDM", "all"),
        default="all",
        help="Model to calibrate (default: all).",
    )
    parser.add_argument(
        "--optimizer",
        choices=("differential_evolution", "least_squares"),
        default=None,
        help="Optional override for calibration.optimizer in the YAML file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = build_argument_parser().parse_args(argv)
    try:
        run_calibration(
            arguments.config,
            arguments.model,
            optimizer_override=arguments.optimizer,
        )
    except (FileNotFoundError, json.JSONDecodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        LOGGER.error("Calibration failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
