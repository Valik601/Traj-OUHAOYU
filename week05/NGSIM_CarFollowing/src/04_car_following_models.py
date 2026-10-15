"""Classical longitudinal car-following acceleration models.

All public acceleration functions expose the same interface::

    acceleration(gap, follower_speed, leader_speed, model_parameters)

Distances are in metres, speeds in metres per second, and returned
accelerations in metres per second squared.  Scalars and one-dimensional
array-like inputs are supported; NumPy broadcasting is limited to scalar-to-
vector broadcasting so that an accidentally supplied matrix cannot silently
produce a meaningless simulation.

The equations implemented here are the uncalibrated reference equations.  The
only numerical safeguard that changes an equation is IDM's finite emergency-
braking cap at a zero or very small positive gap.  A simulator must still
detect a non-positive gap as a collision before calling a model; the safeguard
exists so direct model evaluation cannot return ``NaN`` or ``-inf``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ArrayLike = float | int | list[float] | tuple[float, ...] | np.ndarray
AccelerationFunction = Callable[[ArrayLike, ArrayLike, ArrayLike, Mapping[str, Any]], float | np.ndarray]

MAX_STATE_SPEED_M_S = 100.0
MAX_STATE_GAP_M = 10_000.0
DEFAULT_IDM_GAP_EPSILON_M = 1.0e-3
DEFAULT_IDM_MAXIMUM_BRAKING_M_S2 = 50.0


_PARAMETER_BOUNDS: dict[str, tuple[float, float, bool, bool]] = {
    # name: (minimum, maximum, include_minimum, include_maximum)
    "kappa": (0.0, 10.0, False, True),
    "lambda": (0.0, 10.0, True, True),
    "v_max": (0.0, 100.0, False, True),
    "s_c": (0.0, 1_000.0, True, True),
    "ell": (0.0, 1_000.0, False, True),
    "v0": (0.0, 100.0, False, True),
    "T": (0.0, 10.0, True, True),
    "s0": (0.0, 100.0, True, True),
    "a": (0.0, 20.0, False, True),
    "b": (0.0, 20.0, False, True),
    "delta": (1.0, 10.0, True, True),
    "minimum_gap_epsilon_m": (0.0, 1.0, False, True),
    "maximum_braking_m_s2": (0.0, 1_000.0, False, True),
}

_REQUIRED_PARAMETERS: dict[str, tuple[str, ...]] = {
    "OVM": ("kappa", "v_max", "s_c", "ell"),
    "FVDM": ("kappa", "lambda", "v_max", "s_c", "ell"),
    "IDM": ("v0", "T", "s0", "a", "b", "delta"),
}

_OPTIONAL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "OVM": (),
    "FVDM": (),
    "IDM": ("minimum_gap_epsilon_m", "maximum_braking_m_s2"),
}


def _canonical_model_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise TypeError("model_name must be a non-empty string")
    canonical = model_name.strip().upper()
    if canonical not in _REQUIRED_PARAMETERS:
        choices = ", ".join(_REQUIRED_PARAMETERS)
        raise ValueError(f"unknown car-following model {model_name!r}; choose one of {choices}")
    return canonical


def _finite_real_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"parameter {name!r} must be a real scalar, not boolean")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise TypeError(f"parameter {name!r} must be a real scalar")
    number = float(raw)
    if not np.isfinite(number):
        raise ValueError(f"parameter {name!r} must be finite")
    return number


def _validate_bound(name: str, value: float) -> None:
    minimum, maximum, include_minimum, include_maximum = _PARAMETER_BOUNDS[name]
    lower_ok = value >= minimum if include_minimum else value > minimum
    upper_ok = value <= maximum if include_maximum else value < maximum
    if not (lower_ok and upper_ok):
        left = "[" if include_minimum else "("
        right = "]" if include_maximum else ")"
        raise ValueError(
            f"parameter {name!r}={value!r} is outside the supported range "
            f"{left}{minimum}, {maximum}{right}"
        )


def validate_model_parameters(model_name: str, model_parameters: Mapping[str, Any]) -> dict[str, float]:
    """Validate and normalize a model parameter mapping.

    Unknown keys are rejected to catch misspellings.  Returned values are plain
    Python ``float`` objects suitable for JSON metadata.  IDM's two numerical
    safety parameters are inserted when omitted, making the effective settings
    explicit in simulation status output.
    """

    canonical = _canonical_model_name(model_name)
    if not isinstance(model_parameters, Mapping):
        raise TypeError(f"{canonical} model_parameters must be a mapping")

    required = set(_REQUIRED_PARAMETERS[canonical])
    optional = set(_OPTIONAL_PARAMETERS[canonical])
    supplied = set(model_parameters)
    missing = sorted(required - supplied)
    unknown = sorted(supplied - required - optional)
    if missing:
        raise ValueError(f"{canonical} parameters missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{canonical} parameters contain unknown keys: {', '.join(unknown)}")

    normalized: dict[str, float] = {}
    for name in _REQUIRED_PARAMETERS[canonical]:
        number = _finite_real_scalar(model_parameters[name], name=name)
        _validate_bound(name, number)
        normalized[name] = number

    if canonical == "IDM":
        defaults = {
            "minimum_gap_epsilon_m": DEFAULT_IDM_GAP_EPSILON_M,
            "maximum_braking_m_s2": DEFAULT_IDM_MAXIMUM_BRAKING_M_S2,
        }
        for name in _OPTIONAL_PARAMETERS[canonical]:
            number = _finite_real_scalar(model_parameters.get(name, defaults[name]), name=name)
            _validate_bound(name, number)
            normalized[name] = number

    return normalized


def _numeric_state(value: ArrayLike, *, name: str) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if raw.ndim > 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array")
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    values = raw.astype(float, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validated_state(
    gap: ArrayLike,
    follower_speed: ArrayLike,
    leader_speed: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    gap_values = _numeric_state(gap, name="gap")
    follower_values = _numeric_state(follower_speed, name="follower_speed")
    leader_values = _numeric_state(leader_speed, name="leader_speed")
    scalar_result = gap_values.ndim == follower_values.ndim == leader_values.ndim == 0

    try:
        gap_values, follower_values, leader_values = np.broadcast_arrays(
            gap_values, follower_values, leader_values
        )
    except ValueError as exc:
        raise ValueError("gap, follower_speed, and leader_speed are not broadcast-compatible") from exc

    if gap_values.ndim > 1:
        raise ValueError("broadcast state must remain scalar or one-dimensional")
    if np.any(gap_values < 0.0):
        raise ValueError("gap is a net bumper-to-bumper distance and must be non-negative metres")
    if np.any(gap_values > MAX_STATE_GAP_M):
        raise ValueError(f"gap exceeds the supported {MAX_STATE_GAP_M:g} m range; check units")
    if np.any(follower_values < 0.0) or np.any(leader_values < 0.0):
        raise ValueError("vehicle speeds must be non-negative metres per second")
    if np.any(follower_values > MAX_STATE_SPEED_M_S) or np.any(leader_values > MAX_STATE_SPEED_M_S):
        raise ValueError(f"vehicle speed exceeds {MAX_STATE_SPEED_M_S:g} m/s; check units")

    return gap_values, follower_values, leader_values, scalar_result


def _return_scalar_or_array(values: np.ndarray, scalar_result: bool) -> float | np.ndarray:
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("car-following equation produced a non-finite acceleration")
    if scalar_result:
        return float(result)
    return result


def _optimal_velocity_from_values(
    gap: np.ndarray,
    *,
    v_max: float,
    s_c: float,
    ell: float,
) -> np.ndarray:
    offset = np.tanh(s_c / ell)
    values = v_max * (np.tanh((gap - s_c) / ell) + offset) / (1.0 + offset)
    # For a valid non-negative gap the equation is in [0, v_max].  Clipping
    # removes only floating-point roundoff at the mathematical endpoints.
    return np.clip(values, 0.0, v_max)


def optimal_velocity(gap: ArrayLike, model_parameters: Mapping[str, Any]) -> float | np.ndarray:
    """Evaluate the normalized hyperbolic-tangent optimal-speed curve.

    ``model_parameters`` may be either a valid OVM or FVDM parameter mapping.
    Only ``v_max``, ``s_c`` and ``ell`` affect this shared curve.
    """

    if not isinstance(model_parameters, Mapping):
        raise TypeError("model_parameters must be a mapping")
    family = "FVDM" if "lambda" in model_parameters else "OVM"
    parameters = validate_model_parameters(family, model_parameters)
    gap_values = _numeric_state(gap, name="gap")
    if np.any(gap_values < 0.0):
        raise ValueError("gap must be non-negative metres")
    if np.any(gap_values > MAX_STATE_GAP_M):
        raise ValueError(f"gap exceeds the supported {MAX_STATE_GAP_M:g} m range; check units")
    result = _optimal_velocity_from_values(
        gap_values,
        v_max=parameters["v_max"],
        s_c=parameters["s_c"],
        ell=parameters["ell"],
    )
    return _return_scalar_or_array(result, gap_values.ndim == 0)


def ovm_acceleration(
    gap: ArrayLike,
    follower_speed: ArrayLike,
    leader_speed: ArrayLike,
    model_parameters: Mapping[str, Any],
) -> float | np.ndarray:
    """Return OVM acceleration ``kappa * (V(gap) - follower_speed)``."""

    parameters = validate_model_parameters("OVM", model_parameters)
    gap_values, follower_values, _leader_values, scalar_result = _validated_state(
        gap, follower_speed, leader_speed
    )
    target_speed = _optimal_velocity_from_values(
        gap_values,
        v_max=parameters["v_max"],
        s_c=parameters["s_c"],
        ell=parameters["ell"],
    )
    acceleration = parameters["kappa"] * (target_speed - follower_values)
    return _return_scalar_or_array(acceleration, scalar_result)


def fvdm_acceleration(
    gap: ArrayLike,
    follower_speed: ArrayLike,
    leader_speed: ArrayLike,
    model_parameters: Mapping[str, Any],
) -> float | np.ndarray:
    """Return FVDM acceleration using the same optimal-speed curve as OVM."""

    parameters = validate_model_parameters("FVDM", model_parameters)
    gap_values, follower_values, leader_values, scalar_result = _validated_state(
        gap, follower_speed, leader_speed
    )
    target_speed = _optimal_velocity_from_values(
        gap_values,
        v_max=parameters["v_max"],
        s_c=parameters["s_c"],
        ell=parameters["ell"],
    )
    acceleration = (
        parameters["kappa"] * (target_speed - follower_values)
        + parameters["lambda"] * (leader_values - follower_values)
    )
    return _return_scalar_or_array(acceleration, scalar_result)


def idm_acceleration(
    gap: ArrayLike,
    follower_speed: ArrayLike,
    leader_speed: ArrayLike,
    model_parameters: Mapping[str, Any],
) -> float | np.ndarray:
    """Return Intelligent Driver Model acceleration.

    Zero and extremely small gaps produce finite emergency braking.  Negative
    gaps are rejected because they represent an already-overlapping state; the
    simulation layer records such a state as a collision rather than asking a
    model to continue through it.
    """

    parameters = validate_model_parameters("IDM", model_parameters)
    gap_values, follower_values, leader_values, scalar_result = _validated_state(
        gap, follower_speed, leader_speed
    )

    closing_speed = follower_values - leader_values
    dynamic_term = (
        follower_values * parameters["T"]
        + follower_values
        * closing_speed
        / (2.0 * np.sqrt(parameters["a"] * parameters["b"]))
    )
    desired_gap = parameters["s0"] + np.maximum(0.0, dynamic_term)
    effective_gap = np.maximum(gap_values, parameters["minimum_gap_epsilon_m"])

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        raw_acceleration = parameters["a"] * (
            1.0
            - np.power(follower_values / parameters["v0"], parameters["delta"])
            - np.square(desired_gap / effective_gap)
        )

    acceleration = np.maximum(raw_acceleration, -parameters["maximum_braking_m_s2"])
    acceleration = np.where(
        gap_values <= parameters["minimum_gap_epsilon_m"],
        -parameters["maximum_braking_m_s2"],
        acceleration,
    )
    return _return_scalar_or_array(acceleration, scalar_result)


MODEL_FUNCTIONS: dict[str, AccelerationFunction] = {
    "OVM": ovm_acceleration,
    "FVDM": fvdm_acceleration,
    "IDM": idm_acceleration,
}


def calculate_acceleration(
    model_name: str,
    gap: ArrayLike,
    follower_speed: ArrayLike,
    leader_speed: ArrayLike,
    model_parameters: Mapping[str, Any],
) -> float | np.ndarray:
    """Dispatch to OVM, FVDM, or IDM through one common interface."""

    canonical = _canonical_model_name(model_name)
    return MODEL_FUNCTIONS[canonical](gap, follower_speed, leader_speed, model_parameters)


__all__ = [
    "MODEL_FUNCTIONS",
    "calculate_acceleration",
    "fvdm_acceleration",
    "idm_acceleration",
    "optimal_velocity",
    "ovm_acceleration",
    "validate_model_parameters",
]


def _configured_model_parameters(
    models: Mapping[str, Any], model_name: str
) -> Mapping[str, Any]:
    for key, value in models.items():
        if isinstance(key, str) and key.upper() == model_name:
            if not isinstance(value, Mapping):
                raise ValueError(f"models.{key} must be a mapping")
            return value
    raise ValueError(f"configuration is missing models.{model_name}")


def main(argv: list[str] | None = None) -> int:
    """Validate the configured model parameters when run as a script."""

    parser = argparse.ArgumentParser(
        description="Validate configured OVM, FVDM, and IDM equations and parameters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
        help="YAML configuration path (default: project config.yaml)",
    )
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping) or not isinstance(config.get("models"), Mapping):
        raise ValueError("configuration must contain a models mapping")

    normalized = {
        name: validate_model_parameters(
            name, _configured_model_parameters(config["models"], name)
        )
        for name in MODEL_FUNCTIONS
    }
    # Exercise one ordinary state and the documented zero-gap boundary.  This
    # is a deterministic equation smoke check, not a parameter calibration.
    checks = {
        name: {
            "ordinary_acceleration_m_s2": float(
                calculate_acceleration(name, 20.0, 10.0, 9.0, parameters)
            ),
            "zero_gap_acceleration_m_s2": float(
                calculate_acceleration(name, 0.0, 0.0, 0.0, parameters)
            ),
        }
        for name, parameters in normalized.items()
    }
    print(
        json.dumps(
            {"parameters_valid": True, "normalized_parameters": normalized, "checks": checks},
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

