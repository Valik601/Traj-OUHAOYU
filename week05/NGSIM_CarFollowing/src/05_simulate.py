"""Simulate OVM, FVDM, and IDM against one measured leader trajectory.

The measured follower is used only for the common initial position and speed.
After time zero every model evolves independently, driven by the measured
leader position and speed.  Speed uses a forward-Euler update with a
non-negative projection; position uses the requested trapezoidal update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger("ngsim.simulate")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_NAMES = ("OVM", "FVDM", "IDM")
REQUIRED_TRAJECTORY_COLUMNS = (
    "t",
    "leader_x",
    "leader_v",
    "leader_a",
    "follower_x_true",
    "follower_v_true",
    "follower_a_true",
    "leader_length",
    "gap_true",
)


def _load_models_module() -> ModuleType:
    """Load the numbered model script without relying on the current directory."""

    module_path = SCRIPT_DIR / "04_car_following_models.py"
    module_name = "ngsim_car_following_models"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load car-following models from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODELS = _load_models_module()


def _plain_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not boolean")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
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


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    number = float(value)
    return number if np.isfinite(number) else None


def _state_record(
    *,
    index: int,
    t: Any,
    x: Any,
    v: Any,
    a: Any,
    gap: Any,
) -> dict[str, float | int | None]:
    return {
        "index": int(index),
        "t_s": _json_number(t),
        "x_m": _json_number(x),
        "v_m_s": _json_number(v),
        "a_m_s2": _json_number(a),
        "gap_m": _json_number(gap),
    }


def validate_trajectory(
    trajectory: pd.DataFrame,
    *,
    time_step_s: float,
    time_tolerance_s: float = 1.0e-6,
) -> pd.DataFrame:
    """Return a numeric, validated copy of a prepared trajectory table."""

    if not isinstance(trajectory, pd.DataFrame):
        raise TypeError("trajectory must be a pandas DataFrame")
    missing = [column for column in REQUIRED_TRAJECTORY_COLUMNS if column not in trajectory.columns]
    if missing:
        raise ValueError(f"trajectory is missing required columns: {', '.join(missing)}")
    if len(trajectory) < 2:
        raise ValueError("trajectory must contain at least two time samples")

    dt = _plain_float(time_step_s, name="time_step_s")
    tolerance = _plain_float(time_tolerance_s, name="time_tolerance_s")
    if not 0.0 < dt <= 1.0:
        raise ValueError("time_step_s must be in (0, 1] seconds")
    if not 0.0 < tolerance <= 0.01:
        raise ValueError("time_tolerance_s must be in (0, 0.01] seconds")

    clean = trajectory.loc[:, REQUIRED_TRAJECTORY_COLUMNS].copy()
    for column in REQUIRED_TRAJECTORY_COLUMNS:
        try:
            clean[column] = pd.to_numeric(clean[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"trajectory column {column!r} must be numeric") from exc

    values = clean.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        bad_columns = [
            column
            for column in REQUIRED_TRAJECTORY_COLUMNS
            if not np.all(np.isfinite(clean[column].to_numpy(dtype=float)))
        ]
        raise ValueError(f"trajectory contains non-finite values in: {', '.join(bad_columns)}")

    times = clean["t"].to_numpy()
    if abs(times[0]) > tolerance:
        raise ValueError(f"prepared trajectory must start at t=0, got {times[0]!r}")
    increments = np.diff(times)
    if np.any(increments <= 0.0):
        raise ValueError("trajectory time must be strictly increasing without duplicates")
    if not np.allclose(increments, dt, rtol=0.0, atol=tolerance):
        first_bad = int(np.flatnonzero(np.abs(increments - dt) > tolerance)[0])
        raise ValueError(
            "trajectory does not match configured time_step_s at interval "
            f"{first_bad}->{first_bad + 1}: observed {increments[first_bad]:.12g} s, expected {dt:.12g} s"
        )

    for column in ("leader_v", "follower_v_true"):
        if (clean[column] < 0.0).any():
            raise ValueError(f"trajectory column {column!r} contains negative speed")
        if (clean[column] > MODELS.MAX_STATE_SPEED_M_S).any():
            raise ValueError(f"trajectory column {column!r} exceeds the supported speed range")
    if (clean["leader_length"] <= 0.0).any():
        raise ValueError("leader_length must be positive")
    if not np.allclose(
        clean["leader_length"].to_numpy(),
        clean["leader_length"].iloc[0],
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("leader_length must remain constant within a following event")

    recomputed_gap = (
        clean["leader_x"].to_numpy()
        - clean["follower_x_true"].to_numpy()
        - clean["leader_length"].to_numpy()
    )
    if not np.allclose(recomputed_gap, clean["gap_true"].to_numpy(), rtol=0.0, atol=1.0e-4):
        worst = int(np.argmax(np.abs(recomputed_gap - clean["gap_true"].to_numpy())))
        raise ValueError(
            "gap_true is inconsistent with leader_x - follower_x_true - leader_length "
            f"at row {worst}"
        )
    if (clean["gap_true"] <= 0.0).any():
        raise ValueError("measured trajectory contains a non-positive true net gap")

    return clean


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def validate_initial_conditions(
    initial_conditions: Mapping[str, Any],
    trajectory: pd.DataFrame,
    *,
    tolerance: float = 1.0e-4,
) -> None:
    """Cross-check prepared initial-state metadata against trajectory row zero.

    The trajectory is authoritative because it is the exact state integrated by
    this stage.  Raw-state fields are intentionally not compared when prepared
    aliases are present.
    """

    if not isinstance(initial_conditions, Mapping):
        raise TypeError("initial_conditions.json must contain a JSON object")
    leader = initial_conditions.get("leader", {})
    follower = initial_conditions.get("follower", {})
    relative = initial_conditions.get("relative", {})
    if not isinstance(leader, Mapping) or not isinstance(follower, Mapping):
        raise TypeError("initial_conditions leader and follower entries must be objects")
    if not isinstance(relative, Mapping):
        raise TypeError("initial_conditions relative entry must be an object")

    row = trajectory.iloc[0]
    comparisons: tuple[tuple[str, Mapping[str, Any], tuple[str, ...], float], ...] = (
        (
            "follower initial longitudinal position",
            follower,
            (
                "prepared_longitudinal_position_m",
                "initial_longitudinal_position_m",
                "local_y_m",
                "x_m",
                "x",
            ),
            float(row["follower_x_true"]),
        ),
        (
            "follower initial speed",
            follower,
            ("prepared_speed_m_s", "initial_speed_m_s", "speed_m_s", "v_m_s", "v"),
            float(row["follower_v_true"]),
        ),
        (
            "leader initial longitudinal position",
            leader,
            (
                "prepared_longitudinal_position_m",
                "initial_longitudinal_position_m",
                "local_y_m",
                "x_m",
                "x",
            ),
            float(row["leader_x"]),
        ),
        (
            "leader initial speed",
            leader,
            ("prepared_speed_m_s", "initial_speed_m_s", "speed_m_s", "v_m_s", "v"),
            float(row["leader_v"]),
        ),
        (
            "leader length",
            leader,
            ("vehicle_length_m", "length_m", "leader_length"),
            float(row["leader_length"]),
        ),
        (
            "initial net gap",
            relative,
            ("initial_gap_m", "gap_m", "gap"),
            float(row["gap_true"]),
        ),
    )
    for description, mapping, keys, expected in comparisons:
        key, value = _first_present(mapping, keys)
        if key is None:
            continue
        observed = _plain_float(value, name=f"initial_conditions.{key}")
        if not np.isclose(observed, expected, rtol=0.0, atol=tolerance):
            raise ValueError(
                f"{description} in initial_conditions.json ({observed:.12g}) does not match "
                f"trajectory_clean.csv ({expected:.12g})"
            )


def _failure_payload(
    *,
    status: str,
    reason: str,
    index: int,
    times: list[float],
    positions: list[float],
    speeds: list[float],
    accelerations: list[float],
    gaps: list[float],
    valid: list[bool],
) -> dict[str, Any]:
    failure_state = _state_record(
        index=index,
        t=times[-1] if times else None,
        x=positions[-1] if positions else None,
        v=speeds[-1] if speeds else None,
        a=accelerations[-1] if accelerations else None,
        gap=gaps[-1] if gaps else None,
    )
    valid_indices = [i for i, flag in enumerate(valid) if flag]
    last_valid_state = None
    if valid_indices:
        last = valid_indices[-1]
        last_valid_state = _state_record(
            index=last,
            t=times[last],
            x=positions[last],
            v=speeds[last],
            a=accelerations[last],
            gap=gaps[last],
        )
    return {
        "status": status,
        "completed": False,
        "failure_reason": reason,
        "failure_index": int(index),
        "failure_time_s": _json_number(failure_state["t_s"]),
        "pre_failure_state": last_valid_state,
        "failure_state": failure_state,
    }


def simulate_model(
    trajectory: pd.DataFrame,
    model_name: str,
    model_parameters: Mapping[str, Any],
    *,
    time_step_s: float = 0.1,
    collision_gap_m: float = 0.0,
    maximum_abs_acceleration_m_s2: float = 20.0,
    stop_on_unreasonable_acceleration: bool = True,
    time_tolerance_s: float = 1.0e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate one follower model and return its trajectory and status.

    A collision row is retained with ``valid=False`` so the event time and the
    post-integration state remain auditable.  No later rows are generated.
    """

    canonical = MODELS._canonical_model_name(model_name)
    parameters = MODELS.validate_model_parameters(canonical, model_parameters)
    dt = _plain_float(time_step_s, name="time_step_s")
    collision_threshold = _plain_float(collision_gap_m, name="collision_gap_m")
    maximum_acceleration = _plain_float(
        maximum_abs_acceleration_m_s2, name="maximum_abs_acceleration_m_s2"
    )
    stop_on_extreme = _plain_bool(
        stop_on_unreasonable_acceleration, name="stop_on_unreasonable_acceleration"
    )
    if collision_threshold < 0.0:
        raise ValueError("collision_gap_m must be non-negative")
    if not 0.0 < maximum_acceleration <= 1_000.0:
        raise ValueError("maximum_abs_acceleration_m_s2 must be in (0, 1000]")

    clean = validate_trajectory(
        trajectory,
        time_step_s=dt,
        time_tolerance_s=time_tolerance_s,
    )
    t_external = clean["t"].to_numpy(dtype=float)
    leader_x = clean["leader_x"].to_numpy(dtype=float)
    leader_v = clean["leader_v"].to_numpy(dtype=float)
    leader_length = clean["leader_length"].to_numpy(dtype=float)

    times: list[float] = [float(t_external[0])]
    positions: list[float] = [float(clean["follower_x_true"].iloc[0])]
    speeds: list[float] = [float(clean["follower_v_true"].iloc[0])]
    accelerations: list[float] = [float("nan")]
    gaps: list[float] = [float(leader_x[0] - positions[0] - leader_length[0])]
    valid: list[bool] = [True]
    warning_events: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None

    if gaps[0] <= collision_threshold:
        valid[0] = False
        failure = _failure_payload(
            status="collision",
            reason="initial_net_gap_at_or_below_collision_threshold",
            index=0,
            times=times,
            positions=positions,
            speeds=speeds,
            accelerations=accelerations,
            gaps=gaps,
            valid=valid,
        )
    else:
        for index in range(len(clean) - 1):
            try:
                acceleration = float(
                    MODELS.calculate_acceleration(
                        canonical,
                        gaps[-1],
                        speeds[-1],
                        leader_v[index],
                        parameters,
                    )
                )
            except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
                valid[-1] = False
                failure = _failure_payload(
                    status="numerical_failure",
                    reason=f"model_evaluation_error: {exc}",
                    index=index,
                    times=times,
                    positions=positions,
                    speeds=speeds,
                    accelerations=accelerations,
                    gaps=gaps,
                    valid=valid,
                )
                break

            accelerations[-1] = acceleration
            if not np.isfinite(acceleration):
                valid[-1] = False
                failure = _failure_payload(
                    status="numerical_failure",
                    reason="non_finite_acceleration",
                    index=index,
                    times=times,
                    positions=positions,
                    speeds=speeds,
                    accelerations=accelerations,
                    gaps=gaps,
                    valid=valid,
                )
                break
            if abs(acceleration) > maximum_acceleration:
                warning = {
                    "index": int(index),
                    "t_s": float(t_external[index]),
                    "acceleration_m_s2": acceleration,
                    "limit_m_s2": maximum_acceleration,
                }
                warning_events.append(warning)
                if stop_on_extreme:
                    valid[-1] = False
                    failure = _failure_payload(
                        status="numerical_failure",
                        reason="absolute_acceleration_exceeds_limit",
                        index=index,
                        times=times,
                        positions=positions,
                        speeds=speeds,
                        accelerations=accelerations,
                        gaps=gaps,
                        valid=valid,
                    )
                    break

            next_speed = max(0.0, speeds[-1] + acceleration * dt)
            next_position = positions[-1] + 0.5 * (speeds[-1] + next_speed) * dt
            next_gap = leader_x[index + 1] - next_position - leader_length[index + 1]
            if not np.all(np.isfinite([next_speed, next_position, next_gap])):
                failure = _failure_payload(
                    status="numerical_failure",
                    reason="non_finite_integrated_state",
                    index=index + 1,
                    times=times,
                    positions=positions,
                    speeds=speeds,
                    accelerations=accelerations,
                    gaps=gaps,
                    valid=valid,
                )
                failure["failure_index"] = int(index + 1)
                failure["failure_time_s"] = float(t_external[index + 1])
                failure["attempted_state"] = _state_record(
                    index=index + 1,
                    t=t_external[index + 1],
                    x=next_position,
                    v=next_speed,
                    a=None,
                    gap=next_gap,
                )
                break

            times.append(float(t_external[index + 1]))
            positions.append(float(next_position))
            speeds.append(float(next_speed))
            accelerations.append(float("nan"))
            gaps.append(float(next_gap))
            valid.append(True)

            if next_gap <= collision_threshold:
                valid[-1] = False
                failure = _failure_payload(
                    status="collision",
                    reason="net_gap_at_or_below_collision_threshold",
                    index=index + 1,
                    times=times,
                    positions=positions,
                    speeds=speeds,
                    accelerations=accelerations,
                    gaps=gaps,
                    valid=valid,
                )
                break

    if failure is None:
        final_index = len(times) - 1
        final_acceleration = float(
            MODELS.calculate_acceleration(
                canonical,
                gaps[-1],
                speeds[-1],
                leader_v[final_index],
                parameters,
            )
        )
        accelerations[-1] = final_acceleration
        if not np.isfinite(final_acceleration):
            valid[-1] = False
            failure = _failure_payload(
                status="numerical_failure",
                reason="non_finite_acceleration",
                index=final_index,
                times=times,
                positions=positions,
                speeds=speeds,
                accelerations=accelerations,
                gaps=gaps,
                valid=valid,
            )
        elif abs(final_acceleration) > maximum_acceleration:
            warning_events.append(
                {
                    "index": int(final_index),
                    "t_s": float(times[-1]),
                    "acceleration_m_s2": final_acceleration,
                    "limit_m_s2": maximum_acceleration,
                }
            )
            if stop_on_extreme:
                valid[-1] = False
                failure = _failure_payload(
                    status="numerical_failure",
                    reason="absolute_acceleration_exceeds_limit",
                    index=final_index,
                    times=times,
                    positions=positions,
                    speeds=speeds,
                    accelerations=accelerations,
                    gaps=gaps,
                    valid=valid,
                )

    if failure is None:
        final_status: dict[str, Any] = {
            "status": "completed",
            "completed": True,
            "failure_reason": None,
            "failure_index": None,
            "failure_time_s": None,
            "pre_failure_state": None,
            "failure_state": None,
        }
    else:
        final_status = failure

    row_status = np.full(len(times), "valid", dtype=object)
    if failure is not None and len(row_status):
        row_status[-1] = str(failure["status"])
    result = pd.DataFrame(
        {
            "t": np.asarray(times, dtype=float),
            "x_sim": np.asarray(positions, dtype=float),
            "v_sim": np.asarray(speeds, dtype=float),
            "a_sim": np.asarray(accelerations, dtype=float),
            "gap_sim": np.asarray(gaps, dtype=float),
            "valid": np.asarray(valid, dtype=bool),
            "status": row_status,
        }
    )

    valid_indices = np.flatnonzero(result["valid"].to_numpy(dtype=bool))
    last_valid_state = None
    if valid_indices.size:
        last = int(valid_indices[-1])
        last_valid_state = _state_record(
            index=last,
            t=result["t"].iloc[last],
            x=result["x_sim"].iloc[last],
            v=result["v_sim"].iloc[last],
            a=result["a_sim"].iloc[last],
            gap=result["gap_sim"].iloc[last],
        )

    final_status.update(
        {
            "model": canonical,
            "parameters": parameters,
            "total_requested_samples": int(len(clean)),
            "written_samples": int(len(result)),
            "valid_samples": int(result["valid"].sum()),
            "last_valid_state": last_valid_state,
            "minimum_recorded_gap_m": float(result["gap_sim"].min()),
            "warning_events": warning_events,
            "initial_state": _state_record(
                index=0,
                t=result["t"].iloc[0],
                x=result["x_sim"].iloc[0],
                v=result["v_sim"].iloc[0],
                a=result["a_sim"].iloc[0],
                gap=result["gap_sim"].iloc[0],
            ),
        }
    )
    return result, final_status


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    candidate = Path(config_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("configuration file must contain a YAML mapping")
    return payload


def _required_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration key {key!r} must be a mapping")
    return value


def _model_parameters(config_models: Mapping[str, Any], model_name: str) -> Mapping[str, Any]:
    for key, value in config_models.items():
        if isinstance(key, str) and key.upper() == model_name:
            if not isinstance(value, Mapping):
                raise ValueError(f"configuration models.{key} must be a mapping")
            return value
    raise ValueError(f"configuration is missing models.{model_name}")


def run_simulations(config_path: str | Path | None = None) -> dict[str, Any]:
    """Run all configured models and write CSV/status artifacts."""

    resolved_config = _resolve_config_path(config_path)
    config = _load_config(resolved_config)
    project_root = resolved_config.parent
    paths = _required_mapping(config, "paths")
    model_config = _required_mapping(config, "models")
    simulation_config = _required_mapping(config, "simulation")

    prepared_dir = Path(str(paths.get("prepared_dir", "outputs/04_prepared")))
    simulation_dir = Path(str(paths.get("simulation_dir", "outputs/05_simulation")))
    if not prepared_dir.is_absolute():
        prepared_dir = project_root / prepared_dir
    if not simulation_dir.is_absolute():
        simulation_dir = project_root / simulation_dir
    prepared_dir = prepared_dir.resolve()
    simulation_dir = simulation_dir.resolve()
    trajectory_path = prepared_dir / "trajectory_clean.csv"
    initial_conditions_path = prepared_dir / "initial_conditions.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"prepared trajectory not found: {trajectory_path}")
    if not initial_conditions_path.is_file():
        raise FileNotFoundError(f"initial conditions not found: {initial_conditions_path}")

    time_step = _plain_float(
        simulation_config.get("time_step_s", 0.1), name="simulation.time_step_s"
    )
    collision_gap = _plain_float(
        simulation_config.get("collision_gap_m", 0.0), name="simulation.collision_gap_m"
    )
    maximum_acceleration = _plain_float(
        simulation_config.get("maximum_abs_acceleration_m_s2", 20.0),
        name="simulation.maximum_abs_acceleration_m_s2",
    )
    stop_on_extreme = _plain_bool(
        simulation_config.get("stop_on_unreasonable_acceleration", True),
        name="simulation.stop_on_unreasonable_acceleration",
    )
    data_config = config.get("data", {})
    if data_config is None:
        data_config = {}
    if not isinstance(data_config, Mapping):
        raise ValueError("configuration key 'data' must be a mapping")
    time_tolerance = _plain_float(
        data_config.get("time_grid_tolerance_s", 1.0e-6),
        name="data.time_grid_tolerance_s",
    )

    raw_trajectory = pd.read_csv(trajectory_path)
    trajectory = validate_trajectory(
        raw_trajectory,
        time_step_s=time_step,
        time_tolerance_s=time_tolerance,
    )
    with initial_conditions_path.open("r", encoding="utf-8") as stream:
        initial_conditions = json.load(stream)
    validate_initial_conditions(initial_conditions, trajectory)

    # Validate every model before creating outputs, so a misspelled parameter
    # never leaves a mixture of fresh and stale simulation files.
    normalized_parameters = {
        name: MODELS.validate_model_parameters(name, _model_parameters(model_config, name))
        for name in MODEL_NAMES
    }

    simulation_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        result, status = simulate_model(
            trajectory,
            model_name,
            normalized_parameters[model_name],
            time_step_s=time_step,
            collision_gap_m=collision_gap,
            maximum_abs_acceleration_m_s2=maximum_acceleration,
            stop_on_unreasonable_acceleration=stop_on_extreme,
            time_tolerance_s=time_tolerance,
        )
        output_path = simulation_dir / f"simulation_{model_name}.csv"
        result.to_csv(output_path, index=False, float_format="%.10g")
        status["output_csv"] = str(output_path)
        statuses[model_name] = status
        LOGGER.info(
            "%s: %s (%d/%d valid samples)",
            model_name,
            status["status"],
            status["valid_samples"],
            status["total_requested_samples"],
        )

    first = trajectory.iloc[0]
    status_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(resolved_config),
            "config_sha256": _sha256(resolved_config),
            "source_trajectory": str(trajectory_path),
            "source_trajectory_sha256": _sha256(trajectory_path),
            "initial_conditions": str(initial_conditions_path),
            "initial_conditions_sha256": _sha256(initial_conditions_path),
            "leader_input": "measured leader_x and leader_v from trajectory_clean.csv",
            "follower_truth_usage": "initial follower_x_true and follower_v_true only",
            "integration_method": {
                "speed": "forward Euler with max(0, v + a*dt)",
                "position": "trapezoidal using old and updated speed",
            },
            "time_step_s": time_step,
            "collision_gap_m": collision_gap,
            "maximum_abs_acceleration_m_s2": maximum_acceleration,
            "stop_on_unreasonable_acceleration": stop_on_extreme,
            "requested_samples": int(len(trajectory)),
            "requested_start_time_s": float(trajectory["t"].iloc[0]),
            "requested_end_time_s": float(trajectory["t"].iloc[-1]),
            "common_initial_state": {
                "follower_x_m": float(first["follower_x_true"]),
                "follower_v_m_s": float(first["follower_v_true"]),
                "leader_x_m": float(first["leader_x"]),
                "leader_v_m_s": float(first["leader_v"]),
                "leader_length_m": float(first["leader_length"]),
                "net_gap_m": float(first["gap_true"]),
            },
            "units": {
                "t": "s",
                "x_sim": "m",
                "v_sim": "m/s",
                "a_sim": "m/s^2",
                "gap_sim": "m",
            },
        },
        "models": statuses,
    }
    status_path = simulation_dir / "simulation_status.json"
    status_path.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote simulation status to %s", status_path)
    return status_payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate OVM, FVDM, and IDM using the measured NGSIM leader trajectory."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML configuration path; relative paths are resolved from the project root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_simulations(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

