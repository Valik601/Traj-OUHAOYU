"""Unit tests for the OVM, FVDM, and IDM equations."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_models():
    name = "test_target_car_following_models"
    if name in sys.modules:
        return sys.modules[name]
    path = PROJECT_ROOT / "src" / "04_car_following_models.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODELS = _load_models()
OVM_PARAMETERS = {"kappa": 0.5, "v_max": 33.33, "s_c": 25.0, "ell": 8.0}
FVDM_PARAMETERS = {**OVM_PARAMETERS, "lambda": 0.5}
IDM_PARAMETERS = {"v0": 33.33, "T": 1.5, "s0": 2.0, "a": 1.2, "b": 2.0, "delta": 4.0}


def test_optimal_velocity_matches_normalized_tanh_equation() -> None:
    gap = 18.0
    offset = np.tanh(OVM_PARAMETERS["s_c"] / OVM_PARAMETERS["ell"])
    expected = OVM_PARAMETERS["v_max"] * (
        np.tanh((gap - OVM_PARAMETERS["s_c"]) / OVM_PARAMETERS["ell"]) + offset
    ) / (1.0 + offset)
    assert MODELS.optimal_velocity(gap, OVM_PARAMETERS) == pytest.approx(expected)


def test_optimal_velocity_is_zero_at_zero_gap_and_monotonic() -> None:
    gaps = np.array([0.0, 2.0, 10.0, 25.0, 50.0, 500.0])
    target = MODELS.optimal_velocity(gaps, OVM_PARAMETERS)
    assert target[0] == pytest.approx(0.0, abs=1.0e-14)
    assert np.all(np.diff(target) > 0.0)
    assert target[-1] == pytest.approx(OVM_PARAMETERS["v_max"], rel=1.0e-10)


def test_ovm_matches_equation_and_does_not_use_leader_speed() -> None:
    gap = 20.0
    follower_speed = 11.0
    target = MODELS.optimal_velocity(gap, OVM_PARAMETERS)
    expected = OVM_PARAMETERS["kappa"] * (target - follower_speed)
    low_leader = MODELS.ovm_acceleration(gap, follower_speed, 0.0, OVM_PARAMETERS)
    high_leader = MODELS.ovm_acceleration(gap, follower_speed, 30.0, OVM_PARAMETERS)
    assert low_leader == pytest.approx(expected)
    assert high_leader == pytest.approx(expected)


def test_fvdm_adds_full_velocity_difference_term_to_ovm() -> None:
    gap = 20.0
    follower_speed = 12.0
    leader_speed = 9.0
    ovm = MODELS.ovm_acceleration(gap, follower_speed, leader_speed, OVM_PARAMETERS)
    fvdm = MODELS.fvdm_acceleration(gap, follower_speed, leader_speed, FVDM_PARAMETERS)
    assert fvdm == pytest.approx(
        ovm + FVDM_PARAMETERS["lambda"] * (leader_speed - follower_speed)
    )


def test_idm_matches_reference_equation_away_from_safety_cap() -> None:
    gap = 30.0
    follower_speed = 14.0
    leader_speed = 12.0
    p = IDM_PARAMETERS
    desired_gap = p["s0"] + max(
        0.0,
        follower_speed * p["T"]
        + follower_speed * (follower_speed - leader_speed) / (2.0 * np.sqrt(p["a"] * p["b"])),
    )
    expected = p["a"] * (
        1.0 - (follower_speed / p["v0"]) ** p["delta"] - (desired_gap / gap) ** 2
    )
    assert MODELS.idm_acceleration(gap, follower_speed, leader_speed, p) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (MODELS.ovm_acceleration, OVM_PARAMETERS),
        (MODELS.fvdm_acceleration, FVDM_PARAMETERS),
        (MODELS.idm_acceleration, IDM_PARAMETERS),
    ],
)
def test_all_models_have_common_vectorized_interface(function, parameters) -> None:
    result = function(
        np.array([5.0, 10.0, 20.0]),
        np.array([0.0, 5.0, 10.0]),
        8.0,
        parameters,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))
    assert isinstance(function(10.0, 5.0, 8.0, parameters), float)


@pytest.mark.parametrize("name", ["OVM", "ovm", " FvDm ", "idm"])
def test_dispatcher_is_case_and_whitespace_insensitive(name: str) -> None:
    parameters = {
        "OVM": OVM_PARAMETERS,
        "FVDM": FVDM_PARAMETERS,
        "IDM": IDM_PARAMETERS,
    }[name.strip().upper()]
    expected = MODELS.MODEL_FUNCTIONS[name.strip().upper()](10.0, 5.0, 6.0, parameters)
    assert MODELS.calculate_acceleration(name, 10.0, 5.0, 6.0, parameters) == expected


def test_idm_zero_and_tiny_gaps_produce_finite_emergency_braking() -> None:
    parameters = {**IDM_PARAMETERS, "maximum_braking_m_s2": 42.0}
    at_zero = MODELS.idm_acceleration(0.0, 0.0, 0.0, parameters)
    at_tiny_gap = MODELS.idm_acceleration(1.0e-8, 10.0, 0.0, parameters)
    assert at_zero == pytest.approx(-42.0)
    assert at_tiny_gap == pytest.approx(-42.0)
    assert np.isfinite(at_zero)
    assert np.isfinite(at_tiny_gap)


def test_ovm_and_fvdm_are_finite_at_zero_gap_and_zero_speed() -> None:
    assert MODELS.ovm_acceleration(0.0, 0.0, 0.0, OVM_PARAMETERS) == pytest.approx(0.0)
    assert MODELS.fvdm_acceleration(0.0, 0.0, 0.0, FVDM_PARAMETERS) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("gap", "follower_speed", "leader_speed", "error"),
    [
        (-0.01, 5.0, 5.0, ValueError),
        (10.0, -0.01, 5.0, ValueError),
        (10.0, 5.0, -0.01, ValueError),
        (10.0, np.nan, 5.0, ValueError),
        (10.0, 5.0, np.inf, ValueError),
        (10.0, "5", 5.0, TypeError),
        (10.0, True, 5.0, TypeError),
    ],
)
def test_invalid_state_inputs_are_rejected(gap, follower_speed, leader_speed, error) -> None:
    with pytest.raises(error):
        MODELS.ovm_acceleration(gap, follower_speed, leader_speed, OVM_PARAMETERS)


def test_matrix_and_incompatible_vector_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        MODELS.ovm_acceleration(np.ones((2, 2)), 5.0, 5.0, OVM_PARAMETERS)
    with pytest.raises(ValueError, match="broadcast-compatible"):
        MODELS.ovm_acceleration([10.0, 20.0], [5.0, 6.0, 7.0], 5.0, OVM_PARAMETERS)


@pytest.mark.parametrize(
    ("name", "parameters", "message"),
    [
        ("OVM", {"kappa": 0.5}, "missing required keys"),
        ("OVM", {**OVM_PARAMETERS, "kapppa": 0.5}, "unknown keys"),
        ("OVM", {**OVM_PARAMETERS, "kappa": 0.0}, "outside the supported range"),
        ("FVDM", {**FVDM_PARAMETERS, "lambda": -0.1}, "outside the supported range"),
        ("IDM", {**IDM_PARAMETERS, "b": 0.0}, "outside the supported range"),
        ("IDM", {**IDM_PARAMETERS, "delta": 0.5}, "outside the supported range"),
        ("IDM", {**IDM_PARAMETERS, "v0": np.nan}, "must be finite"),
    ],
)
def test_invalid_parameters_are_rejected(name: str, parameters: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MODELS.validate_model_parameters(name, parameters)


def test_parameter_validation_adds_explicit_idm_safety_defaults_without_mutation() -> None:
    original = dict(IDM_PARAMETERS)
    normalized = MODELS.validate_model_parameters("IDM", original)
    assert original == IDM_PARAMETERS
    assert normalized["minimum_gap_epsilon_m"] > 0.0
    assert normalized["maximum_braking_m_s2"] > 0.0


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown car-following model"):
        MODELS.calculate_acceleration("Gipps", 10.0, 5.0, 5.0, {})

