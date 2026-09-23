"""A transparent anticipation-and-choice pedestrian model.

The implementation borrows two public modelling ideas, but not source code:

* RVO2/ORCA: choose a velocity near the preferred velocity while evaluating
  constant-velocity future encounters.
* JuPedSim AVM: separate perception, prediction and strategy/action selection.

This small implementation is intentionally inspectable and dependency-light.  It
is not presented as a drop-in reimplementation of either upstream model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-9


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return np.divide(vector, norm, out=np.zeros_like(vector, dtype=float), where=norm > EPS)


def _rotate(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate one 2-vector by every angle and return shape (angles, 2)."""
    x, y = float(vectors[0]), float(vectors[1])
    cosine = np.cos(angles)
    sine = np.sin(angles)
    return np.column_stack((x * cosine - y * sine, x * sine + y * cosine))


@dataclass(frozen=True)
class ModelParameters:
    """Parameters have SI units unless their name says otherwise."""

    dt: float = 0.04
    desired_speed: float = 1.90
    body_radius: float = 0.22
    perception_radius: float = 5.0
    rear_awareness_radius: float = 1.2
    field_of_view_deg: float = 220.0
    prediction_horizon: float = 2.0
    comfort_clearance: float = 0.72
    emergency_clearance: float = 0.44
    reaction_time: float = 0.42
    max_acceleration: float = 2.6
    max_deceleration: float = 3.4
    max_speed_factor: float = 1.12
    goal_tolerance: float = 0.30
    waypoint_tolerance: float = 0.65
    route_offset: float = 2.00
    right_choice_logit: float = 1.05
    left_choice_logit: float = 0.0
    central_choice_logit: float = -1.00
    speed_cost: float = 1.0
    smoothness_cost: float = 0.18
    collision_cost: float = 8.0
    emergency_cost: float = 45.0
    right_action_bonus: float = 0.25
    hard_core_distance: float = 0.40
    projection_iterations: int = 4
    seed: int = 7


@dataclass
class AgentDecision:
    pedestrian: int
    frame: int
    risk_neighbor: int
    neighbor_count: int
    predicted_separation: float
    time_to_cpa: float
    preferred_speed: float
    selected_speed: float
    selected_turn_deg: float
    action: str
    route_side: str


@dataclass
class SimulationResult:
    trajectories: pd.DataFrame
    decisions: pd.DataFrame
    route_choices: pd.DataFrame
    parameters: dict[str, float | int]


class PredictiveVelocityModel:
    """Candidate-velocity pedestrian model with an explicit tactical route choice.

    Information used by pedestrian i at time t:
      own position/velocity, current waypoint and final goal; and position and
      velocity of visible neighbours.  No future measured trajectory is used.

    Decision:
      1. Choose a right/left/central centre waypoint once, using a random-utility
         choice with a population right-side prior.
      2. Sample candidate speeds and headings around the desired direction.
      3. Predict closest approach under constant velocity for visible neighbours.
      4. Minimise goal error + velocity change + collision risk.
      5. Apply reaction-time, acceleration/deceleration and speed limits.
    """

    def __init__(self, parameters: ModelParameters | None = None):
        self.p = parameters or ModelParameters()
        self._rng = np.random.default_rng(self.p.seed)
        self._angle_offsets = np.deg2rad(np.array([-65, -45, -30, -18, -9, 0, 9, 18, 30, 45, 65]))
        self._speed_factors = np.array([0.38, 0.58, 0.76, 0.90, 1.00])

    def _choose_routes(
        self, positions: np.ndarray, goals: np.ndarray, pedestrian_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Choose centre waypoint by a seeded random-utility rule.

        All intended straight paths cross the centre in this experiment, so a
        crowd-pressure term suppresses the central option.  In a sparse circle
        (fewer than 12 participants) the central option receives no penalty.
        """
        n = len(positions)
        crowd_pressure = np.clip((n - 8) / 32.0, 0.0, 1.0)
        logits = np.array(
            [
                self.p.left_choice_logit,
                self.p.central_choice_logit * crowd_pressure,
                self.p.right_choice_logit,
            ]
        )
        utilities = logits[None, :] + self._rng.gumbel(size=(n, 3))
        choices = np.argmax(utilities, axis=1) - 1  # left=-1, central=0, right=1
        goal_directions = _unit(goals - positions)
        right_normals = np.column_stack((goal_directions[:, 1], -goal_directions[:, 0]))
        waypoints = right_normals * (choices * self.p.route_offset)[:, None]
        labels = np.array(["left", "central", "right"])[choices + 1]
        table = pd.DataFrame(
            {
                "pedestrian": pedestrian_ids.astype(int),
                "route_side": labels,
                "route_choice": choices,
                "waypoint_x": waypoints[:, 0],
                "waypoint_y": waypoints[:, 1],
                "crowd_pressure": crowd_pressure,
            }
        )
        return choices, waypoints, table

    def _visible_neighbours(
        self, index: int, positions: np.ndarray, velocities: np.ndarray, preferred_direction: np.ndarray
    ) -> np.ndarray:
        relative = positions - positions[index]
        distance = np.linalg.norm(relative, axis=1)
        forward = relative @ preferred_direction
        cosine_limit = np.cos(np.deg2rad(self.p.field_of_view_deg / 2.0))
        cosine = np.divide(
            forward,
            distance,
            out=np.ones_like(distance),
            where=distance > EPS,
        )
        visible = (
            (distance <= self.p.perception_radius)
            & ((cosine >= cosine_limit) | (distance <= self.p.rear_awareness_radius))
        )
        visible[index] = False
        return np.flatnonzero(visible)

    def _candidate_velocities(self, preferred_direction: np.ndarray, desired_speed: float) -> np.ndarray:
        directions = _rotate(preferred_direction, self._angle_offsets)
        candidates = (directions[:, None, :] * (desired_speed * self._speed_factors)[None, :, None]).reshape(-1, 2)
        return np.vstack((np.zeros((1, 2)), candidates))

    def _predict_candidates(
        self,
        relative_positions: np.ndarray,
        neighbour_velocities: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return each candidate's minimum separation, TTC and neighbour index."""
        if len(relative_positions) == 0:
            count = len(candidates)
            return np.full(count, np.inf), np.full(count, np.inf), np.full(count, -1, dtype=int)
        relative_velocity = neighbour_velocities[None, :, :] - candidates[:, None, :]
        speed_sq = np.einsum("cnk,cnk->cn", relative_velocity, relative_velocity)
        dot = np.einsum("nk,cnk->cn", relative_positions, relative_velocity)
        t_cpa = np.divide(-dot, speed_sq, out=np.full_like(speed_sq, np.inf), where=speed_sq > EPS)
        valid = (t_cpa > 0.0) & (t_cpa <= self.p.prediction_horizon)
        clipped = np.clip(t_cpa, 0.0, self.p.prediction_horizon)
        future = relative_positions[None, :, :] + relative_velocity * clipped[:, :, None]
        separation = np.linalg.norm(future, axis=2)
        current = np.linalg.norm(relative_positions, axis=1)
        next_step = np.linalg.norm(
            relative_positions[None, :, :] + relative_velocity * self.p.dt, axis=2
        )
        separation = np.minimum(separation, current[None, :])
        separation[~valid] = np.inf
        close_now = current < self.p.emergency_clearance
        diverging_but_close = (~valid) & close_now[None, :]
        separation[diverging_but_close] = next_step[diverging_but_close]
        neighbour_index = np.argmin(separation, axis=1)
        rows = np.arange(len(candidates))
        minimum = separation[rows, neighbour_index]
        minimum_ttc = t_cpa[rows, neighbour_index]
        no_risk = ~np.isfinite(minimum)
        neighbour_index[no_risk] = -1
        minimum_ttc[no_risk] = np.inf
        return minimum, minimum_ttc, neighbour_index

    def _select_velocity(
        self,
        index: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        preferred_direction: np.ndarray,
        desired_speed: float,
        pedestrian_ids: np.ndarray,
    ) -> tuple[np.ndarray, AgentDecision]:
        neighbours = self._visible_neighbours(index, positions, velocities, preferred_direction)
        candidates = self._candidate_velocities(preferred_direction, desired_speed)
        preferred_velocity = preferred_direction * desired_speed
        separation, ttc, risk_local = self._predict_candidates(
            positions[neighbours] - positions[index], velocities[neighbours], candidates
        )

        goal_cost = np.sum((candidates - preferred_velocity) ** 2, axis=1) / max(desired_speed**2, EPS)
        smooth_cost = np.sum((candidates - velocities[index]) ** 2, axis=1) / max(desired_speed**2, EPS)
        shortfall = np.clip(self.p.comfort_clearance - separation, 0.0, None)
        safe_ttc = np.clip(ttc, 0.0, self.p.prediction_horizon)
        collision_risk = np.where(
            np.isfinite(separation),
            (shortfall / max(self.p.comfort_clearance - self.p.emergency_clearance, 0.05)) ** 2
            * np.exp(-safe_ttc / self.p.prediction_horizon),
            0.0,
        )
        emergency = np.where(
            separation < self.p.emergency_clearance,
            ((self.p.emergency_clearance - separation) / self.p.emergency_clearance) ** 2,
            0.0,
        )
        right_normal = np.array([preferred_direction[1], -preferred_direction[0]])
        lateral = candidates @ right_normal / max(desired_speed, EPS)
        imminent = float(np.any((separation < self.p.comfort_clearance) & (ttc < 1.2)))
        cost = (
            self.p.speed_cost * goal_cost
            + self.p.smoothness_cost * smooth_cost
            + self.p.collision_cost * collision_risk
            + self.p.emergency_cost * emergency
            - self.p.right_action_bonus * imminent * lateral
        )
        chosen_index = int(np.argmin(cost))
        chosen = candidates[chosen_index]

        preferred_heading = np.arctan2(preferred_direction[1], preferred_direction[0])
        chosen_heading = np.arctan2(chosen[1], chosen[0]) if np.linalg.norm(chosen) > EPS else preferred_heading
        turn = np.degrees((chosen_heading - preferred_heading + np.pi) % (2 * np.pi) - np.pi)
        speed_ratio = np.linalg.norm(chosen) / max(desired_speed, EPS)
        if speed_ratio < 0.72 and abs(turn) >= 15:
            action = "brake_and_turn"
        elif speed_ratio < 0.72:
            action = "brake"
        elif abs(turn) >= 15:
            action = "turn"
        else:
            action = "follow_goal"
        local = int(risk_local[chosen_index])
        risk_neighbour = int(pedestrian_ids[neighbours[local]]) if local >= 0 else -1
        decision = AgentDecision(
            pedestrian=int(pedestrian_ids[index]),
            frame=-1,
            risk_neighbor=risk_neighbour,
            neighbor_count=int(len(neighbours)),
            predicted_separation=float(separation[chosen_index]),
            time_to_cpa=float(ttc[chosen_index]),
            preferred_speed=float(desired_speed),
            selected_speed=float(np.linalg.norm(chosen)),
            selected_turn_deg=float(turn),
            action=action,
            route_side="",
        )
        return chosen, decision

    def _bounded_update(self, current: np.ndarray, selected: np.ndarray, desired_speed: float) -> np.ndarray:
        acceleration = (selected - current) / self.p.reaction_time
        slowing = np.dot(acceleration, current) < 0.0
        limit = self.p.max_deceleration if slowing else self.p.max_acceleration
        norm = np.linalg.norm(acceleration)
        if norm > limit:
            acceleration *= limit / norm
        updated = current + acceleration * self.p.dt
        maximum = desired_speed * self.p.max_speed_factor
        speed = np.linalg.norm(updated)
        if speed > maximum:
            updated *= maximum / speed
        return updated

    def _project_hard_core(self, positions: np.ndarray, movable: np.ndarray) -> np.ndarray:
        """Resolve point-track near-coincidences after synchronous integration.

        This is a numerical body constraint, not a behavioural decision. The
        behavioural layer should avoid invoking it often; its activation rate is
        reported by the evaluation script as a failure diagnostic.
        """
        corrected = positions.copy()
        for _ in range(self.p.projection_iterations):
            changed = False
            for i in range(len(corrected) - 1):
                for j in range(i + 1, len(corrected)):
                    delta = corrected[j] - corrected[i]
                    distance = float(np.linalg.norm(delta))
                    if distance >= self.p.hard_core_distance:
                        continue
                    changed = True
                    if distance <= EPS:
                        angle = 2 * np.pi * ((i * 37 + j * 17) % 101) / 101
                        direction = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        direction = delta / distance
                    overlap = self.p.hard_core_distance - distance
                    if movable[i] and movable[j]:
                        corrected[i] -= direction * overlap * 0.5
                        corrected[j] += direction * overlap * 0.5
                    elif movable[i]:
                        corrected[i] -= direction * overlap
                    elif movable[j]:
                        corrected[j] += direction * overlap
            if not changed:
                break
        return corrected

    def simulate(
        self,
        pedestrian_ids: Iterable[int],
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        goals: np.ndarray,
        frames: Iterable[int],
        desired_speeds: np.ndarray | None = None,
        use_avoidance: bool = True,
    ) -> SimulationResult:
        pedestrian_ids = np.asarray(list(pedestrian_ids), dtype=int)
        frames = np.asarray(list(frames), dtype=int)
        positions = np.asarray(initial_positions, dtype=float).copy()
        velocities = np.asarray(initial_velocities, dtype=float).copy()
        goals = np.asarray(goals, dtype=float).copy()
        n = len(pedestrian_ids)
        if desired_speeds is None:
            desired_speeds = np.full(n, self.p.desired_speed)
        else:
            desired_speeds = np.asarray(desired_speeds, dtype=float)
        route_choices, waypoints, route_table = self._choose_routes(positions, goals, pedestrian_ids)
        waypoint_active = route_choices != 0
        route_labels = route_table["route_side"].to_numpy(str)
        arrived = np.zeros(n, dtype=bool)
        hard_core_correction = np.zeros(n, dtype=float)
        trajectory_rows: list[dict[str, float | int | str]] = []
        decision_rows: list[dict[str, float | int | str]] = []

        for step_index, frame in enumerate(frames):
            for i in range(n):
                trajectory_rows.append(
                    {
                        "pedestrian": int(pedestrian_ids[i]),
                        "frame": int(frame),
                        "time": float(step_index * self.p.dt),
                        "x": float(positions[i, 0]),
                        "y": float(positions[i, 1]),
                        "vx_model": float(velocities[i, 0]),
                        "vy_model": float(velocities[i, 1]),
                        "route_side_model": route_labels[i],
                        "hard_core_correction_m": float(hard_core_correction[i]),
                    }
                )
            if step_index == len(frames) - 1:
                break

            targets = np.where(waypoint_active[:, None], waypoints, goals)
            distance_to_waypoint = np.linalg.norm(positions - waypoints, axis=1)
            passed_centre = np.einsum("ij,ij->i", positions - waypoints, goals - waypoints) > 0.0
            waypoint_active &= ~((distance_to_waypoint <= self.p.waypoint_tolerance) | passed_centre)
            targets = np.where(waypoint_active[:, None], waypoints, goals)
            desired_directions = _unit(targets - positions)
            selected_velocities = np.zeros_like(velocities)
            decisions: list[AgentDecision] = []

            for i in range(n):
                goal_distance = np.linalg.norm(goals[i] - positions[i])
                if goal_distance <= self.p.goal_tolerance:
                    arrived[i] = True
                if arrived[i]:
                    selected_velocities[i] = np.zeros(2)
                    decisions.append(
                        AgentDecision(
                            int(pedestrian_ids[i]), int(frame), -1, 0, np.inf, np.inf,
                            0.0, 0.0, 0.0, "arrived", route_labels[i]
                        )
                    )
                    continue
                if use_avoidance:
                    selected, decision = self._select_velocity(
                        i,
                        positions,
                        velocities,
                        desired_directions[i],
                        float(desired_speeds[i]),
                        pedestrian_ids,
                    )
                else:
                    selected = desired_directions[i] * desired_speeds[i]
                    decision = AgentDecision(
                        int(pedestrian_ids[i]), int(frame), -1, n - 1, np.nan, np.nan,
                        float(desired_speeds[i]), float(desired_speeds[i]), 0.0,
                        "follow_goal", route_labels[i]
                    )
                decision.frame = int(frame)
                decision.route_side = route_labels[i]
                selected_velocities[i] = selected
                decisions.append(decision)

            new_velocities = np.vstack(
                [
                    self._bounded_update(velocities[i], selected_velocities[i], float(desired_speeds[i]))
                    for i in range(n)
                ]
            )
            new_velocities[arrived] = 0.0
            old_positions = positions.copy()
            proposed_positions = positions + new_velocities * self.p.dt
            proposed_positions[arrived] = positions[arrived]
            if use_avoidance:
                positions = self._project_hard_core(proposed_positions, ~arrived)
            else:
                positions = proposed_positions
            hard_core_correction = np.linalg.norm(positions - proposed_positions, axis=1)
            new_velocities = (positions - old_positions) / self.p.dt
            velocities = new_velocities
            decision_rows.extend(asdict(decision) for decision in decisions)

        return SimulationResult(
            trajectories=pd.DataFrame(trajectory_rows),
            decisions=pd.DataFrame(decision_rows),
            route_choices=route_table,
            parameters=asdict(self.p),
        )
