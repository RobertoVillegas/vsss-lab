"""Native C7/C8 three-agent curriculum and evaluation."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from vsss_baselines import DynamicTeamController
from vsss_baselines.controllers import robot_pose
from vsss_env._native import BatchSimulator

from vsss_train.ablations import (
    EntityAttentionActor,
    LatticeSharedActor,
    RecurrentSharedActor,
    SymmetricWheelLattice,
)
from vsss_train.marl import (
    CircularPrimitiveRoleActor,
    ParametricPrimitiveRoleActor,
    PrimitiveRoleActor,
    RoleSharedActor,
    SharedActor,
    TeamBatch,
    build_team_observation,
    stack_team_batches,
)
from vsss_train.ppo import seed_everything
from vsss_train.primitives import (
    CircularPrimitiveSet,
    SoccerPrimitiveSet,
    circular_primitive_wheel_actions,
    nearest_canonical_direction,
    parametric_primitive_wheel_actions,
    primitive_wheel_actions,
)
from vsss_train.roles import DynamicRoleAssigner, Role, RoleAssignment, assign_roles

FloatArray = NDArray[np.float32]
ROBOT_BASE = 10
ROBOT_WIDTH = 11
# The teacher demonstrates full authority, but a target of exactly one is unreachable
# through tanh, so it is expressed just inside the interval the policy can express.
TEACHER_AUTHORITY = 0.9
# Rule 15 of the LARC VSSS rulebook. The figure fixes the free-ball marks 37.5 cm from each
# end; the lateral placement is read as the quadrant centre. A robot within 20 cm of the mark
# is moved to its own half.
FREE_BALL_X = 0.375
FREE_BALL_Y = 0.325
# Rule 2.4: the goal area is a 70 by 15 cm rectangle in front of each goal. An impasse there
# is a goal kick, which is not modelled, so the free ball does not apply inside it.
GOAL_AREA_DEPTH = 0.15
GOAL_AREA_HALF_WIDTH = 0.35
ACTION_PARSERS = (
    "continuous",
    "lattice",
    "primitive",
    "parametric_primitive",
    "circular_primitive",
)


def team_action_width(action_parser: str) -> int:
    """Return the per-agent transport width a parser consumes before wheel conversion."""
    if action_parser not in ACTION_PARSERS:
        raise ValueError("unsupported action parser")
    if action_parser == "parametric_primitive":
        return 4
    return CircularPrimitiveSet.token_width if action_parser == "circular_primitive" else 2


def check_team_actions(actions: FloatArray, expected: tuple[int, ...], role: str) -> None:
    """Reject transport tokens that do not match the parser the environment was built with."""
    if actions.shape != expected:
        raise ValueError(
            f"{role} actions must have shape {expected} for this action parser, "
            f"received {actions.shape}"
        )


@dataclass(frozen=True)
class TeamReward:
    ball_progress: float
    ball_direction: float
    attacker_alignment: float
    time: float
    goal: float
    goal_geometry: float = 0.0
    action_delta: float = 0.0
    wheel_effort: float = 0.0
    teammate_congestion: float = 0.0
    defensive_coverage: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.ball_progress
            + self.ball_direction
            + self.attacker_alignment
            + self.time
            + self.goal
            + self.goal_geometry
            + self.action_delta
            + self.wheel_effort
            + self.teammate_congestion
            + self.defensive_coverage
        )


@dataclass(frozen=True)
class MarlEvaluation:
    seeds: int
    policy_progress: float
    random_progress: float
    margin: float
    passed: bool


@dataclass(frozen=True)
class ContactMetrics:
    ally_penalty: float
    opponent_penalty: float
    ally_streaks: NDArray[np.int64]
    opponent_streaks: NDArray[np.int64]
    ally_contacts: int
    opponent_contacts: int
    ally_deadlocks: int
    opponent_deadlocks: int
    escapes: int


class MarlMatchEnv:
    """Blue-team C7/C8 task with three decentralized actions."""

    def __init__(
        self,
        config_json: str,
        state_json: str,
        *,
        stage: int,
        horizon: int = 1_000,
        action_repeat: int = 4,
        action_delta_coefficient: float = 0.0,
        goal_coefficient: float = 10.0,
        progress_coefficient: float = 0.0,
        wheel_effort_coefficient: float = 0.0,
        ball_direction_coefficient: float = 0.0,
        goal_geometry_coefficient: float = 0.0,
        goal_geometry_discount: float = 0.99,
        attacker_alignment_coefficient: float = 0.0,
        time_penalty_coefficient: float = 0.0,
        movement_speed_threshold: float = 0.03,
        teammate_spacing: float = 0.14,
        teammate_congestion_coefficient: float = 0.0,
        defensive_coverage_coefficient: float = 0.0,
        defensive_activation_x: float = 0.15,
        draw_penalty: float = 0.0,
        stagnation_penalty: float = 0.0,
        stagnation_seconds: float = 5.0,
        stagnation_ball_distance: float = 0.02,
        action_parser: str = "continuous",
    ) -> None:
        if stage not in (7, 8):
            raise ValueError("stage must be C7 or C8")
        self._config = json.loads(config_json)
        self._max_wheel_speed = float(self._config["max_wheel_speed"])
        self._template = json.loads(state_json)
        self._native = BatchSimulator(config_json, state_json, 1)
        self._yellow = DynamicTeamController(3, -1)
        self.stage = stage
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.action_delta_coefficient = action_delta_coefficient
        self.goal_coefficient = goal_coefficient
        self.progress_coefficient = progress_coefficient
        self.wheel_effort_coefficient = wheel_effort_coefficient
        self.ball_direction_coefficient = ball_direction_coefficient
        self.goal_geometry_coefficient = goal_geometry_coefficient
        self.goal_geometry_discount = goal_geometry_discount
        self.attacker_alignment_coefficient = attacker_alignment_coefficient
        self.time_penalty_coefficient = time_penalty_coefficient
        self.movement_speed_threshold = movement_speed_threshold
        self.teammate_spacing = teammate_spacing
        self.teammate_congestion_coefficient = teammate_congestion_coefficient
        self.defensive_coverage_coefficient = defensive_coverage_coefficient
        self.defensive_activation_x = defensive_activation_x
        self.draw_penalty = draw_penalty
        self.stagnation_penalty = stagnation_penalty
        if action_parser not in ACTION_PARSERS:
            raise ValueError("unsupported action parser")
        self.action_parser = action_parser
        self._decision_period = float(self._config["timestep"]) * action_repeat
        self.stagnation_limit = max(1, round(stagnation_seconds / self._decision_period))
        self.stagnation_ball_distance = stagnation_ball_distance
        self.steps = 0
        self.state = np.zeros(BatchSimulator.state_width(), dtype=np.float32)
        self._ball_x = 0.0
        self._closest = 0.0
        self._initial_ball_x = 0.0
        self._initial_closest = 0.0
        self._previous_blue_actions = np.zeros((3, 2), dtype=np.float32)
        self._goal_grace_remaining: int | None = None
        self._defensive_distance = 0.0
        self._stagnation_anchor = np.zeros(2, dtype=np.float32)
        self._stagnation_steps = 0
        self._role_assigner = DynamicRoleAssigner()
        self._role_assignment: RoleAssignment | None = None
        self._goal_geometry_potential = 0.0

    def reset(self, seed: int) -> TeamBatch:
        snapshot = _seeded_snapshot(self._template, seed)
        return self.reset_state(snapshot)

    def reset_state(self, snapshot: dict[str, Any]) -> TeamBatch:
        """Restore one explicit scenario for deterministic skill evaluation."""
        self._native.restore(0, json.dumps(snapshot, separators=(",", ":")))
        self.state = self._native.step(np.zeros((1, 6, 2), dtype=np.float32))[0]
        self.steps = 0
        self._ball_x = float(self.state[5])
        self._closest = self._closest_blue_distance()
        self._previous_blue_actions.fill(0.0)
        self._goal_grace_remaining = None
        self._defensive_distance = _defensive_distance(self.state, self._config)
        self._stagnation_anchor = self.state[5:7].copy()
        self._stagnation_steps = 0
        self._role_assigner.reset()
        self._role_assignment = self._role_assigner.assign(self.state, 0)
        self._goal_geometry_potential = _goal_geometry_potential(self.state, self._config, 0)
        return build_team_observation(self.state, team=0, role_assignment=self._role_assignment)

    def step(
        self,
        blue_actions: FloatArray,
        opponent_actions: FloatArray | None = None,
    ) -> tuple[TeamBatch, TeamReward, bool, dict[str, Any]]:
        expected = (3, team_action_width(self.action_parser))
        check_team_actions(blue_actions, expected, "controlled team")
        if opponent_actions is not None:
            check_team_actions(opponent_actions, expected, "opponent team")
        normalized_blue = np.clip(blue_actions, -1.0, 1.0)
        if self.action_parser == "primitive":
            normalized_blue = primitive_wheel_actions(
                self.state,
                team=0,
                tokens=normalized_blue,
            )
        elif self.action_parser == "parametric_primitive":
            normalized_blue = parametric_primitive_wheel_actions(
                self.state,
                team=0,
                tokens=normalized_blue,
            )
        elif self.action_parser == "circular_primitive":
            normalized_blue = circular_primitive_wheel_actions(
                self.state,
                team=0,
                tokens=normalized_blue,
            )
        action_delta = normalized_blue - self._previous_blue_actions
        actions = np.zeros((1, 6, 2), dtype=np.float32)
        actions[0, :3] = normalized_blue * self._max_wheel_speed
        for slot in range(3):
            if not bool(float(self.state[ROBOT_BASE + slot * ROBOT_WIDTH + 10])):
                actions[0, slot] = 0.0
        # A learned opponent is parsed once per decision, exactly like the learner, so
        # evaluation and replay execute one token the way training executes it. Only the
        # scripted controller re-plans per physics substep, as it does in the vector env.
        if opponent_actions is not None:
            normalized_opponent = np.clip(opponent_actions, -1.0, 1.0)
            if self.action_parser == "primitive":
                normalized_opponent = primitive_wheel_actions(
                    self.state,
                    team=1,
                    tokens=normalized_opponent,
                )
            elif self.action_parser == "parametric_primitive":
                normalized_opponent = parametric_primitive_wheel_actions(
                    self.state,
                    team=1,
                    tokens=normalized_opponent,
                )
            elif self.action_parser == "circular_primitive":
                normalized_opponent = circular_primitive_wheel_actions(
                    self.state,
                    team=1,
                    tokens=normalized_opponent,
                )
            actions[0, 3:] = normalized_opponent * self._max_wheel_speed
        events = 0
        for _ in range(self.action_repeat):
            if opponent_actions is None and self.stage == 8:
                actions[0, 3:] = self._yellow.actions(self.state) * self._max_wheel_speed
            self.state = self._native.step(actions)[0]
            events |= int(self.state[-1])
        self.steps += 1
        ball_x = float(self.state[5])
        closest = self._closest_blue_distance()
        defensive_distance = _defensive_distance(self.state, self._config)
        threat = _defensive_threat(ball_x, self.defensive_activation_x)
        ball_displacement = math.dist(
            (ball_x, float(self.state[6])),
            (float(self._stagnation_anchor[0]), float(self._stagnation_anchor[1])),
        )
        if ball_displacement >= self.stagnation_ball_distance:
            self._stagnation_anchor = self.state[5:7].copy()
            self._stagnation_steps = 0
        else:
            self._stagnation_steps += 1
        if events & 0b11 and self._goal_grace_remaining is None:
            self._goal_grace_remaining = round(
                float(self._config["reset"]["goal_pause"]) / self._decision_period
            )
        goal_complete = False
        if self._goal_grace_remaining is not None:
            self._goal_grace_remaining -= 1
            goal_complete = self._goal_grace_remaining <= 0
        stagnated = (
            self._goal_grace_remaining is None and self._stagnation_steps >= self.stagnation_limit
        )
        draw = self.steps >= self.horizon and not goal_complete and not stagnated
        # Potential shaping is only policy-invariant when a terminal state carries no
        # potential; otherwise the last transition pays for how the episode ended.
        goal_geometry_potential = (
            0.0
            if goal_complete or stagnated or draw
            else _goal_geometry_potential(self.state, self._config, 0)
        )
        reward = TeamReward(
            ball_progress=self.progress_coefficient
            * (2.0 * (self._closest - closest) + (ball_x - self._ball_x)),
            ball_direction=self.ball_direction_coefficient
            * _ball_direction_reward(self.state, self._config, self.movement_speed_threshold)
            / self.horizon,
            attacker_alignment=self.attacker_alignment_coefficient
            * _attacker_alignment_reward(self.state, self.movement_speed_threshold)
            / self.horizon,
            time=-self.time_penalty_coefficient / self.horizon,
            goal=(
                self.goal_coefficient * float(bool(events & 1))
                - self.goal_coefficient * float(bool(events & 2))
                - self.draw_penalty * float(draw)
                - self.stagnation_penalty * float(stagnated)
            ),
            goal_geometry=self.goal_geometry_coefficient
            * (
                self.goal_geometry_discount * goal_geometry_potential
                - self._goal_geometry_potential
            ),
            action_delta=-self.action_delta_coefficient * float(np.square(action_delta).mean()),
            wheel_effort=-self.wheel_effort_coefficient * float(np.square(normalized_blue).mean()),
            teammate_congestion=-self.teammate_congestion_coefficient
            * _teammate_congestion(self.state, self.teammate_spacing, 0),
            defensive_coverage=self.defensive_coverage_coefficient
            * threat
            * (self._defensive_distance - defensive_distance),
        )
        self._previous_blue_actions = normalized_blue.copy()
        self._ball_x = ball_x
        self._closest = closest
        self._defensive_distance = defensive_distance
        self._goal_geometry_potential = goal_geometry_potential
        done = draw or goal_complete or stagnated
        terminal_reason = (
            "goal" if goal_complete else "stagnation" if stagnated else "draw" if draw else None
        )
        self._role_assignment = self._role_assigner.assign(self.state, 0)
        return (
            build_team_observation(self.state, team=0, role_assignment=self._role_assignment),
            reward,
            done,
            {
                "events": events,
                "goal_pending": self._goal_grace_remaining is not None,
                "terminal_reason": terminal_reason,
                "closest_ball_distance": closest,
                "ball_x": ball_x,
                "opponent_mode": (
                    "policy"
                    if opponent_actions is not None
                    else "heuristic"
                    if self.stage == 8
                    else "inactive"
                ),
                "actions": actions[0].copy(),
                "roles": list(self._role_assignment.roles),
                "role_changes": list(self._role_assignment.changed),
                "coverage_uncovered": self._role_assignment.uncovered,
                "goal_geometry": _goal_geometry_metrics(self.state, self._config, 0),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the current canonical lifecycle snapshot."""
        return cast(dict[str, Any], json.loads(self._native.snapshots()[0]))

    @property
    def decision_period(self) -> float:
        return self._decision_period

    def progress_score(self) -> float:
        return 2.0 * (self._initial_closest - self._closest) + (self._ball_x - self._initial_ball_x)

    def mark_progress_origin(self) -> None:
        self._initial_closest = self._closest
        self._initial_ball_x = self._ball_x

    def _closest_blue_distance(self) -> float:
        return min(
            math.hypot(
                float(self.state[5] - self.state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
                float(self.state[6] - self.state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
            )
            for slot in range(3)
            if bool(float(self.state[ROBOT_BASE + slot * ROBOT_WIDTH + 10]))
        )


class VectorMarlMatchEnv:
    """One native parallel batch of independent C7/C8 worlds."""

    def __init__(
        self,
        config_json: str,
        state_json: str,
        *,
        num_envs: int,
        stage: int,
        horizon: int,
        action_repeat: int,
        action_delta_coefficient: float,
        goal_coefficient: float,
        progress_coefficient: float,
        wheel_effort_coefficient: float,
        ball_direction_coefficient: float,
        useful_touch_impulse_coefficient: float,
        goal_geometry_coefficient: float,
        goal_geometry_discount: float,
        idle_spin_coefficient: float,
        idle_spin_grace_seconds: float,
        idle_spin_angular_speed: float,
        idle_spin_drive_threshold: float,
        idle_spin_speed_threshold: float,
        idle_spin_ball_distance: float,
        attacker_alignment_coefficient: float,
        time_penalty_coefficient: float,
        movement_speed_threshold: float,
        teammate_spacing: float,
        teammate_congestion_coefficient: float,
        contact_distance: float,
        contact_grace_seconds: float,
        ally_deadlock_coefficient: float,
        opponent_deadlock_coefficient: float,
        defensive_coverage_coefficient: float,
        defensive_activation_x: float,
        draw_penalty: float,
        stagnation_penalty: float,
        stagnation_seconds: float,
        stagnation_ball_distance: float,
        action_parser: str = "continuous",
        free_ball_seconds: float = 10.0,
        free_ball_clearance: float = 0.20,
    ) -> None:
        if stage not in (7, 8):
            raise ValueError("stage must be C7 or C8")
        self._config = json.loads(config_json)
        self._template = json.loads(state_json)
        self._native = BatchSimulator(config_json, state_json, num_envs)
        self._blue = DynamicTeamController(0, 1)
        self._yellow = DynamicTeamController(3, -1)
        self._max_wheel_speed = float(self._config["max_wheel_speed"])
        self.num_envs = num_envs
        self.stage = stage
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.action_delta_coefficient = action_delta_coefficient
        self.goal_coefficient = goal_coefficient
        self.progress_coefficient = progress_coefficient
        self.wheel_effort_coefficient = wheel_effort_coefficient
        self.ball_direction_coefficient = ball_direction_coefficient
        self.useful_touch_impulse_coefficient = useful_touch_impulse_coefficient
        self.goal_geometry_coefficient = goal_geometry_coefficient
        self.goal_geometry_discount = goal_geometry_discount
        self._decision_period = float(self._config["timestep"]) * action_repeat
        self.idle_spin_coefficient = idle_spin_coefficient
        self.idle_spin_grace_steps = max(1, round(idle_spin_grace_seconds / self._decision_period))
        self.idle_spin_angular_speed = idle_spin_angular_speed
        self.idle_spin_drive_threshold = idle_spin_drive_threshold
        self.idle_spin_speed_threshold = idle_spin_speed_threshold
        self.idle_spin_ball_distance = idle_spin_ball_distance
        self.attacker_alignment_coefficient = attacker_alignment_coefficient
        self.time_penalty_coefficient = time_penalty_coefficient
        self.movement_speed_threshold = movement_speed_threshold
        self.teammate_spacing = teammate_spacing
        self.teammate_congestion_coefficient = teammate_congestion_coefficient
        self.contact_distance = contact_distance
        self.contact_grace_steps = max(1, round(contact_grace_seconds / self._decision_period))
        self.ally_deadlock_coefficient = ally_deadlock_coefficient
        self.opponent_deadlock_coefficient = opponent_deadlock_coefficient
        self.defensive_coverage_coefficient = defensive_coverage_coefficient
        self.defensive_activation_x = defensive_activation_x
        self.draw_penalty = draw_penalty
        self.stagnation_penalty = stagnation_penalty
        if action_parser not in ACTION_PARSERS:
            raise ValueError("unsupported action parser")
        self.action_parser = action_parser
        self.stagnation_limit = max(1, round(stagnation_seconds / self._decision_period))
        self.stagnation_ball_distance = stagnation_ball_distance
        self.free_ball_limit = max(1, round(free_ball_seconds / self._decision_period))
        self.free_ball_clearance = free_ball_clearance
        self.free_balls = np.zeros(num_envs, dtype=np.int64)
        self.states = np.zeros((num_envs, BatchSimulator.state_width()), dtype=np.float32)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self._ball_x = np.zeros(num_envs, dtype=np.float32)
        self._closest = np.zeros(num_envs, dtype=np.float32)
        self._initial_ball_x = np.zeros(num_envs, dtype=np.float32)
        self._initial_closest = np.zeros(num_envs, dtype=np.float32)
        self._previous_blue_actions = np.zeros((num_envs, 3, 2), dtype=np.float32)
        self._normalized_blue_actions = np.zeros((num_envs, 3, 2), dtype=np.float32)
        self._action_delta = np.zeros((num_envs, 3, 2), dtype=np.float32)
        self._native_actions = np.zeros((num_envs, 6, 2), dtype=np.float32)
        self._events = np.zeros(num_envs, dtype=np.int64)
        self._episode_goal_events = np.zeros(num_envs, dtype=np.int64)
        self._goal_grace_remaining = np.full(num_envs, -1, dtype=np.int64)
        self._defensive_distance = np.zeros(num_envs, dtype=np.float32)
        self._stagnation_anchor = np.zeros((num_envs, 2), dtype=np.float32)
        self._stagnation_steps = np.zeros(num_envs, dtype=np.int64)
        self._previous_ball_positions = np.zeros((num_envs, 2), dtype=np.float32)
        self._previous_ball_velocities = np.zeros((num_envs, 2), dtype=np.float32)
        self._controlled_ball_contact = np.zeros(num_envs, dtype=np.bool_)
        self._goal_geometry_potential = np.zeros(num_envs, dtype=np.float32)
        self._idle_spin_streaks = np.zeros((num_envs, 3), dtype=np.int64)
        self._ally_contact_streaks = np.zeros((num_envs, 3), dtype=np.int64)
        self._opponent_contact_streaks = np.zeros((num_envs, 9), dtype=np.int64)
        self.ally_contact_steps = np.zeros(num_envs, dtype=np.int64)
        self.reward_terms: dict[str, float] = {}
        self.reward_decisions = 0
        self.opponent_contact_steps = np.zeros(num_envs, dtype=np.int64)
        self.ally_deadlocks = np.zeros(num_envs, dtype=np.int64)
        self.opponent_deadlocks = np.zeros(num_envs, dtype=np.int64)
        self.contact_escapes = np.zeros(num_envs, dtype=np.int64)
        self.idle_spin_steps = np.zeros(num_envs, dtype=np.int64)
        self.active_agent_decisions = np.zeros(num_envs, dtype=np.int64)
        self.last_terminal_reasons = np.full(num_envs, "", dtype="<U24")
        self.controlled_teams = np.zeros(num_envs, dtype=np.int64)
        self._role_assigners = [DynamicRoleAssigner() for _ in range(num_envs)]
        self.role_assignments: list[RoleAssignment | None] = [None] * num_envs
        self.role_switches = np.zeros(num_envs, dtype=np.int64)
        self.uncovered_steps = np.zeros(num_envs, dtype=np.int64)
        self.role_decisions = np.zeros(num_envs, dtype=np.int64)

    def reset_reward_terms(self) -> None:
        """Scope reward accounting to one rollout so contributions stay comparable."""
        self.reward_terms = {}
        self.reward_decisions = 0

    def set_controlled_team(self, world: int, team: int) -> None:
        if team not in (0, 1):
            raise ValueError("controlled team must be 0 or 1")
        self.controlled_teams[world] = team

    def reset(self, world: int, seed: int) -> TeamBatch:
        snapshot = _seeded_snapshot(self._template, seed)
        return self.reset_state(world, snapshot)

    def reset_state(self, world: int, snapshot: dict[str, Any]) -> TeamBatch:
        """Restore one validated scenario without disturbing neighboring worlds."""
        self.states[world] = self._native.restore_state(
            world, json.dumps(snapshot, separators=(",", ":"))
        )
        self.steps[world] = 0
        self._ball_x[world] = self.states[world, 5]
        team = int(self.controlled_teams[world])
        self._closest[world] = _closest_team_distance(self.states[world], team)
        self._initial_ball_x[world] = self._ball_x[world]
        self._initial_closest[world] = self._closest[world]
        self._previous_blue_actions[world].fill(0.0)
        self._goal_grace_remaining[world] = -1
        self._episode_goal_events[world] = 0
        self._defensive_distance[world] = _defensive_distance(
            self.states[world], self._config, team
        )
        self._stagnation_anchor[world] = self.states[world, 5:7]
        self._previous_ball_positions[world] = self.states[world, 5:7]
        self._previous_ball_velocities[world] = self.states[world, 7:9]
        self._controlled_ball_contact[world] = _team_touches_ball(
            self.states[world],
            int(self.controlled_teams[world]),
            self._config,
        )
        self._goal_geometry_potential[world] = _goal_geometry_potential(
            self.states[world], self._config, team
        )
        self._idle_spin_streaks[world].fill(0)
        self._stagnation_steps[world] = 0
        self._ally_contact_streaks[world].fill(0)
        self._opponent_contact_streaks[world].fill(0)
        self.last_terminal_reasons[world] = ""
        # The hysteresis now lives in the native simulator, so restarting it there is what an
        # episode boundary means. Resetting a Python assigner instead would leave the two
        # histories to diverge silently, which no shape or value check would catch.
        _, changed, uncovered, cost, names = self._native.restart_roles(world, team)
        assignment = RoleAssignment(
            cast("tuple[Role, Role, Role]", tuple(names[0])),
            cast("tuple[bool, bool, bool]", tuple(bool(flag) for flag in changed[0])),
            float(cost[0]),
            bool(uncovered[0]),
        )
        self.role_assignments[world] = assignment
        return build_team_observation(self.states[world], team=team, role_assignment=assignment)

    def _native_team_scalars(self) -> NDArray[np.float64]:
        """Return the ball-touch flag, the closest distance, the congestion and post distance."""
        return np.asarray(
            self._native.team_scalars(
                self.controlled_teams,
                (
                    float(self._config["field"]["length"]),
                    float(self._config["field"]["goal_width"]),
                    float(self._config["robot"]["length"]),
                    float(self._config["robot"]["width"]),
                    float(self._config["ball"]["radius"]),
                    self.teammate_spacing,
                ),
                self.movement_speed_threshold,
            )
        )

    def _native_goal_geometry(self) -> NDArray[np.float64]:
        """Describe every world's attacking line, potential first, then its four components."""
        return np.asarray(
            self._native.goal_geometry(
                self.controlled_teams,
                float(self._config["field"]["length"]),
                float(self._config["field"]["goal_width"]),
                float(self._config["ball"]["radius"]),
            )
        )

    def step(
        self,
        blue_actions: FloatArray,
        opponent_actions: FloatArray | None,
    ) -> tuple[
        TeamBatch,
        FloatArray,
        NDArray[np.bool_],
        NDArray[np.int64],
        NDArray[np.bool_],
    ]:
        expected = (self.num_envs, 3, team_action_width(self.action_parser))
        check_team_actions(blue_actions, expected, "controlled team")
        if opponent_actions is not None:
            check_team_actions(opponent_actions, expected, "opponent team")
        if self.action_parser == "circular_primitive":
            normalized_blue = self._normalized_blue_actions
            np.copyto(
                normalized_blue,
                self._native.circular_wheel_actions(
                    self.controlled_teams,
                    np.clip(blue_actions, -1.0, 1.0),
                    CIRCULAR_BALL_DECELERATION,
                ),
            )
        elif self.action_parser == "parametric_primitive":
            primitive_tokens = np.clip(blue_actions, -1.0, 1.0)
            normalized_blue = self._normalized_blue_actions
            for world, team in enumerate(self.controlled_teams):
                normalized_blue[world] = parametric_primitive_wheel_actions(
                    self.states[world],
                    team=int(team),
                    tokens=primitive_tokens[world],
                )
        else:
            normalized_blue = np.clip(
                blue_actions,
                -1.0,
                1.0,
                out=self._normalized_blue_actions,
            )
        if self.action_parser == "primitive":
            for world, team in enumerate(self.controlled_teams):
                normalized_blue[world] = primitive_wheel_actions(
                    self.states[world],
                    team=int(team),
                    tokens=normalized_blue[world].copy(),
                )
        action_delta = np.subtract(
            normalized_blue,
            self._previous_blue_actions,
            out=self._action_delta,
        )
        actions = self._native_actions
        actions.fill(0.0)
        for world, team in enumerate(self.controlled_teams):
            learner_slice = slice(0, 3) if team == 0 else slice(3, 6)
            actions[world, learner_slice] = normalized_blue[world] * self._max_wheel_speed
            offset = 0 if team == 0 else 3
            for local_slot in range(3):
                if not bool(
                    float(self.states[world, ROBOT_BASE + (offset + local_slot) * ROBOT_WIDTH + 10])
                ):
                    actions[world, offset + local_slot] = 0.0
        events = self._events
        events.fill(0)
        if opponent_actions is not None:
            normalized_opponents = np.clip(opponent_actions, -1.0, 1.0)
            for world, team in enumerate(self.controlled_teams):
                opponent_slice = slice(3, 6) if team == 0 else slice(0, 3)
                opponent_team = 1 - int(team)
                parsed_opponent = (
                    primitive_wheel_actions(
                        self.states[world],
                        team=opponent_team,
                        tokens=normalized_opponents[world],
                    )
                    if self.action_parser == "primitive"
                    else parametric_primitive_wheel_actions(
                        self.states[world],
                        team=opponent_team,
                        tokens=normalized_opponents[world],
                    )
                    if self.action_parser == "parametric_primitive"
                    else circular_primitive_wheel_actions(
                        self.states[world],
                        team=opponent_team,
                        tokens=normalized_opponents[world],
                    )
                    if self.action_parser == "circular_primitive"
                    else normalized_opponents[world]
                )
                actions[world, opponent_slice] = parsed_opponent * self._max_wheel_speed
            self.states = self._native.step_repeated(actions, self.action_repeat)
            events |= self.states[:, -1].astype(np.int64)
        elif self.stage == 8:
            # The scripted opponent is a controller, not an oracle: it plans once per
            # decision like everything else on the field. Re-planning every physics substep
            # ran it at 200 Hz against the learner's 50 Hz, which the configured control
            # period does not allow, and it cost a quarter of all rollout time.
            scripted = self._native.scripted_actions(1 - self.controlled_teams)
            scripted = scripted * self._max_wheel_speed
            blue_controlled = self.controlled_teams == 0
            actions[blue_controlled, 3:] = scripted[blue_controlled]
            actions[~blue_controlled, :3] = scripted[~blue_controlled]
            # The command is constant across the repeat now, so the substeps belong in the
            # native loop rather than four round trips through Python.
            self.states = self._native.step_repeated(actions, self.action_repeat)
            events |= self.states[:, -1].astype(np.int64)
        else:
            self.states = self._native.step_repeated(actions, self.action_repeat)
            events |= self.states[:, -1].astype(np.int64)
        self.steps += 1
        ball_x = self.states[:, 5]
        # Four scalars over the same six robot rows, so one native pass answers all of them.
        scalars = self._native_team_scalars()
        closest = scalars[:, 1].astype(np.float32)
        defensive_distance = scalars[:, 3].astype(np.float32)
        threat = np.asarray(
            [
                _defensive_threat(
                    float(x),
                    self.defensive_activation_x,
                    int(team),
                )
                for x, team in zip(ball_x, self.controlled_teams, strict=True)
            ],
            dtype=np.float32,
        )
        ball_positions = self.states[:, 5:7]
        ball_displacement = np.linalg.norm(ball_positions - self._stagnation_anchor, axis=1)
        moved = ball_displacement >= self.stagnation_ball_distance
        self._stagnation_anchor[moved] = ball_positions[moved]
        self._stagnation_steps[moved] = 0
        self._stagnation_steps[~moved] += 1
        congestion = scalars[:, 2].astype(np.float32)
        ally_streaks, opponent_streaks, contact_summary = self._native.contacts(
            self.controlled_teams,
            self._previous_ball_positions,
            self._ally_contact_streaks,
            self._opponent_contact_streaks,
            self.contact_distance,
            self.contact_grace_steps,
            self.movement_speed_threshold * self._decision_period,
            (
                float(self._config["robot"]["length"]),
                float(self._config["robot"]["width"]),
                float(self._config["ball"]["radius"]),
            ),
        )
        ally_deadlock_penalty = contact_summary[:, 0].astype(np.float32)
        opponent_deadlock_penalty = contact_summary[:, 1].astype(np.float32)
        self._ally_contact_streaks = ally_streaks
        self._opponent_contact_streaks = opponent_streaks
        self.ally_contact_steps += contact_summary[:, 2].astype(np.int64)
        self.opponent_contact_steps += contact_summary[:, 3].astype(np.int64)
        self.ally_deadlocks += contact_summary[:, 4].astype(np.int64)
        self.opponent_deadlocks += contact_summary[:, 5].astype(np.int64)
        self.contact_escapes += contact_summary[:, 6].astype(np.int64)
        newly_scored = ((events & 0b11) != 0) & (self._goal_grace_remaining < 0)
        self._episode_goal_events[newly_scored] = events[newly_scored] & 0b11
        self._goal_grace_remaining[newly_scored] = round(
            float(self._config["reset"]["goal_pause"]) / self._decision_period
        )
        active_grace = self._goal_grace_remaining >= 0
        self._goal_grace_remaining[active_grace] -= 1
        goal_complete = active_grace & (self._goal_grace_remaining <= 0)
        # Rule 15: an impasse away from both goal areas is resolved by placing the ball on
        # the quadrant's free-ball mark and continuing, not by ending the game. Ending it and
        # charging a penalty taught the policy that a stalled ball is a loss, and four of six
        # episodes in one capture died at exactly the old five-second limit.
        impasse = (self._goal_grace_remaining < 0) & (
            self._stagnation_steps >= self.free_ball_limit
        )
        for stalled in np.flatnonzero(impasse):
            self._restart_free_ball(int(stalled))
        stagnated = np.zeros(self.num_envs, dtype=np.bool_)
        draw = (self.steps >= self.horizon) & ~goal_complete & ~stagnated
        attack_sign = np.where(self.controlled_teams == 0, 1.0, -1.0)
        current_ball_contact = scalars[:, 0].astype(np.bool_)
        useful_touch_impulse = np.asarray(
            [
                _useful_touch_impulse(
                    float(state[7]),
                    float(previous_velocity[0]),
                    int(team),
                    bool(contact),
                    bool(previous_contact),
                )
                for state, previous_velocity, team, contact, previous_contact in zip(
                    self.states,
                    self._previous_ball_velocities,
                    self.controlled_teams,
                    current_ball_contact,
                    self._controlled_ball_contact,
                    strict=True,
                )
            ],
            dtype=np.float32,
        )
        # Potential shaping is only policy-invariant when a terminal state carries no
        # potential; otherwise the last transition pays for how the episode ended.
        goal_geometry_potential = np.where(
            goal_complete | stagnated | draw,
            0.0,
            self._native_goal_geometry()[:, 0],
        ).astype(np.float32)
        spin_flags, turn_intensity = self._native.idle_spin(
            self.controlled_teams,
            normalized_blue,
            self.idle_spin_angular_speed,
            self.idle_spin_drive_threshold,
            self.idle_spin_speed_threshold,
            self.idle_spin_ball_distance,
        )
        self._idle_spin_streaks = np.where(spin_flags, self._idle_spin_streaks + 1, 0)
        penalized = spin_flags & (self._idle_spin_streaks > self.idle_spin_grace_steps)
        idle_spin_penalty = np.where(penalized, turn_intensity, 0.0).mean(axis=1).astype(np.float32)
        self.idle_spin_steps += spin_flags.sum(axis=1)
        team_offsets = np.where(self.controlled_teams == 0, 0, 3)
        slots = team_offsets[:, None] + np.arange(3)[None, :]
        enabled_columns = ROBOT_BASE + slots * ROBOT_WIDTH + 10
        self.active_agent_decisions += (
            np.take_along_axis(self.states, enabled_columns, axis=1) != 0.0
        ).sum(axis=1)
        scored = np.where(self.controlled_teams == 0, (events & 1) != 0, (events & 2) != 0)
        conceded = np.where(
            self.controlled_teams == 0,
            (events & 2) != 0,
            (events & 1) != 0,
        )
        # Named terms summed in their original order, so the total is bit-identical to the
        # single expression this replaces while each contribution stays accountable. Only
        # the total was ever recorded, which made it impossible to say which term a policy
        # was actually optimizing.
        terms: dict[str, FloatArray] = {
            "progress": (
                self.progress_coefficient
                * (2.0 * (self._closest - closest) + attack_sign * (ball_x - self._ball_x))
            ).astype(np.float32),
            "ball_direction": (
                self.ball_direction_coefficient * scalars[:, 5].astype(np.float32) / self.horizon
            ).astype(np.float32),
            "useful_touch": (
                self.useful_touch_impulse_coefficient * np.tanh(useful_touch_impulse)
            ).astype(np.float32),
            "goal_geometry": (
                self.goal_geometry_coefficient
                * (
                    self.goal_geometry_discount * goal_geometry_potential
                    - self._goal_geometry_potential
                )
            ).astype(np.float32),
            "idle_spin": (-self.idle_spin_coefficient * idle_spin_penalty).astype(np.float32),
            "attacker_alignment": (
                self.attacker_alignment_coefficient
                * scalars[:, 4].astype(np.float32)
                / self.horizon
            ).astype(np.float32),
            "time": np.full(
                self.num_envs,
                -self.time_penalty_coefficient / self.horizon,
                dtype=np.float32,
            ),
            "goal_scored": (self.goal_coefficient * scored).astype(np.float32),
            "goal_conceded": (-self.goal_coefficient * conceded).astype(np.float32),
            "action_delta": (
                -self.action_delta_coefficient * np.square(action_delta).mean(axis=(1, 2))
            ).astype(np.float32),
            "wheel_effort": (
                -self.wheel_effort_coefficient * np.square(normalized_blue).mean(axis=(1, 2))
            ).astype(np.float32),
            "teammate_congestion": (-self.teammate_congestion_coefficient * congestion).astype(
                np.float32
            ),
            "ally_deadlock": (-self.ally_deadlock_coefficient * ally_deadlock_penalty).astype(
                np.float32
            ),
            "opponent_deadlock": (
                -self.opponent_deadlock_coefficient * opponent_deadlock_penalty
            ).astype(np.float32),
            "defensive_coverage": (
                self.defensive_coverage_coefficient
                * threat
                * (self._defensive_distance - defensive_distance)
            ).astype(np.float32),
            "draw": (-self.draw_penalty * draw).astype(np.float32),
            "stagnation": (-self.stagnation_penalty * stagnated).astype(np.float32),
        }
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        for name, value in terms.items():
            rewards = rewards + value
            self.reward_terms[name] = self.reward_terms.get(name, 0.0) + float(value.sum())
        self.reward_decisions += self.num_envs
        rewards = rewards.astype(np.float32)
        np.copyto(self._previous_blue_actions, normalized_blue)
        np.copyto(self._ball_x, ball_x)
        np.copyto(self._previous_ball_positions, ball_positions)
        np.copyto(self._previous_ball_velocities, self.states[:, 7:9])
        np.copyto(self._controlled_ball_contact, current_ball_contact)
        np.copyto(self._goal_geometry_potential, goal_geometry_potential)
        self._closest = closest
        self._defensive_distance = defensive_distance
        done = draw | goal_complete | stagnated
        self.last_terminal_reasons.fill("")
        self.last_terminal_reasons[draw] = "draw"
        self.last_terminal_reasons[stagnated] = "stagnation"
        self.last_terminal_reasons[goal_complete] = "goal"
        observations, native_assignments = _native_team_features(
            self._native, self.controlled_teams, self._config, hysteretic=True
        )
        assignments: list[RoleAssignment | None] = list(native_assignments)
        self.role_assignments = assignments
        self.role_switches += np.asarray(
            [sum(assignment.changed) for assignment in native_assignments],
            dtype=np.int64,
        )
        self.uncovered_steps += np.asarray(
            [assignment.uncovered for assignment in native_assignments],
            dtype=np.int64,
        )
        self.role_decisions += 1
        # Episode completion and value-bootstrap termination have different
        # semantics for timeouts. Never alias them: the collector may mark a
        # skill timeout as truncated without cancelling the required reset.
        reported_events = events.copy()
        reported_events[goal_complete] |= self._episode_goal_events[goal_complete]
        return observations, rewards, done, reported_events, done.copy()

    def mark_progress_origin(self) -> None:
        self._initial_closest = self._closest.copy()
        self._initial_ball_x = self._ball_x.copy()

    def progress_scores(self) -> FloatArray:
        scores = 2.0 * (self._initial_closest - self._closest) + np.where(
            self.controlled_teams == 0,
            1.0,
            -1.0,
        ) * (self._ball_x - self._initial_ball_x)
        return np.asarray(scores, dtype=np.float32)

    def _restart_free_ball(self, world: int) -> None:
        """Place the ball on the quadrant's free-ball mark and let play continue."""
        snapshot = self.snapshot(world)
        ball = snapshot["ball"]
        if (
            abs(float(ball["x"])) >= float(self._config["field"]["length"]) / 2 - GOAL_AREA_DEPTH
            and abs(float(ball["y"])) <= GOAL_AREA_HALF_WIDTH
        ):
            # Inside a goal area this is a goal kick, which is not modelled. Restart the
            # impasse clock so the ball is not repositioned out of the area by this rule.
            self._stagnation_steps[world] = 0
            return
        mark_x = math.copysign(FREE_BALL_X, float(ball["x"]) or 1.0)
        mark_y = math.copysign(FREE_BALL_Y, float(ball["y"]) or 1.0)
        ball.update(x=mark_x, y=mark_y, vx=0.0, vy=0.0, omega=0.0)
        for robot in snapshot["robots"]:
            pose = robot["pose"]
            if math.dist((pose["x"], pose["y"]), (mark_x, mark_y)) < self.free_ball_clearance:
                own_sign = -1.0 if robot["team"] == "blue" else 1.0
                pose["x"] = own_sign * FREE_BALL_X
                pose["y"] = -mark_y
            robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
            robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0)
        self.states[world] = self._native.restore_state(
            world,
            json.dumps(snapshot, separators=(",", ":")),
        )
        self._stagnation_anchor[world] = self.states[world, 5:7]
        self._stagnation_steps[world] = 0
        self.free_balls[world] += 1

    def snapshot(self, world: int) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._native.snapshots()[world]))

    @property
    def decision_period(self) -> float:
        return self._decision_period


def _seeded_snapshot(template: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    snapshot = copy.deepcopy(template)
    snapshot.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    snapshot["ball"].update(
        x=float(rng.uniform(-0.15, 0.25)),
        y=float(rng.uniform(-0.35, 0.35)),
        vx=0.0,
        vy=0.0,
        omega=0.0,
    )
    starts = (
        (-0.52, 0.0),
        (-0.35, 0.30),
        (-0.35, -0.30),
        (0.52, 0.0),
        (0.35, 0.30),
        (0.35, -0.30),
    )
    for robot, (x, y) in zip(snapshot["robots"], starts, strict=True):
        robot["pose"].update(
            x=float(x + rng.uniform(-0.04, 0.04)),
            y=float(y + rng.uniform(-0.04, 0.04)),
            theta=float(rng.uniform(-0.25, 0.25)),
        )
        robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
        robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0)
    return snapshot


def _team_touches_ball(
    state: FloatArray,
    team: int,
    config: dict[str, Any],
) -> bool:
    """Return whether an enabled team robot overlaps the ball contact envelope."""
    robot = config["robot"]
    robot_radius = math.hypot(float(robot["length"]), float(robot["width"])) / 2.0
    contact_radius = robot_radius + float(config["ball"]["radius"]) + 0.002
    offset = 0 if team == 0 else 3
    ball_x, ball_y = float(state[5]), float(state[6])
    return any(
        bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10]))
        and math.hypot(
            ball_x - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            ball_y - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        )
        <= contact_radius
        for slot in range(offset, offset + 3)
    )


def _useful_touch_impulse(
    ball_vx: float,
    previous_ball_vx: float,
    team: int,
    contact: bool,
    previous_contact: bool,
) -> float:
    """Measure new contact's signed ball-velocity delta toward the enemy goal.

    The contact edge keeps this an impulse rather than a rate, so contact cannot become
    the dense ball-advancement reward M20 deliberately removed. Keeping the sign is what
    stops the trigger from being farmed: a robot flickering at the envelope boundary
    re-enters on velocity noise, which integrates to zero when signed and accumulates
    when clamped to the positive side.
    """
    if not contact or previous_contact:
        return 0.0
    attack_sign = 1.0 if team == 0 else -1.0
    return attack_sign * (ball_vx - previous_ball_vx)


def _closest_team_distance(state: FloatArray, team: int = 0) -> float:
    offset = 0 if team == 0 else 3
    active = [
        slot
        for slot in range(offset, offset + 3)
        if bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10]))
    ]
    if not active:
        raise ValueError("controlled team must have at least one active robot")
    return min(
        math.hypot(
            float(state[5] - state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            float(state[6] - state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        )
        for slot in active
    )


def _teammate_congestion(state: FloatArray, spacing: float, team: int = 0) -> float:
    offset = 0 if team == 0 else 3
    positions = [
        (
            float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        )
        for slot in range(offset, offset + 3)
        if bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10]))
    ]
    penalties = [
        max(0.0, (spacing - math.dist(positions[first], positions[second])) / spacing) ** 2
        for first in range(len(positions))
        for second in range(first + 1, len(positions))
    ]
    return sum(penalties) / len(penalties) if penalties else 0.0


def _contact_deadlock_metrics(
    state: FloatArray,
    team: int,
    *,
    contact_distance: float,
    grace_steps: int,
    ally_streaks: NDArray[np.int64],
    opponent_streaks: NDArray[np.int64],
    previous_ball: FloatArray,
    meaningful_ball_displacement: float,
    config: dict[str, Any],
) -> ContactMetrics:
    """Measure sustained contacts without punishing brief or productive challenges."""
    controlled = tuple(range(0, 3)) if team == 0 else tuple(range(3, 6))
    opponents = tuple(range(3, 6)) if team == 0 else tuple(range(0, 3))
    ally_pairs = (
        (controlled[0], controlled[1]),
        (controlled[0], controlled[2]),
        (controlled[1], controlled[2]),
    )
    opponent_pairs = tuple((ally, rival) for ally in controlled for rival in opponents)
    ball = (float(state[5]), float(state[6]))
    ball_moved = math.dist(ball, (float(previous_ball[0]), float(previous_ball[1])))
    robot = config["robot"]
    ball_contact_distance = (
        math.hypot(float(robot["length"]), float(robot["width"])) / 2
        + float(config["ball"]["radius"])
        + 0.002
    )

    def active(slot: int) -> bool:
        return bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10]))

    def position(slot: int) -> tuple[float, float]:
        base = ROBOT_BASE + slot * ROBOT_WIDTH
        return float(state[base + 2]), float(state[base + 3])

    def update(
        pairs: tuple[tuple[int, int], ...],
        previous: NDArray[np.int64],
        *,
        preserve_ball_challenges: bool,
        moving_ball_is_productive: bool,
    ) -> tuple[NDArray[np.int64], int, int, int, float]:
        current = np.zeros_like(previous)
        contacts = deadlocks = escapes = 0
        penalties: list[float] = []
        for index, (first, second) in enumerate(pairs):
            touching = (
                active(first)
                and active(second)
                and math.dist(position(first), position(second)) <= contact_distance
            )
            if touching:
                contacts += 1
                current[index] = previous[index] + 1
                if current[index] == grace_steps + 1:
                    deadlocks += 1
                ball_involved = (
                    math.dist(position(first), ball) <= ball_contact_distance
                    or math.dist(position(second), ball) <= ball_contact_distance
                )
                productive = preserve_ball_challenges and (
                    ball_involved
                    or (moving_ball_is_productive and ball_moved >= meaningful_ball_displacement)
                )
                if current[index] > grace_steps and not productive:
                    penalties.append(min(1.0, (current[index] - grace_steps) / grace_steps))
            elif previous[index] > grace_steps:
                escapes += 1
        penalty = sum(penalties) / max(1, len(pairs))
        return current, contacts, deadlocks, escapes, penalty

    ally = update(
        ally_pairs,
        ally_streaks,
        preserve_ball_challenges=True,
        moving_ball_is_productive=False,
    )
    rival = update(
        opponent_pairs,
        opponent_streaks,
        preserve_ball_challenges=True,
        moving_ball_is_productive=True,
    )
    return ContactMetrics(
        ally_penalty=ally[4],
        opponent_penalty=rival[4],
        ally_streaks=ally[0],
        opponent_streaks=rival[0],
        ally_contacts=ally[1],
        opponent_contacts=rival[1],
        ally_deadlocks=ally[2],
        opponent_deadlocks=rival[2],
        escapes=ally[3] + rival[3],
    )


def _cosine_similarity(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return 0.0
    similarity = (first[0] * second[0] + first[1] * second[1]) / (first_norm * second_norm)
    return float(np.clip(similarity, -1.0, 1.0))


def _ball_direction_reward(
    state: FloatArray,
    config: dict[str, Any],
    speed_threshold: float,
    team: int = 0,
) -> float:
    velocity = (float(state[7]), float(state[8]))
    if math.hypot(*velocity) < speed_threshold:
        return 0.0
    ball = (float(state[5]), float(state[6]))
    attack_sign = 1.0 if team == 0 else -1.0
    goal_x = attack_sign * float(config["field"]["length"]) / 2.0
    enemy = (goal_x - ball[0], -ball[1])
    ally = (-goal_x - ball[0], -ball[1])
    enemy_similarity = _cosine_similarity(enemy, velocity)
    ally_similarity = _cosine_similarity(ally, velocity)
    return math.tanh(enemy_similarity) - math.tanh(ally_similarity)


#: Ball deceleration the circular executor plans an intercept against, in metres per second
#: squared. The Python reference carries it as a default argument; the native call takes it
#: explicitly, so the value has to be named somewhere the two agree on.
CIRCULAR_BALL_DECELERATION = 0.8

#: Shapes the native observation groups are folded back into, per world.
_GROUP_SHAPES = ((3, 8), (3, 7), (3, 4), (3, 9), (3, 2, 6), (3, 3, 6))


def _native_team_features(
    simulator: BatchSimulator,
    teams: NDArray[np.int64],
    config: dict[str, Any],
    *,
    hysteretic: bool,
) -> tuple[TeamBatch, list[RoleAssignment]]:
    """Assign roles and build every world's observation without a Python loop over worlds.

    The two are computed together because the observation consumes the role features, and
    keeping them in one place is what lets the whole per-world construction stay native. The
    `RoleAssignment` objects are still returned: single-world callers rebuild an observation
    from them, and they are cheap next to the permutation search they no longer perform.
    """
    worlds = len(teams)
    features, changed, uncovered, cost, names = simulator.team_roles(teams, hysteretic)
    groups = simulator.observations(
        teams,
        features,
        float(config["field"]["length"]),
        float(config["field"]["width"]),
        float(config["match_duration"]),
    )
    observations = TeamBatch(
        *(
            torch.from_numpy(np.ascontiguousarray(group)).reshape(worlds, *shape)
            for group, shape in zip(groups, _GROUP_SHAPES, strict=True)
        )
    )
    assignments = [
        RoleAssignment(
            cast("tuple[Role, Role, Role]", tuple(names[world])),
            cast("tuple[bool, bool, bool]", tuple(bool(flag) for flag in changed[world])),
            float(cost[world]),
            bool(uncovered[world]),
        )
        for world in range(worlds)
    ]
    return observations, assignments


def _goal_geometry_metrics(
    state: FloatArray,
    config: dict[str, Any],
    team: int = 0,
) -> dict[str, float]:
    """Describe a controllable attacking line without declaring field zones good or bad."""
    # The environment already holds a hysteretic assignment for the same state, and reusing it
    # here would look like the obvious saving. It would break the shaping: a potential must be a
    # function of the state alone, and the hysteretic assignment depends on the previous one.
    # The two disagree on about seven per cent of decisions, so this is not a duplicate call.
    assignment = assign_roles(state, team)
    local_slot = assignment.roles.index("attacker")
    slot = local_slot + (0 if team == 0 else 3)
    base = ROBOT_BASE + slot * ROBOT_WIDTH
    robot = (float(state[base + 2]), float(state[base + 3]))
    ball = (float(state[5]), float(state[6]))
    attack_sign = 1.0 if team == 0 else -1.0
    goal_x = attack_sign * float(config["field"]["length"]) / 2.0
    to_ball = (ball[0] - robot[0], ball[1] - robot[1])
    ball_to_goal = (goal_x - ball[0], -ball[1])
    alignment = 0.5 * (_cosine_similarity(to_ball, ball_to_goal) + 1.0)

    forward_separation = attack_sign * to_ball[0]
    usable_half_goal = max(
        1e-6,
        float(config["field"]["goal_width"]) / 2.0 - float(config["ball"]["radius"]),
    )
    if forward_separation <= 1e-6:
        aperture = 0.0
    else:
        goal_intersection_y = ball[1] + (ball[1] - robot[1]) * (
            abs(goal_x - ball[0]) / forward_separation
        )
        aperture = float(np.clip(1.0 - abs(goal_intersection_y) / usable_half_goal, 0.0, 1.0))

    distance = math.hypot(*to_ball)
    controllable_proximity = math.exp(-distance / 0.25)
    field_length = float(config["field"]["length"])
    attacking_progress = float(
        np.clip((attack_sign * ball[0] + field_length / 2.0) / field_length, 0.0, 1.0)
    )
    potential = float(
        np.clip(
            0.45 * alignment
            + 0.25 * aperture
            + 0.15 * controllable_proximity
            + 0.15 * attacking_progress,
            0.0,
            1.0,
        )
    )
    return {
        "potential": potential,
        "attacker_alignment": alignment,
        "goal_aperture": aperture,
        "controllable_proximity": controllable_proximity,
        "attacking_progress": attacking_progress,
    }


def _goal_geometry_potential(
    state: FloatArray,
    config: dict[str, Any],
    team: int = 0,
) -> float:
    """Bounded state potential; reward its discounted change, never the static pose."""
    return _goal_geometry_metrics(state, config, team)["potential"]


def _idle_spin_flags(
    state: FloatArray,
    team: int,
    normalized_actions: FloatArray,
    *,
    angular_speed_threshold: float,
    drive_threshold: float,
    speed_threshold: float,
    ball_distance: float,
) -> tuple[NDArray[np.bool_], FloatArray]:
    """Find robots rotating in place, slow, remote from the ball, and not asking to drive.

    Rotation is judged on measured angular speed rather than on the wheel differential,
    because the differential a policy can request depends on the action parser. A
    geometric controller spends at most a small fraction of the wheel limit on turning,
    so a command-space threshold calibrated for direct wheel control either cannot fire
    at all or, once rescaled by that fraction, degenerates into "is the robot aiming a
    few degrees off". Angular speed carries the same meaning for every parser.
    """
    left = normalized_actions[:, 0]
    right = normalized_actions[:, 1]
    drive_intensity = np.abs(right + left) / 2.0
    flags = np.zeros(3, dtype=np.bool_)
    angular_speed = np.zeros(3, dtype=np.float32)
    offset = 0 if team == 0 else 3
    ball_x, ball_y = float(state[5]), float(state[6])
    for local_slot in range(3):
        base = ROBOT_BASE + (offset + local_slot) * ROBOT_WIDTH
        if not bool(float(state[base + 10])):
            continue
        angular_speed[local_slot] = abs(float(state[base + 7]))
        speed = math.hypot(float(state[base + 5]), float(state[base + 6]))
        distance = math.hypot(ball_x - float(state[base + 2]), ball_y - float(state[base + 3]))
        flags[local_slot] = (
            angular_speed[local_slot] > angular_speed_threshold
            and drive_intensity[local_slot] < drive_threshold
            and speed < speed_threshold
            and distance > ball_distance
        )
    # Proportional above the threshold and saturating at twice it, so the configured
    # coefficient keeps a bounded per-decision meaning.
    reference = max(2.0 * angular_speed_threshold, 1e-6)
    return flags, np.clip(angular_speed / reference, 0.0, 1.0).astype(np.float32)


def _attacker_alignment_reward(
    state: FloatArray,
    speed_threshold: float,
    team: int = 0,
) -> float:
    ball = (float(state[5]), float(state[6]))
    offset = 0 if team == 0 else 3
    attacker = min(
        (
            slot
            for slot in range(offset, offset + 3)
            if bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10]))
        ),
        key=lambda slot: math.hypot(
            ball[0] - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            ball[1] - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        ),
    )
    base = ROBOT_BASE + attacker * ROBOT_WIDTH
    position = (float(state[base + 2]), float(state[base + 3]))
    velocity = (float(state[base + 5]), float(state[base + 6]))
    if math.hypot(*velocity) <= speed_threshold:
        return -2.0 * math.tanh(1.0)
    similarity = _cosine_similarity(
        (ball[0] - position[0], ball[1] - position[1]),
        velocity,
    )
    return math.tanh(similarity) - math.tanh(1.0)


def _defensive_distance(
    state: FloatArray,
    config: dict[str, Any],
    team: int = 0,
) -> float:
    field = config["field"]
    attack_sign = 1.0 if team == 0 else -1.0
    target_x = -attack_sign * (float(field["length"]) / 2.0 - 0.12)
    half_goal = float(field["goal_width"]) / 2.0
    target_y = float(np.clip(state[6], -half_goal, half_goal))
    return min(
        math.hypot(
            target_x - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            target_y - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        )
        for slot in range(0 if team == 0 else 3, 3 if team == 0 else 6)
        if bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10]))
    )


def _defensive_threat(ball_x: float, activation_x: float, team: int = 0) -> float:
    attack_sign = 1.0 if team == 0 else -1.0
    return float(np.clip((activation_x - attack_sign * ball_x) / 0.75, 0.0, 1.0))


def distill_dynamic_teacher(
    actor: SharedActor
    | RoleSharedActor
    | PrimitiveRoleActor
    | ParametricPrimitiveRoleActor
    | CircularPrimitiveRoleActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor,
    config_json: str,
    state_json: str,
    *,
    seed: int,
    samples: int = 2_048,
    epochs: int = 20,
) -> float:
    """Initialize a shared actor from M4 dynamic assignments over seeded resets."""
    seed_everything(seed)

    def reset_module(module: torch.nn.Module) -> None:
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()

    actor.apply(reset_module)
    env = MarlMatchEnv(config_json, state_json, stage=8, horizon=1)
    teacher = DynamicTeamController(0, 1)
    observations: list[TeamBatch] = []
    actions: list[torch.Tensor] = []
    primitive_indices: list[torch.Tensor] = []
    parametric_skills: list[torch.Tensor] = []
    parametric_parameters: list[torch.Tensor] = []
    circular_headings: list[torch.Tensor] = []
    for sample_seed in range(seed, seed + samples):
        observations.append(env.reset(sample_seed))
        actions.append(torch.from_numpy(teacher.actions(env.state).copy()))
        roles = teacher.assign(env.state)
        labels: list[int] = []
        skill_labels: list[int] = []
        heading_labels: list[float] = []
        parameter_labels: list[tuple[float, float, float]] = []
        ball = (float(env.state[5]), float(env.state[6]))
        for local_slot, role in enumerate(roles):
            pose = robot_pose(env.state, local_slot)
            if role == "pressor":
                vector = (0.75 - ball[0], -ball[1])
                direction = nearest_canonical_direction(vector, 0)
                labels.append(1 + SoccerPrimitiveSet.directions + direction)
                skill_labels.append(2)
            else:
                target = (
                    (-0.68, float(np.clip(ball[1], -0.18, 0.18)))
                    if role == "goalie"
                    else (ball[0] - 0.28, -0.5 * ball[1])
                )
                direction = nearest_canonical_direction(
                    (target[0] - pose[0], target[1] - pose[1]),
                    0,
                )
                labels.append(1 + direction)
                vector = (target[0] - pose[0], target[1] - pose[1])
                skill_labels.append(1)
            angle = math.atan2(vector[1], vector[0])
            # The deterministic teacher already modulates its normalized wheel
            # command from geometry. Distill full primitive authority here;
            # multiplying by the teacher wheel magnitude a second time makes
            # the learned controller artificially slow.
            # A target of exactly full authority is unreachable through tanh and drags
            # the parameter mean into saturation, which pins intensity for the whole run.
            parameter_labels.append((math.cos(angle), math.sin(angle), TEACHER_AUTHORITY))
            heading_labels.append(angle)
        primitive_indices.append(torch.tensor(labels, dtype=torch.int64))
        parametric_skills.append(torch.tensor(skill_labels, dtype=torch.int64))
        parametric_parameters.append(torch.tensor(parameter_labels, dtype=torch.float32))
        circular_headings.append(torch.tensor(heading_labels, dtype=torch.float32))
    batch = stack_team_batches(observations)
    device = next(actor.parameters()).device
    batch = batch.to(device)
    targets = torch.stack(actions).to(device)
    primitive_targets = torch.stack(primitive_indices).to(device)
    parametric_skill_targets = torch.stack(parametric_skills).to(device)
    parametric_parameter_targets = torch.stack(parametric_parameters).to(device)
    circular_heading_targets = torch.stack(circular_headings).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    loss = torch.zeros(())
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(samples, generator=generator).split(256):  # type: ignore[no-untyped-call]
            indices = indices.to(device)
            selected = batch.select_batch(indices)
            if isinstance(actor, CircularPrimitiveRoleActor):
                skill_logits, heading, _, intensity_mean, _ = actor(selected)
                skill_loss = torch.nn.functional.cross_entropy(
                    skill_logits.reshape(-1, skill_logits.shape[-1]),
                    parametric_skill_targets[indices].reshape(-1),
                )
                # A circular residual, so the loss does not jump across the wrap.
                heading_loss = (1.0 - torch.cos(heading - circular_heading_targets[indices])).mean()
                intensity_loss = (
                    (
                        torch.tanh(intensity_mean).squeeze(-1)
                        - parametric_parameter_targets[indices][..., 2]
                    )
                    .square()
                    .mean()
                )
                loss = skill_loss + heading_loss + intensity_loss
            elif isinstance(actor, ParametricPrimitiveRoleActor):
                skill_logits, parameter_mean, _ = actor(selected)
                skill_loss = torch.nn.functional.cross_entropy(
                    skill_logits.reshape(-1, skill_logits.shape[-1]),
                    parametric_skill_targets[indices].reshape(-1),
                )
                parameter_loss = (
                    (torch.tanh(parameter_mean) - parametric_parameter_targets[indices])
                    .square()
                    .mean()
                )
                loss = skill_loss + parameter_loss
            else:
                mean, _ = actor(selected)
            if isinstance(actor, PrimitiveRoleActor):
                loss = torch.nn.functional.cross_entropy(
                    mean.reshape(-1, mean.shape[-1]),
                    primitive_targets[indices].reshape(-1),
                )
            elif isinstance(actor, LatticeSharedActor):
                lattice = torch.tensor(
                    SymmetricWheelLattice.values,
                    dtype=torch.float32,
                    device=device,
                )
                distances = (targets[indices].unsqueeze(-2) - lattice).square().sum(dim=-1)
                lattice_labels = distances.argmin(dim=-1)
                loss = torch.nn.functional.cross_entropy(
                    mean.reshape(-1, mean.shape[-1]),
                    lattice_labels.reshape(-1),
                )
            elif not isinstance(actor, (ParametricPrimitiveRoleActor, CircularPrimitiveRoleActor)):
                loss = (torch.tanh(mean) - targets[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    return float(loss.detach())


def evaluate_against_random(
    actor: SharedActor
    | RoleSharedActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor,
    config_json: str,
    state_json: str,
    *,
    stage: int,
    seeds: range,
    horizon: int,
    action_repeat: int = 4,
    required_margin: float = 0.05,
    action_parser: str = "continuous",
) -> MarlEvaluation:
    policy_scores: list[float] = []
    random_scores: list[float] = []
    actor.eval()
    device = next(actor.parameters()).device
    action_width = team_action_width(action_parser)
    for seed in seeds:
        policy_env = MarlMatchEnv(
            config_json,
            state_json,
            stage=stage,
            horizon=horizon,
            action_repeat=action_repeat,
            action_parser=action_parser,
        )
        observation = policy_env.reset(seed)
        policy_env.mark_progress_origin()
        done = False
        while not done:
            with torch.no_grad():
                action = actor.deterministic_action(observation.to(device)).cpu().numpy()
            observation, _, done, _ = policy_env.step(action)
        policy_scores.append(policy_env.progress_score())

        random_env = MarlMatchEnv(
            config_json,
            state_json,
            stage=stage,
            horizon=horizon,
            action_repeat=action_repeat,
            action_parser=action_parser,
        )
        random_env.reset(seed)
        random_env.mark_progress_origin()
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            _, _, done, _ = random_env.step(
                rng.uniform(-1.0, 1.0, (3, action_width)).astype(np.float32)
            )
        random_scores.append(random_env.progress_score())
    policy_progress = float(np.mean(policy_scores))
    random_progress = float(np.mean(random_scores))
    margin = policy_progress - random_progress
    return MarlEvaluation(
        seeds=len(seeds),
        policy_progress=policy_progress,
        random_progress=random_progress,
        margin=margin,
        passed=margin >= required_margin,
    )
