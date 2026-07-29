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
from vsss_env._native import BatchSimulator

from vsss_train.ablations import (
    EntityAttentionActor,
    LatticeSharedActor,
    RecurrentSharedActor,
    SymmetricWheelLattice,
)
from vsss_train.marl import (
    RoleSharedActor,
    SharedActor,
    TeamBatch,
    build_team_observation,
    stack_team_batches,
)
from vsss_train.ppo import seed_everything
from vsss_train.roles import DynamicRoleAssigner, RoleAssignment

FloatArray = NDArray[np.float32]
ROBOT_BASE = 10
ROBOT_WIDTH = 11


@dataclass(frozen=True)
class TeamReward:
    ball_progress: float
    ball_direction: float
    attacker_alignment: float
    time: float
    goal: float
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
        self.attacker_alignment_coefficient = attacker_alignment_coefficient
        self.time_penalty_coefficient = time_penalty_coefficient
        self.movement_speed_threshold = movement_speed_threshold
        self.teammate_spacing = teammate_spacing
        self.teammate_congestion_coefficient = teammate_congestion_coefficient
        self.defensive_coverage_coefficient = defensive_coverage_coefficient
        self.defensive_activation_x = defensive_activation_x
        self.draw_penalty = draw_penalty
        self.stagnation_penalty = stagnation_penalty
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
        return build_team_observation(self.state, team=0, role_assignment=self._role_assignment)

    def step(
        self,
        blue_actions: FloatArray,
        opponent_actions: FloatArray | None = None,
    ) -> tuple[TeamBatch, TeamReward, bool, dict[str, Any]]:
        normalized_blue = np.clip(blue_actions, -1.0, 1.0)
        action_delta = normalized_blue - self._previous_blue_actions
        actions = np.zeros((1, 6, 2), dtype=np.float32)
        actions[0, :3] = normalized_blue * self._max_wheel_speed
        for slot in range(3):
            if not bool(float(self.state[ROBOT_BASE + slot * ROBOT_WIDTH + 10])):
                actions[0, slot] = 0.0
        events = 0
        for _ in range(self.action_repeat):
            if opponent_actions is not None:
                actions[0, 3:] = np.clip(opponent_actions, -1.0, 1.0) * self._max_wheel_speed
            elif self.stage == 8:
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
            },
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the current canonical lifecycle snapshot."""
        return cast(dict[str, Any], json.loads(self._native.snapshots()[0]))

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
        self.attacker_alignment_coefficient = attacker_alignment_coefficient
        self.time_penalty_coefficient = time_penalty_coefficient
        self.movement_speed_threshold = movement_speed_threshold
        self.teammate_spacing = teammate_spacing
        self.teammate_congestion_coefficient = teammate_congestion_coefficient
        self._decision_period = float(self._config["timestep"]) * action_repeat
        self.contact_distance = contact_distance
        self.contact_grace_steps = max(1, round(contact_grace_seconds / self._decision_period))
        self.ally_deadlock_coefficient = ally_deadlock_coefficient
        self.opponent_deadlock_coefficient = opponent_deadlock_coefficient
        self.defensive_coverage_coefficient = defensive_coverage_coefficient
        self.defensive_activation_x = defensive_activation_x
        self.draw_penalty = draw_penalty
        self.stagnation_penalty = stagnation_penalty
        self.stagnation_limit = max(1, round(stagnation_seconds / self._decision_period))
        self.stagnation_ball_distance = stagnation_ball_distance
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
        self._goal_grace_remaining = np.full(num_envs, -1, dtype=np.int64)
        self._defensive_distance = np.zeros(num_envs, dtype=np.float32)
        self._stagnation_anchor = np.zeros((num_envs, 2), dtype=np.float32)
        self._stagnation_steps = np.zeros(num_envs, dtype=np.int64)
        self._previous_ball_positions = np.zeros((num_envs, 2), dtype=np.float32)
        self._ally_contact_streaks = np.zeros((num_envs, 3), dtype=np.int64)
        self._opponent_contact_streaks = np.zeros((num_envs, 9), dtype=np.int64)
        self.ally_contact_steps = np.zeros(num_envs, dtype=np.int64)
        self.opponent_contact_steps = np.zeros(num_envs, dtype=np.int64)
        self.ally_deadlocks = np.zeros(num_envs, dtype=np.int64)
        self.opponent_deadlocks = np.zeros(num_envs, dtype=np.int64)
        self.contact_escapes = np.zeros(num_envs, dtype=np.int64)
        self.last_terminal_reasons = np.full(num_envs, "", dtype="<U24")
        self.controlled_teams = np.zeros(num_envs, dtype=np.int64)
        self._role_assigners = [DynamicRoleAssigner() for _ in range(num_envs)]
        self.role_assignments: list[RoleAssignment | None] = [None] * num_envs
        self.role_switches = np.zeros(num_envs, dtype=np.int64)
        self.uncovered_steps = np.zeros(num_envs, dtype=np.int64)
        self.role_decisions = np.zeros(num_envs, dtype=np.int64)

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
        self._defensive_distance[world] = _defensive_distance(
            self.states[world], self._config, team
        )
        self._stagnation_anchor[world] = self.states[world, 5:7]
        self._previous_ball_positions[world] = self.states[world, 5:7]
        self._stagnation_steps[world] = 0
        self._ally_contact_streaks[world].fill(0)
        self._opponent_contact_streaks[world].fill(0)
        self.last_terminal_reasons[world] = ""
        self._role_assigners[world].reset()
        assignment = self._role_assigners[world].assign(self.states[world], team)
        self.role_assignments[world] = assignment
        return build_team_observation(self.states[world], team=team, role_assignment=assignment)

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
        normalized_blue = np.clip(blue_actions, -1.0, 1.0, out=self._normalized_blue_actions)
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
                actions[world, opponent_slice] = normalized_opponents[world] * self._max_wheel_speed
            self.states = self._native.step_repeated(actions, self.action_repeat)
            events |= self.states[:, -1].astype(np.int64)
        elif self.stage == 8:
            for _ in range(self.action_repeat):
                for world in range(self.num_envs):
                    if self.controlled_teams[world] == 0:
                        actions[world, 3:] = (
                            self._yellow.actions(self.states[world]) * self._max_wheel_speed
                        )
                    else:
                        actions[world, :3] = (
                            self._blue.actions(self.states[world]) * self._max_wheel_speed
                        )
                self.states = self._native.step(actions)
                events |= self.states[:, -1].astype(np.int64)
        else:
            self.states = self._native.step_repeated(actions, self.action_repeat)
            events |= self.states[:, -1].astype(np.int64)
        self.steps += 1
        ball_x = self.states[:, 5]
        closest = np.asarray(
            [
                _closest_team_distance(state, int(team))
                for state, team in zip(self.states, self.controlled_teams, strict=True)
            ],
            dtype=np.float32,
        )
        defensive_distance = np.asarray(
            [
                _defensive_distance(state, self._config, int(team))
                for state, team in zip(self.states, self.controlled_teams, strict=True)
            ],
            dtype=np.float32,
        )
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
        congestion = np.asarray(
            [
                _teammate_congestion(state, self.teammate_spacing, int(team))
                for state, team in zip(self.states, self.controlled_teams, strict=True)
            ],
            dtype=np.float32,
        )
        contact_metrics = [
            _contact_deadlock_metrics(
                state,
                int(team),
                contact_distance=self.contact_distance,
                grace_steps=self.contact_grace_steps,
                ally_streaks=self._ally_contact_streaks[world],
                opponent_streaks=self._opponent_contact_streaks[world],
                previous_ball=self._previous_ball_positions[world],
                meaningful_ball_displacement=self.movement_speed_threshold * self._decision_period,
                config=self._config,
            )
            for world, (state, team) in enumerate(
                zip(self.states, self.controlled_teams, strict=True)
            )
        ]
        ally_deadlock_penalty = np.asarray(
            [metric.ally_penalty for metric in contact_metrics], dtype=np.float32
        )
        opponent_deadlock_penalty = np.asarray(
            [metric.opponent_penalty for metric in contact_metrics], dtype=np.float32
        )
        for world, metric in enumerate(contact_metrics):
            self._ally_contact_streaks[world] = metric.ally_streaks
            self._opponent_contact_streaks[world] = metric.opponent_streaks
            self.ally_contact_steps[world] += metric.ally_contacts
            self.opponent_contact_steps[world] += metric.opponent_contacts
            self.ally_deadlocks[world] += metric.ally_deadlocks
            self.opponent_deadlocks[world] += metric.opponent_deadlocks
            self.contact_escapes[world] += metric.escapes
        newly_scored = ((events & 0b11) != 0) & (self._goal_grace_remaining < 0)
        self._goal_grace_remaining[newly_scored] = round(
            float(self._config["reset"]["goal_pause"]) / self._decision_period
        )
        active_grace = self._goal_grace_remaining >= 0
        self._goal_grace_remaining[active_grace] -= 1
        goal_complete = active_grace & (self._goal_grace_remaining <= 0)
        stagnated = (self._goal_grace_remaining < 0) & (
            self._stagnation_steps >= self.stagnation_limit
        )
        draw = (self.steps >= self.horizon) & ~goal_complete & ~stagnated
        attack_sign = np.where(self.controlled_teams == 0, 1.0, -1.0)
        scored = np.where(self.controlled_teams == 0, (events & 1) != 0, (events & 2) != 0)
        conceded = np.where(
            self.controlled_teams == 0,
            (events & 2) != 0,
            (events & 1) != 0,
        )
        rewards = (
            self.progress_coefficient
            * (2.0 * (self._closest - closest) + attack_sign * (ball_x - self._ball_x))
            + self.ball_direction_coefficient
            * np.asarray(
                [
                    _ball_direction_reward(
                        state,
                        self._config,
                        self.movement_speed_threshold,
                        int(team),
                    )
                    for state, team in zip(self.states, self.controlled_teams, strict=True)
                ],
                dtype=np.float32,
            )
            / self.horizon
            + self.attacker_alignment_coefficient
            * np.asarray(
                [
                    _attacker_alignment_reward(
                        state,
                        self.movement_speed_threshold,
                        int(team),
                    )
                    for state, team in zip(self.states, self.controlled_teams, strict=True)
                ],
                dtype=np.float32,
            )
            / self.horizon
            - self.time_penalty_coefficient / self.horizon
            + self.goal_coefficient * scored
            - self.goal_coefficient * conceded
            - self.action_delta_coefficient * np.square(action_delta).mean(axis=(1, 2))
            - self.wheel_effort_coefficient * np.square(normalized_blue).mean(axis=(1, 2))
            - self.teammate_congestion_coefficient * congestion
            - self.ally_deadlock_coefficient * ally_deadlock_penalty
            - self.opponent_deadlock_coefficient * opponent_deadlock_penalty
            + self.defensive_coverage_coefficient
            * threat
            * (self._defensive_distance - defensive_distance)
            - self.draw_penalty * draw
            - self.stagnation_penalty * stagnated
        ).astype(np.float32)
        np.copyto(self._previous_blue_actions, normalized_blue)
        np.copyto(self._ball_x, ball_x)
        np.copyto(self._previous_ball_positions, ball_positions)
        self._closest = closest
        self._defensive_distance = defensive_distance
        done = draw | goal_complete | stagnated
        self.last_terminal_reasons.fill("")
        self.last_terminal_reasons[draw] = "draw"
        self.last_terminal_reasons[stagnated] = "stagnation"
        self.last_terminal_reasons[goal_complete] = "goal"
        assignments: list[RoleAssignment | None] = [
            assigner.assign(state, int(team))
            for assigner, state, team in zip(
                self._role_assigners, self.states, self.controlled_teams, strict=True
            )
        ]
        self.role_assignments = assignments
        self.role_switches += np.asarray(
            [sum(assignment.changed) for assignment in assignments if assignment is not None],
            dtype=np.int64,
        )
        self.uncovered_steps += np.asarray(
            [assignment.uncovered for assignment in assignments if assignment is not None],
            dtype=np.int64,
        )
        self.role_decisions += 1
        observations = stack_team_batches(
            [
                build_team_observation(state, team=int(team), role_assignment=assignment)
                for state, team, assignment in zip(
                    self.states, self.controlled_teams, assignments, strict=True
                )
            ]
        )
        # Episode completion and value-bootstrap termination have different
        # semantics for timeouts. Never alias them: the collector may mark a
        # skill timeout as truncated without cancelling the required reset.
        return observations, rewards, done, events, done.copy()

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
    for sample_seed in range(seed, seed + samples):
        observations.append(env.reset(sample_seed))
        actions.append(torch.from_numpy(teacher.actions(env.state).copy()))
    batch = stack_team_batches(observations)
    device = next(actor.parameters()).device
    batch = batch.to(device)
    targets = torch.stack(actions).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    loss = torch.zeros(())
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(samples, generator=generator).split(256):  # type: ignore[no-untyped-call]
            indices = indices.to(device)
            mean, _ = actor(batch.select_batch(indices))
            if isinstance(actor, LatticeSharedActor):
                lattice = torch.tensor(
                    SymmetricWheelLattice.values,
                    dtype=torch.float32,
                    device=device,
                )
                distances = (targets[indices].unsqueeze(-2) - lattice).square().sum(dim=-1)
                labels = distances.argmin(dim=-1)
                loss = torch.nn.functional.cross_entropy(
                    mean.reshape(-1, mean.shape[-1]),
                    labels.reshape(-1),
                )
            else:
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
) -> MarlEvaluation:
    policy_scores: list[float] = []
    random_scores: list[float] = []
    actor.eval()
    device = next(actor.parameters()).device
    for seed in seeds:
        policy_env = MarlMatchEnv(
            config_json, state_json, stage=stage, horizon=horizon, action_repeat=action_repeat
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
            config_json, state_json, stage=stage, horizon=horizon, action_repeat=action_repeat
        )
        random_env.reset(seed)
        random_env.mark_progress_origin()
        rng = np.random.default_rng(seed)
        done = False
        while not done:
            _, _, done, _ = random_env.step(rng.uniform(-1.0, 1.0, (3, 2)).astype(np.float32))
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
