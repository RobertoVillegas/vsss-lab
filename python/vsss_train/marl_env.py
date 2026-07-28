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

from vsss_train.ablations import EntityAttentionActor
from vsss_train.marl import (
    SharedActor,
    TeamBatch,
    build_team_observation,
    stack_team_batches,
)
from vsss_train.ppo import seed_everything

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

    def reset(self, seed: int) -> TeamBatch:
        snapshot = _seeded_snapshot(self._template, seed)
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
        return build_team_observation(self.state, team=0)

    def step(
        self,
        blue_actions: FloatArray,
        opponent_actions: FloatArray | None = None,
    ) -> tuple[TeamBatch, TeamReward, bool, dict[str, Any]]:
        normalized_blue = np.clip(blue_actions, -1.0, 1.0)
        action_delta = normalized_blue - self._previous_blue_actions
        actions = np.zeros((1, 6, 2), dtype=np.float32)
        actions[0, :3] = normalized_blue * self._max_wheel_speed
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
            ball_progress=0.0,
            ball_direction=self.ball_direction_coefficient
            * _ball_direction_reward(self.state, self._config, self.movement_speed_threshold)
            / self.horizon,
            attacker_alignment=self.attacker_alignment_coefficient
            * _attacker_alignment_reward(self.state, self.movement_speed_threshold)
            / self.horizon,
            time=-self.time_penalty_coefficient / self.horizon,
            goal=(
                10.0 * float(bool(events & 1))
                - 10.0 * float(bool(events & 2))
                - self.draw_penalty * float(draw)
                - self.stagnation_penalty * float(stagnated)
            ),
            action_delta=-self.action_delta_coefficient * float(np.square(action_delta).mean()),
            wheel_effort=-self.wheel_effort_coefficient * float(np.square(normalized_blue).mean()),
            teammate_congestion=-self.teammate_congestion_coefficient
            * _teammate_congestion(self.state, self.teammate_spacing),
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
        return (
            build_team_observation(self.state, team=0),
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
        wheel_effort_coefficient: float,
        ball_direction_coefficient: float,
        attacker_alignment_coefficient: float,
        time_penalty_coefficient: float,
        movement_speed_threshold: float,
        teammate_spacing: float,
        teammate_congestion_coefficient: float,
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
        self._yellow = DynamicTeamController(3, -1)
        self._max_wheel_speed = float(self._config["max_wheel_speed"])
        self.num_envs = num_envs
        self.stage = stage
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.action_delta_coefficient = action_delta_coefficient
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
        self.states = np.zeros((num_envs, BatchSimulator.state_width()), dtype=np.float32)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self._ball_x = np.zeros(num_envs, dtype=np.float32)
        self._closest = np.zeros(num_envs, dtype=np.float32)
        self._initial_ball_x = np.zeros(num_envs, dtype=np.float32)
        self._initial_closest = np.zeros(num_envs, dtype=np.float32)
        self._previous_blue_actions = np.zeros((num_envs, 3, 2), dtype=np.float32)
        self._goal_grace_remaining = np.full(num_envs, -1, dtype=np.int64)
        self._defensive_distance = np.zeros(num_envs, dtype=np.float32)
        self._stagnation_anchor = np.zeros((num_envs, 2), dtype=np.float32)
        self._stagnation_steps = np.zeros(num_envs, dtype=np.int64)
        self.last_terminal_reasons = np.full(num_envs, "", dtype="<U10")

    def reset(self, world: int, seed: int) -> TeamBatch:
        snapshot = _seeded_snapshot(self._template, seed)
        self.states[world] = self._native.restore_state(
            world, json.dumps(snapshot, separators=(",", ":"))
        )
        self.steps[world] = 0
        self._ball_x[world] = self.states[world, 5]
        self._closest[world] = _closest_blue_distance(self.states[world])
        self._initial_ball_x[world] = self._ball_x[world]
        self._initial_closest[world] = self._closest[world]
        self._previous_blue_actions[world].fill(0.0)
        self._goal_grace_remaining[world] = -1
        self._defensive_distance[world] = _defensive_distance(self.states[world], self._config)
        self._stagnation_anchor[world] = self.states[world, 5:7]
        self._stagnation_steps[world] = 0
        self.last_terminal_reasons[world] = ""
        return build_team_observation(self.states[world], team=0)

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
        normalized_blue = np.clip(blue_actions, -1.0, 1.0)
        action_delta = normalized_blue - self._previous_blue_actions
        actions = np.zeros((self.num_envs, 6, 2), dtype=np.float32)
        actions[:, :3] = normalized_blue * self._max_wheel_speed
        events = np.zeros(self.num_envs, dtype=np.int64)
        if opponent_actions is not None:
            actions[:, 3:] = np.clip(opponent_actions, -1.0, 1.0) * self._max_wheel_speed
            self.states = self._native.step_repeated(actions, self.action_repeat)
            events |= self.states[:, -1].astype(np.int64)
        elif self.stage == 8:
            for _ in range(self.action_repeat):
                for world in range(self.num_envs):
                    actions[world, 3:] = (
                        self._yellow.actions(self.states[world]) * self._max_wheel_speed
                    )
                self.states = self._native.step(actions)
                events |= self.states[:, -1].astype(np.int64)
        else:
            self.states = self._native.step_repeated(actions, self.action_repeat)
            events |= self.states[:, -1].astype(np.int64)
        self.steps += 1
        ball_x = self.states[:, 5]
        closest = np.asarray(
            [_closest_blue_distance(state) for state in self.states],
            dtype=np.float32,
        )
        defensive_distance = np.asarray(
            [_defensive_distance(state, self._config) for state in self.states],
            dtype=np.float32,
        )
        threat = np.asarray(
            [_defensive_threat(float(x), self.defensive_activation_x) for x in ball_x],
            dtype=np.float32,
        )
        ball_positions = self.states[:, 5:7]
        ball_displacement = np.linalg.norm(ball_positions - self._stagnation_anchor, axis=1)
        moved = ball_displacement >= self.stagnation_ball_distance
        self._stagnation_anchor[moved] = ball_positions[moved]
        self._stagnation_steps[moved] = 0
        self._stagnation_steps[~moved] += 1
        congestion = np.asarray(
            [_teammate_congestion(state, self.teammate_spacing) for state in self.states],
            dtype=np.float32,
        )
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
        rewards = (
            self.ball_direction_coefficient
            * np.asarray(
                [
                    _ball_direction_reward(state, self._config, self.movement_speed_threshold)
                    for state in self.states
                ],
                dtype=np.float32,
            )
            / self.horizon
            + self.attacker_alignment_coefficient
            * np.asarray(
                [
                    _attacker_alignment_reward(state, self.movement_speed_threshold)
                    for state in self.states
                ],
                dtype=np.float32,
            )
            / self.horizon
            - self.time_penalty_coefficient / self.horizon
            + 10.0 * ((events & 1) != 0)
            - 10.0 * ((events & 2) != 0)
            - self.action_delta_coefficient * np.square(action_delta).mean(axis=(1, 2))
            - self.wheel_effort_coefficient * np.square(normalized_blue).mean(axis=(1, 2))
            - self.teammate_congestion_coefficient * congestion
            + self.defensive_coverage_coefficient
            * threat
            * (self._defensive_distance - defensive_distance)
            - self.draw_penalty * draw
            - self.stagnation_penalty * stagnated
        ).astype(np.float32)
        self._previous_blue_actions = normalized_blue.copy()
        self._ball_x = ball_x.copy()
        self._closest = closest
        self._defensive_distance = defensive_distance
        done = draw | goal_complete | stagnated
        self.last_terminal_reasons.fill("")
        self.last_terminal_reasons[draw] = "draw"
        self.last_terminal_reasons[stagnated] = "stagnation"
        self.last_terminal_reasons[goal_complete] = "goal"
        observations = stack_team_batches(
            [build_team_observation(state, team=0) for state in self.states]
        )
        return observations, rewards, done, events, done

    def mark_progress_origin(self) -> None:
        self._initial_closest = self._closest.copy()
        self._initial_ball_x = self._ball_x.copy()

    def progress_scores(self) -> FloatArray:
        return (
            2.0 * (self._initial_closest - self._closest) + (self._ball_x - self._initial_ball_x)
        ).astype(np.float32)

    def snapshot(self, world: int) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._native.snapshots()[world]))


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


def _closest_blue_distance(state: FloatArray) -> float:
    return min(
        math.hypot(
            float(state[5] - state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            float(state[6] - state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        )
        for slot in range(3)
    )


def _teammate_congestion(state: FloatArray, spacing: float) -> float:
    positions = [
        (
            float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        )
        for slot in range(3)
    ]
    penalties = [
        max(0.0, (spacing - math.dist(positions[first], positions[second])) / spacing) ** 2
        for first, second in ((0, 1), (0, 2), (1, 2))
    ]
    return sum(penalties) / len(penalties)


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
) -> float:
    velocity = (float(state[7]), float(state[8]))
    if math.hypot(*velocity) < speed_threshold:
        return 0.0
    ball = (float(state[5]), float(state[6]))
    goal_x = float(config["field"]["length"]) / 2.0
    enemy = (goal_x - ball[0], -ball[1])
    ally = (-goal_x - ball[0], -ball[1])
    enemy_similarity = _cosine_similarity(enemy, velocity)
    ally_similarity = _cosine_similarity(ally, velocity)
    return math.tanh(enemy_similarity) - math.tanh(ally_similarity)


def _attacker_alignment_reward(state: FloatArray, speed_threshold: float) -> float:
    ball = (float(state[5]), float(state[6]))
    attacker = min(
        range(3),
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


def _defensive_distance(state: FloatArray, config: dict[str, Any]) -> float:
    field = config["field"]
    target_x = -float(field["length"]) / 2.0 + 0.12
    half_goal = float(field["goal_width"]) / 2.0
    target_y = float(np.clip(state[6], -half_goal, half_goal))
    return min(
        math.hypot(
            target_x - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 2]),
            target_y - float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 3]),
        )
        for slot in range(3)
    )


def _defensive_threat(ball_x: float, activation_x: float) -> float:
    return float(np.clip((activation_x - ball_x) / 0.75, 0.0, 1.0))


def distill_dynamic_teacher(
    actor: SharedActor | EntityAttentionActor,
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
            loss = (torch.tanh(mean) - targets[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    return float(loss.detach())


def evaluate_against_random(
    actor: SharedActor | EntityAttentionActor,
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
