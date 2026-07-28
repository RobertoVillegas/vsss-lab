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
    approach_progress: float
    goal: float
    action_delta: float = 0.0

    @property
    def total(self) -> float:
        return self.ball_progress + self.approach_progress + self.goal + self.action_delta


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
        self.steps = 0
        self.state = np.zeros(BatchSimulator.state_width(), dtype=np.float32)
        self._ball_x = 0.0
        self._closest = 0.0
        self._initial_ball_x = 0.0
        self._initial_closest = 0.0
        self._previous_blue_actions = np.zeros((3, 2), dtype=np.float32)

    def reset(self, seed: int) -> TeamBatch:
        rng = np.random.default_rng(seed)
        snapshot = copy.deepcopy(self._template)
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
        self._native.restore(0, json.dumps(snapshot, separators=(",", ":")))
        self.state = self._native.step(np.zeros((1, 6, 2), dtype=np.float32))[0]
        self.steps = 0
        self._ball_x = float(self.state[5])
        self._closest = self._closest_blue_distance()
        self._previous_blue_actions.fill(0.0)
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
        for _ in range(self.action_repeat):
            if opponent_actions is not None:
                actions[0, 3:] = np.clip(opponent_actions, -1.0, 1.0) * self._max_wheel_speed
            elif self.stage == 8:
                actions[0, 3:] = self._yellow.actions(self.state) * self._max_wheel_speed
            self.state = self._native.step(actions)[0]
        self.steps += 1
        ball_x = float(self.state[5])
        closest = self._closest_blue_distance()
        events = int(self.state[-1])
        reward = TeamReward(
            ball_progress=4.0 * (ball_x - self._ball_x),
            approach_progress=2.0 * (self._closest - closest),
            goal=10.0 * float(bool(events & 1)) - 10.0 * float(bool(events & 2)),
            action_delta=-self.action_delta_coefficient * float(np.square(action_delta).mean()),
        )
        self._previous_blue_actions = normalized_blue.copy()
        self._ball_x = ball_x
        self._closest = closest
        done = self.steps >= self.horizon or bool(events & 0b11)
        return (
            build_team_observation(self.state, team=0),
            reward,
            done,
            {
                "events": events,
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


def distill_dynamic_teacher(
    actor: SharedActor,
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
    actor: SharedActor,
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
