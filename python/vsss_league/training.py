"""Native synchronous self-play rollout and optimization iterations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.distributions import Normal
from vsss_train.marl import SharedActor, TeamBatch, build_team_observation, stack_team_batches
from vsss_train.marl_env import MarlMatchEnv
from vsss_train.marl_ppo import (
    TRAJECTORY_SCHEMA,
    MarlLearner,
    TeamTrajectory,
    TrajectoryMetadata,
)


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    policy_version: int
    opponent: str
    seed: int
    frames: int
    return_total: float
    progress: float
    checkpoint: str
    losses: dict[str, float]


def collect_self_play_trajectory(
    learner: MarlLearner,
    opponent: SharedActor | None,
    config_json: str,
    state_json: str,
    *,
    seed: int,
    opponent_id: str,
) -> tuple[TeamTrajectory, float, float]:
    """Collect one fresh on-policy trajectory against a frozen opponent."""
    environment = MarlMatchEnv(
        config_json,
        state_json,
        stage=8,
        horizon=learner.config.horizon,
        action_repeat=learner.config.action_repeat,
    )
    observation = environment.reset(seed)
    environment.mark_progress_origin()
    initial_snapshot = environment.snapshot()
    observations: list[TeamBatch] = []
    actions: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminated: list[torch.Tensor] = []
    truncated: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    ticks: list[torch.Tensor] = []
    total_return = 0.0
    done = False
    while not done:
        with torch.no_grad():
            mean, log_std = learner.actor(observation)
            distribution = Normal(mean, log_std.exp())
            raw_action = distribution.sample()  # type: ignore[no-untyped-call]
            log_probability = distribution.log_prob(raw_action).sum(-1)  # type: ignore[no-untyped-call]
            value = learner.critic(observation)
            opponent_action = (
                opponent.deterministic_action(
                    build_team_observation(environment.state, team=1)
                ).numpy()
                if opponent is not None
                else None
            )
        observations.append(observation)
        actions.append(raw_action)
        log_probabilities.append(log_probability)
        values.append(value)
        ticks.append(torch.full((3,), int(environment.state[1]), dtype=torch.int64))
        observation, reward, done, info = environment.step(
            torch.tanh(raw_action).numpy(),
            opponent_action,
        )
        reward_tensor = torch.full((3,), reward.total, dtype=torch.float32)
        rewards.append(reward_tensor)
        total_return += reward.total
        is_terminated = bool(int(info["events"]) & 0b11)
        terminated.append(torch.full((3,), is_terminated, dtype=torch.bool))
        truncated.append(torch.full((3,), done and not is_terminated, dtype=torch.bool))
    batch = stack_team_batches(observations)
    data = TensorDict(
        {
            "tick": torch.stack(ticks),
            **dict(zip(TeamBatch._fields, batch, strict=True)),
            "action": torch.stack(actions),
            "sample_log_prob": torch.stack(log_probabilities),
            "reward_total": torch.stack(rewards),
            "terminated": torch.stack(terminated),
            "truncated": torch.stack(truncated),
            "state_value": torch.stack(values),
        },
        batch_size=[len(observations), 3],
    )
    state_reference = hashlib.sha256(
        json.dumps(initial_snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = TrajectoryMetadata(
        schema_version=TRAJECTORY_SCHEMA,
        run_id=f"self-play-{learner.config.seed}",
        episode_id=learner.policy_version,
        world_id=0,
        team=0,
        policy_id=learner.config.policy_id,
        policy_version=learner.policy_version,
        global_state_ref=f"sha256:{state_reference};opponent:{opponent_id}",
    )
    return TeamTrajectory(metadata, data), total_return, environment.progress_score()


def train_iteration(
    learner: MarlLearner,
    opponent: SharedActor | None,
    config_json: str,
    state_json: str,
    *,
    iteration: int,
    seed: int,
    opponent_id: str,
    checkpoint: Path,
) -> IterationResult:
    trajectory, total_return, progress = collect_self_play_trajectory(
        learner,
        opponent,
        config_json,
        state_json,
        seed=seed,
        opponent_id=opponent_id,
    )
    losses = learner.optimize(trajectory)
    learner.save(checkpoint)
    return IterationResult(
        iteration=iteration,
        policy_version=learner.policy_version,
        opponent=opponent_id,
        seed=seed,
        frames=len(trajectory.data),
        return_total=total_return,
        progress=progress,
        checkpoint=str(checkpoint.resolve()),
        losses=losses,
    )
