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
from vsss_train.marl_env import FloatArray, MarlMatchEnv
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
    checkpoint: str | None
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
    """Collect fixed-horizon vector self-play on the learner device."""
    if opponent is not None:
        opponent = opponent.to(learner.device)
    environments = [
        MarlMatchEnv(
            config_json,
            state_json,
            stage=8,
            horizon=learner.config.horizon,
            action_repeat=learner.config.action_repeat,
            action_delta_coefficient=learner.config.action_delta_coefficient,
        )
        for _ in range(learner.config.num_envs)
    ]
    observations_by_world = [
        environment.reset(seed + world) for world, environment in enumerate(environments)
    ]
    for environment in environments:
        environment.mark_progress_origin()
    observation = stack_team_batches(observations_by_world).to(learner.device)
    initial_snapshot = environments[0].snapshot()
    observations: list[TeamBatch] = []
    actions: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminated: list[torch.Tensor] = []
    truncated: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    ticks: list[torch.Tensor] = []
    returns = [0.0] * learner.config.num_envs
    completed_progress = [0.0] * learner.config.num_envs
    episode_counts = [0] * learner.config.num_envs
    for step in range(learner.config.horizon):
        with torch.no_grad():
            mean, log_std = learner.actor(observation)
            distribution = Normal(mean, log_std.exp())
            raw_action = distribution.sample()  # type: ignore[no-untyped-call]
            log_probability = distribution.log_prob(raw_action).sum(-1)  # type: ignore[no-untyped-call]
            value = learner.critic(observation)
            opponent_actions: list[FloatArray | None]
            if opponent is not None:
                opponent_observation = stack_team_batches(
                    [
                        build_team_observation(environment.state, team=1)
                        for environment in environments
                    ]
                ).to(learner.device)
                opponent_batch = opponent.deterministic_action(opponent_observation).cpu().numpy()
                opponent_actions = [
                    opponent_batch[world] for world in range(learner.config.num_envs)
                ]
            else:
                opponent_actions = [None] * learner.config.num_envs
        observations.append(observation)
        actions.append(raw_action)
        log_probabilities.append(log_probability)
        values.append(value)
        ticks.append(
            torch.tensor(
                [[int(environment.state[1])] * 3 for environment in environments],
                dtype=torch.int64,
                device=learner.device,
            )
        )
        blue_actions = torch.tanh(raw_action).cpu().numpy()
        next_observations: list[TeamBatch] = []
        step_rewards: list[float] = []
        step_terminated: list[bool] = []
        for world, environment in enumerate(environments):
            next_observation, reward, done, info = environment.step(
                blue_actions[world],
                opponent_actions[world],
            )
            returns[world] += reward.total
            is_terminated = bool(int(info["events"]) & 0b11)
            step_rewards.append(reward.total)
            step_terminated.append(is_terminated)
            if done:
                completed_progress[world] += environment.progress_score()
                episode_counts[world] += 1
                next_observation = environment.reset(
                    seed + (episode_counts[world] + 1) * learner.config.num_envs + world
                )
                environment.mark_progress_origin()
            next_observations.append(next_observation)
        rewards.append(
            torch.tensor(
                [[reward] * 3 for reward in step_rewards],
                dtype=torch.float32,
                device=learner.device,
            )
        )
        terminated.append(
            torch.tensor(
                [[value] * 3 for value in step_terminated],
                dtype=torch.bool,
                device=learner.device,
            )
        )
        truncated.append(
            torch.full(
                (learner.config.num_envs, 3),
                step == learner.config.horizon - 1,
                dtype=torch.bool,
                device=learner.device,
            )
        )
        observation = stack_team_batches(next_observations).to(learner.device)
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
        batch_size=[len(observations), learner.config.num_envs, 3],
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
    progress = (
        sum(
            completed + environment.progress_score()
            for completed, environment in zip(completed_progress, environments, strict=True)
        )
        / learner.config.num_envs
    )
    return TeamTrajectory(metadata, data), sum(returns) / learner.config.num_envs, progress


def train_iteration(
    learner: MarlLearner,
    opponent: SharedActor | None,
    config_json: str,
    state_json: str,
    *,
    iteration: int,
    seed: int,
    opponent_id: str,
    checkpoint: Path | None,
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
    if checkpoint is not None:
        learner.save(checkpoint)
    return IterationResult(
        iteration=iteration,
        policy_version=learner.policy_version,
        opponent=opponent_id,
        seed=seed,
        frames=len(trajectory.data) * learner.config.num_envs,
        return_total=total_return,
        progress=progress,
        checkpoint=str(checkpoint.resolve()) if checkpoint is not None else None,
        losses=losses,
    )
