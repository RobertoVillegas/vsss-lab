"""Native synchronous self-play rollout and optimization iterations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.distributions import Normal
from vsss_train.config import MarlConfig
from vsss_train.marl import SharedActor, TeamBatch, build_team_observation, stack_team_batches
from vsss_train.marl_env import FloatArray, VectorMarlMatchEnv
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
    matches: int
    return_total: float
    progress: float
    checkpoint: str | None
    losses: dict[str, float]


@dataclass
class RolloutSession:
    """Persistent vector worlds so matches span multiple PPO updates."""

    environment: VectorMarlMatchEnv
    episode_counts: list[int]
    initialized: bool = False


def create_rollout_session(config: MarlConfig, config_json: str, state_json: str) -> RolloutSession:
    return RolloutSession(
        VectorMarlMatchEnv(
            config_json,
            state_json,
            num_envs=config.num_envs,
            stage=8,
            horizon=config.horizon,
            action_repeat=config.action_repeat,
            action_delta_coefficient=config.action_delta_coefficient,
        ),
        [0] * config.num_envs,
    )


def collect_self_play_trajectory(
    learner: MarlLearner,
    opponent: SharedActor | None,
    config_json: str,
    state_json: str,
    *,
    seed: int,
    opponent_id: str,
    session: RolloutSession | None = None,
) -> tuple[TeamTrajectory, float, float, int]:
    """Collect fixed-horizon vector self-play on the learner device."""
    if opponent is not None:
        opponent = opponent.to(learner.device)
    session = session or create_rollout_session(learner.config, config_json, state_json)
    environment = session.environment
    if session.initialized:
        observations_by_world = [
            build_team_observation(state, team=0) for state in environment.states
        ]
    else:
        observations_by_world = [
            environment.reset(world, seed + world) for world in range(learner.config.num_envs)
        ]
        session.initialized = True
    environment.mark_progress_origin()
    observation = stack_team_batches(observations_by_world).to(learner.device)
    initial_snapshot = environment.snapshot(0)
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
    completed_matches = 0
    for step in range(learner.config.rollout_steps):
        with torch.no_grad():
            mean, log_std = learner.actor(observation)
            distribution = Normal(mean, log_std.exp())
            raw_action = distribution.sample()  # type: ignore[no-untyped-call]
            log_probability = distribution.log_prob(raw_action).sum(-1)  # type: ignore[no-untyped-call]
            value = learner.critic(observation)
            opponent_actions: FloatArray | None
            if opponent is not None:
                opponent_observation = stack_team_batches(
                    [build_team_observation(state, team=1) for state in environment.states]
                ).to(learner.device)
                opponent_actions = opponent.deterministic_action(opponent_observation).cpu().numpy()
            else:
                opponent_actions = None
        observations.append(observation)
        actions.append(raw_action)
        log_probabilities.append(log_probability)
        values.append(value)
        ticks.append(
            torch.tensor(
                [[int(state[1])] * 3 for state in environment.states],
                dtype=torch.int64,
                device=learner.device,
            )
        )
        blue_actions = torch.tanh(raw_action).cpu().numpy()
        (
            next_observation,
            step_rewards,
            step_done,
            _step_events,
            step_terminated,
        ) = environment.step(blue_actions, opponent_actions)
        returns = [
            total + float(reward) for total, reward in zip(returns, step_rewards, strict=True)
        ]
        progress_scores = environment.progress_scores()
        reset_occurred = False
        for world, done in enumerate(step_done):
            if done:
                completed_progress[world] += float(progress_scores[world])
                session.episode_counts[world] += 1
                completed_matches += 1
                environment.reset(
                    world,
                    seed + (session.episode_counts[world] + 1) * learner.config.num_envs + world,
                )
                reset_occurred = True
        if reset_occurred:
            next_observation = stack_team_batches(
                [build_team_observation(state, team=0) for state in environment.states]
            )
        rewards.append(
            torch.tensor(
                [[reward] * 3 for reward in step_rewards],
                dtype=torch.float32,
                device=learner.device,
            )
        )
        terminated.append(
            torch.tensor(
                [[bool(value)] * 3 for value in step_terminated],
                dtype=torch.bool,
                device=learner.device,
            )
        )
        truncated.append(
            torch.full(
                (learner.config.num_envs, 3),
                step == learner.config.rollout_steps - 1,
                dtype=torch.bool,
                device=learner.device,
            )
        )
        observation = next_observation.to(learner.device)
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
            completed + float(current)
            for completed, current in zip(
                completed_progress, environment.progress_scores(), strict=True
            )
        )
        / learner.config.num_envs
    )
    return (
        TeamTrajectory(metadata, data),
        sum(returns) / learner.config.num_envs,
        progress,
        completed_matches,
    )


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
    session: RolloutSession | None = None,
) -> IterationResult:
    trajectory, total_return, progress, completed_matches = collect_self_play_trajectory(
        learner,
        opponent,
        config_json,
        state_json,
        seed=seed,
        opponent_id=opponent_id,
        session=session,
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
        matches=completed_matches,
        return_total=total_return,
        progress=progress,
        checkpoint=str(checkpoint.resolve()) if checkpoint is not None else None,
        losses=losses,
    )
