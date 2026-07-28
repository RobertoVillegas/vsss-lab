"""Native synchronous self-play rollout and optimization iterations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.distributions import Normal
from vsss_train.config import MarlConfig
from vsss_train.marl import TeamBatch, build_team_observation, stack_team_batches
from vsss_train.marl_env import FloatArray, VectorMarlMatchEnv
from vsss_train.marl_ppo import (
    TRAJECTORY_SCHEMA,
    MarlLearner,
    PolicyActor,
    TeamTrajectory,
    TrajectoryMetadata,
    sample_bounded_action,
)
from vsss_train.scenarios import Scenario, ScenarioCurriculum, load_suite


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
    terminations: dict[str, int] = field(default_factory=dict)
    curriculum: dict[str, object] | None = None


@dataclass
class RolloutSession:
    """Persistent vector worlds so matches span multiple PPO updates."""

    environment: VectorMarlMatchEnv
    episode_counts: list[int]
    curriculum: ScenarioCurriculum | None = None
    scenarios: list[Scenario | None] = field(default_factory=list)
    initialized: bool = False


def create_rollout_session(config: MarlConfig, config_json: str, state_json: str) -> RolloutSession:
    curriculum = (
        ScenarioCurriculum(
            load_suite(Path(config.scenario_suite), json.loads(config_json)),
            json.loads(config_json),
            seed=config.seed,
        )
        if config.adaptive_curriculum
        else None
    )
    return RolloutSession(
        VectorMarlMatchEnv(
            config_json,
            state_json,
            num_envs=config.num_envs,
            stage=8,
            horizon=config.horizon,
            action_repeat=config.action_repeat,
            action_delta_coefficient=config.action_delta_coefficient,
            goal_coefficient=config.goal_coefficient,
            progress_coefficient=config.progress_coefficient,
            wheel_effort_coefficient=config.wheel_effort_coefficient,
            ball_direction_coefficient=config.ball_direction_coefficient,
            attacker_alignment_coefficient=config.attacker_alignment_coefficient,
            time_penalty_coefficient=config.time_penalty_coefficient,
            movement_speed_threshold=config.movement_speed_threshold,
            teammate_spacing=config.teammate_spacing,
            teammate_congestion_coefficient=config.teammate_congestion_coefficient,
            defensive_coverage_coefficient=config.defensive_coverage_coefficient,
            defensive_activation_x=config.defensive_activation_x,
            draw_penalty=config.draw_penalty,
            stagnation_penalty=config.stagnation_penalty,
            stagnation_seconds=config.stagnation_seconds,
            stagnation_ball_distance=config.stagnation_ball_distance,
        ),
        [0] * config.num_envs,
        curriculum,
        [None] * config.num_envs,
    )


def collect_self_play_trajectory(
    learner: MarlLearner,
    opponent: PolicyActor | None,
    config_json: str,
    state_json: str,
    *,
    seed: int,
    opponent_id: str,
    session: RolloutSession | None = None,
) -> tuple[TeamTrajectory, float, float, int, dict[str, int]]:
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
        observations_by_world = []
        for world in range(learner.config.num_envs):
            observations_by_world.append(_reset_world(session, world, seed + world))
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
    termination_counts = {"goal": 0, "draw": 0, "stagnation": 0}
    for step in range(learner.config.rollout_steps):
        with torch.no_grad():
            mean, log_std = learner.actor(observation)
            distribution = Normal(mean, log_std.exp())
            action, log_probability = sample_bounded_action(distribution)
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
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        ticks.append(
            torch.tensor(
                [[int(state[1])] * 3 for state in environment.states],
                dtype=torch.int64,
                device=learner.device,
            )
        )
        blue_actions = action.cpu().numpy()
        (
            next_observation,
            step_rewards,
            step_done,
            step_events,
            step_terminated,
        ) = environment.step(blue_actions, opponent_actions)
        returns = [
            total + float(reward) for total, reward in zip(returns, step_rewards, strict=True)
        ]
        progress_scores = environment.progress_scores()
        reset_occurred = False
        for world, done in enumerate(step_done):
            if done:
                reason = str(environment.last_terminal_reasons[world])
                if reason in termination_counts:
                    termination_counts[reason] += 1
                completed_progress[world] += float(progress_scores[world])
                session.episode_counts[world] += 1
                completed_matches += 1
                scenario = session.scenarios[world]
                if session.curriculum is not None and scenario is not None:
                    session.curriculum.record(
                        scenario,
                        success=bool(int(step_events[world]) & 1),
                    )
                _reset_world(
                    session,
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
        termination_counts,
    )


def train_iteration(
    learner: MarlLearner,
    opponent: PolicyActor | None,
    config_json: str,
    state_json: str,
    *,
    iteration: int,
    seed: int,
    opponent_id: str,
    checkpoint: Path | None,
    session: RolloutSession | None = None,
) -> IterationResult:
    (
        trajectory,
        total_return,
        progress,
        completed_matches,
        termination_counts,
    ) = collect_self_play_trajectory(
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
        terminations=termination_counts,
        curriculum=(
            session.curriculum.telemetry(reset=True)
            if session is not None and session.curriculum is not None
            else None
        ),
    )


def _reset_world(session: RolloutSession, world: int, index: int) -> TeamBatch:
    if session.curriculum is None:
        session.scenarios[world] = None
        return session.environment.reset(world, index)
    selection = session.curriculum.select_training(index)
    session.scenarios[world] = selection.scenario
    state = json.loads(json.dumps(selection.scenario.state))
    state.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    return session.environment.reset_state(world, state)
