"""Native synchronous self-play rollout and optimization iterations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.distributions import Categorical, Normal
from vsss_train.ablations import (
    LatticeSharedActor,
    RecurrentSharedActor,
    RecurrentState,
    SymmetricWheelLattice,
)
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
from vsss_train.semantic_scenarios import (
    SemanticScenario,
    SemanticSkillCurriculum,
)
from vsss_train.skill_predicates import (
    SkillEvaluator,
    SkillOutcome,
    SkillStatus,
    skill_frame_from_native,
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
    terminations: dict[str, int] = field(default_factory=dict)
    curriculum: dict[str, object] | None = None
    completed_episode_return: float | None = None
    match_outcomes: dict[str, int] = field(default_factory=dict)


@dataclass
class RolloutSession:
    """Persistent vector worlds so matches span multiple PPO updates."""

    environment: VectorMarlMatchEnv
    episode_counts: list[int]
    episode_returns: list[float]
    curriculum: ScenarioCurriculum | None = None
    semantic_curriculum: SemanticSkillCurriculum | None = None
    scenarios: list[Scenario | None] = field(default_factory=list)
    semantic_scenarios: list[SemanticScenario | None] = field(default_factory=list)
    skill_evaluators: list[SkillEvaluator | None] = field(default_factory=list)
    skill_outcomes: dict[str, int] = field(default_factory=dict)
    skill_trials: list[dict[str, object]] = field(default_factory=list)
    blue_host_actions: torch.Tensor | None = None
    opponent_host_actions: torch.Tensor | None = None
    initialized: bool = False


def create_rollout_session(config: MarlConfig, config_json: str, state_json: str) -> RolloutSession:
    match_config = json.loads(config_json)
    base_state = json.loads(state_json)
    curriculum = (
        ScenarioCurriculum(
            load_suite(Path(config.scenario_suite), match_config),
            match_config,
            seed=config.seed,
        )
        if config.adaptive_curriculum
        else None
    )
    semantic_curriculum = (
        SemanticSkillCurriculum(
            base_state,
            match_config,
            seed=config.seed,
            full_match_fraction=config.semantic_full_match_fraction,
            phased=config.semantic_phased_curriculum,
            phase_patience=config.semantic_phase_patience,
        )
        if config.semantic_curriculum
        else None
    )
    return RolloutSession(
        environment=VectorMarlMatchEnv(
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
            useful_touch_impulse_coefficient=config.useful_touch_impulse_coefficient,
            goal_geometry_coefficient=config.goal_geometry_coefficient,
            goal_geometry_discount=config.goal_geometry_discount,
            idle_spin_coefficient=config.idle_spin_coefficient,
            idle_spin_grace_seconds=config.idle_spin_grace_seconds,
            idle_spin_turn_threshold=config.idle_spin_turn_threshold,
            idle_spin_drive_threshold=config.idle_spin_drive_threshold,
            idle_spin_speed_threshold=config.idle_spin_speed_threshold,
            idle_spin_ball_distance=config.idle_spin_ball_distance,
            attacker_alignment_coefficient=config.attacker_alignment_coefficient,
            time_penalty_coefficient=config.time_penalty_coefficient,
            movement_speed_threshold=config.movement_speed_threshold,
            teammate_spacing=config.teammate_spacing,
            teammate_congestion_coefficient=config.teammate_congestion_coefficient,
            contact_distance=config.contact_distance,
            contact_grace_seconds=config.contact_grace_seconds,
            ally_deadlock_coefficient=config.ally_deadlock_coefficient,
            opponent_deadlock_coefficient=config.opponent_deadlock_coefficient,
            defensive_coverage_coefficient=config.defensive_coverage_coefficient,
            defensive_activation_x=config.defensive_activation_x,
            draw_penalty=config.draw_penalty,
            stagnation_penalty=config.stagnation_penalty,
            stagnation_seconds=config.stagnation_seconds,
            stagnation_ball_distance=config.stagnation_ball_distance,
        ),
        episode_counts=[0] * config.num_envs,
        episode_returns=[0.0] * config.num_envs,
        curriculum=curriculum,
        semantic_curriculum=semantic_curriculum,
        scenarios=[None] * config.num_envs,
        semantic_scenarios=[None] * config.num_envs,
        skill_evaluators=[None] * config.num_envs,
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
) -> tuple[
    TeamTrajectory,
    float,
    float,
    int,
    dict[str, int],
    list[float],
    dict[str, int],
]:
    """Collect fixed-horizon vector self-play on the learner device."""
    if opponent is not None:
        opponent = opponent.to(learner.device)
    session = session or create_rollout_session(learner.config, config_json, state_json)
    environment = session.environment
    if session.initialized:
        observations_by_world = [
            build_team_observation(state, team=int(environment.controlled_teams[world]))
            if environment.role_assignments[world] is None
            else build_team_observation(
                state,
                team=int(environment.controlled_teams[world]),
                role_assignment=environment.role_assignments[world],
            )
            for world, state in enumerate(environment.states)
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
    action_indices: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminated: list[torch.Tensor] = []
    truncated: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    ticks: list[torch.Tensor] = []
    recurrent_hidden: list[torch.Tensor] = []
    recurrent_state = (
        RecurrentState.zeros(
            worlds=learner.config.num_envs,
            agents=3,
            hidden_size=learner.config.hidden_size,
            device=learner.device,
        )
        if isinstance(learner.actor, RecurrentSharedActor)
        else None
    )
    opponent_recurrent_state = (
        RecurrentState.zeros(
            worlds=learner.config.num_envs,
            agents=3,
            hidden_size=learner.config.hidden_size,
            device=learner.device,
        )
        if isinstance(opponent, RecurrentSharedActor)
        else None
    )
    returns = [0.0] * learner.config.num_envs
    completed_progress = [0.0] * learner.config.num_envs
    completed_matches = 0
    completed_episode_returns: list[float] = []
    match_outcomes = {"win": 0, "draw": 0, "loss": 0}
    termination_counts = {
        "goal": 0,
        "draw": 0,
        "stagnation": 0,
        "skill_success": 0,
        "skill_failure": 0,
        "skill_unresolved": 0,
    }
    for step in range(learner.config.rollout_steps):
        policy_observation = _degrade_observation(
            observation,
            dropout=learner.config.observation_dropout,
            noise_std=learner.config.observation_noise_std,
            seed=seed + step,
        )
        with torch.no_grad():
            if isinstance(learner.actor, LatticeSharedActor):
                logits, _ = learner.actor(policy_observation)
                categorical = Categorical(logits=logits)
                action_index = categorical.sample()  # type: ignore[no-untyped-call]
                action = SymmetricWheelLattice().parse(action_index)
                log_probability = categorical.log_prob(  # type: ignore[no-untyped-call]
                    action_index
                )
                action_indices.append(action_index)
            elif isinstance(learner.actor, RecurrentSharedActor):
                if recurrent_state is None:
                    raise AssertionError("recurrent actor requires recurrent state")
                recurrent_hidden.append(recurrent_state.hidden.clone())
                mean, log_std, recurrent_state = learner.actor.forward_with_state(
                    policy_observation,
                    recurrent_state,
                )
            else:
                mean, log_std = learner.actor(policy_observation)
            if not isinstance(learner.actor, LatticeSharedActor):
                distribution = Normal(mean, log_std.exp())
                action, log_probability = sample_bounded_action(distribution)
            value = learner.critic(policy_observation)
            opponent_actions: FloatArray | None
            if opponent is not None:
                opponent_observation = stack_team_batches(
                    [
                        build_team_observation(
                            state,
                            team=1 - int(environment.controlled_teams[world]),
                        )
                        for world, state in enumerate(environment.states)
                    ]
                ).to(learner.device)
                if isinstance(opponent, RecurrentSharedActor):
                    if opponent_recurrent_state is None:
                        raise AssertionError("recurrent opponent requires recurrent state")
                    opponent_mean, _, opponent_recurrent_state = opponent.forward_with_state(
                        opponent_observation,
                        opponent_recurrent_state,
                    )
                    opponent_actions, session.opponent_host_actions = _host_actions(
                        torch.tanh(opponent_mean),
                        session.opponent_host_actions,
                    )
                else:
                    opponent_actions, session.opponent_host_actions = _host_actions(
                        opponent.deterministic_action(opponent_observation),
                        session.opponent_host_actions,
                    )
            else:
                opponent_actions = None
        observations.append(policy_observation)
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
        blue_actions, session.blue_host_actions = _host_actions(
            action,
            session.blue_host_actions,
        )
        (
            next_observation,
            step_rewards,
            step_done,
            step_events,
            step_terminated,
        ) = environment.step(blue_actions, opponent_actions)
        step_skill_outcomes: list[SkillOutcome | None] = [None] * learner.config.num_envs
        for world, evaluator in enumerate(session.skill_evaluators):
            if evaluator is None:
                continue
            outcome = evaluator.observe(
                skill_frame_from_native(
                    environment.states[world],
                    step=int(environment.steps[world]),
                    events=int(step_events[world]),
                    role_assignment=environment.role_assignments[world],
                    controlled_team=evaluator.context.controlled_team,
                )
            )
            step_skill_outcomes[world] = outcome
            if not outcome.terminal:
                continue
            step_done[world] = True
            step_terminated[world] = outcome.status is not SkillStatus.UNRESOLVED
            environment.last_terminal_reasons[world] = f"skill_{outcome.status.value}"
            session.skill_outcomes[outcome.status.value] = (
                session.skill_outcomes.get(outcome.status.value, 0) + 1
            )
            if outcome.status is SkillStatus.SUCCESS:
                step_rewards[world] += learner.config.semantic_terminal_reward
            elif outcome.status is SkillStatus.FAILURE:
                step_rewards[world] -= learner.config.semantic_terminal_reward
        returns = [
            total + float(reward) for total, reward in zip(returns, step_rewards, strict=True)
        ]
        session.episode_returns = [
            total + float(reward)
            for total, reward in zip(session.episode_returns, step_rewards, strict=True)
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
                completed_episode_returns.append(session.episode_returns[world])
                session.episode_returns[world] = 0.0
                if session.semantic_curriculum is None or session.semantic_scenarios[world] is None:
                    events = int(step_events[world])
                    controlled_team = int(environment.controlled_teams[world])
                    scored = bool(events & (1 if controlled_team == 0 else 2))
                    conceded = bool(events & (2 if controlled_team == 0 else 1))
                    match_outcome = "win" if scored else "loss" if conceded else "draw"
                    match_outcomes[match_outcome] += 1
                scenario = session.scenarios[world]
                if session.curriculum is not None and scenario is not None:
                    session.curriculum.record(
                        scenario,
                        success=bool(int(step_events[world]) & 1),
                    )
                semantic_scenario = session.semantic_scenarios[world]
                if session.semantic_curriculum is not None and semantic_scenario is not None:
                    evaluator = session.skill_evaluators[world]
                    skill_outcome = step_skill_outcomes[world]
                    if evaluator is not None and (
                        skill_outcome is None or not skill_outcome.terminal
                    ):
                        skill_outcome = evaluator.observe(
                            skill_frame_from_native(
                                environment.states[world],
                                step=semantic_scenario.context.horizon,
                                events=int(step_events[world]),
                                role_assignment=environment.role_assignments[world],
                                controlled_team=semantic_scenario.parameters.controlled_team,
                            )
                        )
                    if skill_outcome is not None:
                        session.skill_trials.append(
                            {
                                "schema_version": 1,
                                "scenario_id": semantic_scenario.scenario.scenario_id,
                                "family": semantic_scenario.parameters.family,
                                "controlled_team": (semantic_scenario.parameters.controlled_team),
                                "difficulty": asdict(semantic_scenario.parameters.difficulty),
                                "roster": semantic_scenario.parameters.roster,
                                "parameter_hash": semantic_scenario.parameters.digest,
                                "state_hash": semantic_scenario.scenario.digest,
                                "status": skill_outcome.status.value,
                                "reason": skill_outcome.reason.value,
                                "steps": skill_outcome.step,
                                "controlled_touches": skill_outcome.controlled_touches,
                                "opponent_touches": skill_outcome.opponent_touches,
                            }
                        )
                    session.semantic_curriculum.record(
                        semantic_scenario,
                        success=bool(
                            skill_outcome is not None
                            and skill_outcome.status is SkillStatus.SUCCESS
                        ),
                    )
                _reset_world(
                    session,
                    world,
                    seed + (session.episode_counts[world] + 1) * learner.config.num_envs + world,
                )
                reset_occurred = True
        if reset_occurred:
            next_observation = stack_team_batches(
                [
                    build_team_observation(
                        state,
                        team=int(environment.controlled_teams[world]),
                        role_assignment=environment.role_assignments[world],
                    )
                    for world, state in enumerate(environment.states)
                ]
            )
        if recurrent_state is not None or opponent_recurrent_state is not None:
            done_mask = torch.as_tensor(step_done, dtype=torch.bool, device=learner.device)
        if recurrent_state is not None:
            recurrent_state.reset_worlds(done_mask)
        if opponent_recurrent_state is not None:
            opponent_recurrent_state.reset_worlds(done_mask)
        rewards.append(
            torch.tensor(
                [[reward] * 3 for reward in step_rewards],
                dtype=torch.float32,
                device=learner.device,
            )
            * policy_observation.self_features[..., -1]
        )
        terminated.append(
            torch.tensor(
                [[bool(value)] * 3 for value in step_terminated],
                dtype=torch.bool,
                device=learner.device,
            )
        )
        truncated.append(
            torch.tensor(
                [
                    [bool(done and not terminal)] * 3
                    for done, terminal in zip(step_done, step_terminated, strict=True)
                ],
                dtype=torch.bool,
                device=learner.device,
            )
        )
        observation = next_observation.to(learner.device)
    with torch.no_grad():
        bootstrap_value = learner.critic(observation)
    batch = stack_team_batches(observations)
    trajectory_fields = {
        "tick": torch.stack(ticks),
        **dict(zip(TeamBatch._fields, batch, strict=True)),
        "action": torch.stack(actions),
        "sample_log_prob": torch.stack(log_probabilities),
        "reward_total": torch.stack(rewards),
        "terminated": torch.stack(terminated),
        "truncated": torch.stack(truncated),
        "state_value": torch.stack(values),
        "bootstrap_value": bootstrap_value.unsqueeze(0).expand(
            learner.config.rollout_steps, -1, -1
        ),
    }
    if recurrent_hidden:
        trajectory_fields["recurrent_hidden"] = torch.stack(recurrent_hidden)
    if action_indices:
        trajectory_fields["action_index"] = torch.stack(action_indices)
    data = TensorDict(
        trajectory_fields,
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
        completed_episode_returns,
        match_outcomes,
    )


def _host_actions(
    actions: torch.Tensor,
    buffer: torch.Tensor | None,
) -> tuple[FloatArray, torch.Tensor]:
    """Reuse one CPU bridge buffer instead of allocating every simulation step."""
    if buffer is None or tuple(buffer.shape) != tuple(actions.shape):
        buffer = torch.empty(actions.shape, dtype=torch.float32, device="cpu")
    buffer.copy_(actions.detach(), non_blocking=False)
    return buffer.numpy(), buffer


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
        completed_episode_returns,
        match_outcomes,
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
        curriculum=_curriculum_telemetry(session),
        completed_episode_return=(
            sum(completed_episode_returns) / len(completed_episode_returns)
            if completed_episode_returns
            else None
        ),
        match_outcomes=match_outcomes,
    )


def _reset_world(session: RolloutSession, world: int, index: int) -> TeamBatch:
    if session.semantic_curriculum is not None:
        selection = session.semantic_curriculum.select_training(index)
        semantic = selection.scenario
        session.semantic_scenarios[world] = semantic
        session.scenarios[world] = None
        if semantic is None:
            session.skill_evaluators[world] = None
            session.environment.set_controlled_team(world, index % 2)
            return session.environment.reset(world, index)
        team = 0 if semantic.parameters.controlled_team == "blue" else 1
        session.environment.set_controlled_team(world, team)
        match_config = session.semantic_curriculum.config
        robot = match_config["robot"]
        ball = match_config["ball"]
        evaluator = SkillEvaluator(
            semantic.context,
            robot_radius=(float(robot["length"]) ** 2 + float(robot["width"]) ** 2) ** 0.5 / 2,
            ball_radius=float(ball["radius"]),
            goal_half_width=float(match_config["field"]["goal_width"]) / 2,
        )
        observation = session.environment.reset_state(world, semantic.scenario.state)
        evaluator.observe(
            skill_frame_from_native(
                session.environment.states[world],
                step=0,
                events=0,
                role_assignment=session.environment.role_assignments[world],
                controlled_team=semantic.parameters.controlled_team,
            )
        )
        session.skill_evaluators[world] = evaluator
        return observation
    if session.curriculum is None:
        session.environment.set_controlled_team(world, 0)
        session.scenarios[world] = None
        return session.environment.reset(world, index)
    static_selection = session.curriculum.select_training(index)
    session.scenarios[world] = static_selection.scenario
    state = json.loads(json.dumps(static_selection.scenario.state))
    state.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    return session.environment.reset_state(world, state)


def _curriculum_telemetry(session: RolloutSession | None) -> dict[str, object] | None:
    if session is None:
        return None
    if session.semantic_curriculum is not None:
        telemetry = session.semantic_curriculum.telemetry(reset=True)
        telemetry["outcomes"] = dict(session.skill_outcomes)
        telemetry["trials"] = tuple(session.skill_trials)
        total_steps = max(1, int(session.environment.role_decisions.sum()))
        rotation_trials = [
            trial for trial in session.skill_trials if trial["family"] == "rotation_recovery"
        ]
        completed_rotations = sum(
            trial["status"] == SkillStatus.SUCCESS.value for trial in rotation_trials
        )
        telemetry["rotation"] = {
            "role_switches": int(session.environment.role_switches.sum()),
            "uncovered_world_steps": int(session.environment.uncovered_steps.sum()),
            "uncovered_ratio": float(session.environment.uncovered_steps.sum()) / total_steps,
            "completed": completed_rotations,
            "attempts": len(rotation_trials),
            "completion_rate": completed_rotations / len(rotation_trials)
            if rotation_trials
            else None,
        }
        telemetry["contact"] = {
            "ally_seconds": float(session.environment.ally_contact_steps.sum())
            * session.environment.decision_period,
            "opponent_seconds": float(session.environment.opponent_contact_steps.sum())
            * session.environment.decision_period,
            "ally_deadlocks": int(session.environment.ally_deadlocks.sum()),
            "opponent_deadlocks": int(session.environment.opponent_deadlocks.sum()),
            "escapes": int(session.environment.contact_escapes.sum()),
        }
        active_agent_decisions = max(1, int(session.environment.active_agent_decisions.sum()))
        telemetry["motion"] = {
            "idle_spin_agent_seconds": float(session.environment.idle_spin_steps.sum())
            * session.environment.decision_period,
            "idle_spin_ratio": float(session.environment.idle_spin_steps.sum())
            / active_agent_decisions,
        }
        session.environment.role_switches.fill(0)
        session.environment.uncovered_steps.fill(0)
        session.environment.role_decisions.fill(0)
        session.environment.ally_contact_steps.fill(0)
        session.environment.opponent_contact_steps.fill(0)
        session.environment.ally_deadlocks.fill(0)
        session.environment.opponent_deadlocks.fill(0)
        session.environment.contact_escapes.fill(0)
        session.environment.idle_spin_steps.fill(0)
        session.environment.active_agent_decisions.fill(0)
        session.skill_outcomes.clear()
        session.skill_trials.clear()
        return telemetry
    if session.curriculum is not None:
        return session.curriculum.telemetry(reset=True)
    return None


def _degrade_observation(
    observation: TeamBatch,
    *,
    dropout: float,
    noise_std: float,
    seed: int,
) -> TeamBatch:
    if dropout == 0.0 and noise_std == 0.0:
        return observation
    generator = torch.Generator(device=observation.self_features.device).manual_seed(seed)

    def perturb(value: torch.Tensor, *, can_drop: bool) -> torch.Tensor:
        result = value
        if noise_std:
            noise = torch.randn(
                value.shape,
                generator=generator,
                device=value.device,
                dtype=value.dtype,
            )
            result = result + noise_std * noise
        if can_drop and dropout:
            mask_shape = (*value.shape[:-1], 1)
            visible = (
                torch.rand(
                    mask_shape,
                    generator=generator,
                    device=value.device,
                )
                >= dropout
            )
            result = result * visible
        return result

    return TeamBatch(
        perturb(observation.self_features, can_drop=False),
        perturb(observation.ball, can_drop=True),
        perturb(observation.goals, can_drop=False),
        observation.context,
        perturb(observation.teammates, can_drop=True),
        perturb(observation.opponents, can_drop=True),
    )
