"""Paired, multi-seed evaluation for M15 semantic skill scenarios."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from vsss_baselines import DynamicTeamController

from vsss_train.marl import build_team_observation, stack_team_batches
from vsss_train.marl_env import VectorMarlMatchEnv
from vsss_train.marl_ppo import PolicyActor
from vsss_train.semantic_scenarios import SemanticScenario
from vsss_train.skill_predicates import (
    SkillEvaluator,
    SkillOutcome,
    SkillStatus,
    skill_frame_from_native,
)

EvaluationControl = Literal["policy", "random", "heuristic"]


@dataclass(frozen=True)
class SkillEvaluation:
    family: str
    controlled_team: str
    seed: int
    difficulty: dict[str, float]
    status: str
    reason: str
    steps: int
    parameter_hash: str
    state_hash: str
    controlled_touches: int
    opponent_touches: int


@dataclass(frozen=True)
class FamilyEvaluation:
    family: str
    attempts: int
    successes: int
    failures: int
    unresolved: int
    success_rate: float
    confidence_low: float
    confidence_high: float
    mean_steps_to_resolution: float | None


@dataclass(frozen=True)
class SemanticEvaluationReport:
    schema_version: int
    control: EvaluationControl
    attempts: int
    elapsed_seconds: float
    resolved_drills_per_second: float
    physical_validity_rate: float
    mean_controlled_touches: float
    idle_spin_ratio: float
    difficulty_bands: dict[str, dict[str, float | int]]
    difficulty_levels: dict[str, dict[str, float | int]]
    families: tuple[FamilyEvaluation, ...]
    trials: tuple[SkillEvaluation, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_semantic_skills(
    actor: PolicyActor | None,
    scenarios: tuple[SemanticScenario, ...],
    config_json: str,
    state_json: str,
    *,
    control: EvaluationControl = "policy",
    device: torch.device | str = "cpu",
    action_parser: str = "continuous",
) -> SemanticEvaluationReport:
    """Evaluate paired immutable drills without using their outcomes for gradients."""
    if not scenarios:
        raise ValueError("semantic evaluation requires scenarios")
    if control == "policy" and actor is None:
        raise ValueError("policy evaluation requires an actor")
    config = json.loads(config_json)
    environment = _environment(
        config_json,
        state_json,
        len(scenarios),
        action_parser=action_parser,
    )
    evaluators: list[SkillEvaluator] = []
    active = np.ones(len(scenarios), dtype=np.bool_)
    outcomes: list[SkillOutcome | None] = [None] * len(scenarios)
    for world, scenario in enumerate(scenarios):
        team = 0 if scenario.parameters.controlled_team == "blue" else 1
        environment.set_controlled_team(world, team)
        environment.reset_state(world, scenario.scenario.state)
        evaluator = _evaluator(scenario, config)
        evaluator.observe(
            skill_frame_from_native(
                environment.states[world],
                step=0,
                events=0,
                role_assignment=environment.role_assignments[world],
                controlled_team=scenario.parameters.controlled_team,
            )
        )
        evaluators.append(evaluator)
    random = np.random.default_rng(17)
    blue = DynamicTeamController(0, 1)
    yellow = DynamicTeamController(3, -1)
    started = time.perf_counter()
    maximum_horizon = max(scenario.context.horizon for scenario in scenarios)
    for _ in range(maximum_horizon):
        idle_spin_before = environment.idle_spin_steps.copy()
        decisions_before = environment.active_agent_decisions.copy()
        actions = np.zeros((len(scenarios), 3, 2), dtype=np.float32)
        if control == "random":
            actions[active] = random.uniform(-1.0, 1.0, size=(int(active.sum()), 3, 2))
        elif control == "heuristic":
            for world, state in enumerate(environment.states):
                if active[world]:
                    controller = blue if environment.controlled_teams[world] == 0 else yellow
                    actions[world] = controller.actions(state)
        else:
            assert actor is not None
            observations = stack_team_batches(
                [
                    build_team_observation(
                        state,
                        team=int(environment.controlled_teams[world]),
                        role_assignment=environment.role_assignments[world],
                    )
                    for world, state in enumerate(environment.states)
                ]
            ).to(torch.device(device))
            with torch.no_grad():
                actions[:] = actor.deterministic_action(observations).detach().cpu().numpy()
        _, _, _, events, _ = environment.step(actions, None)
        environment.idle_spin_steps[~active] = idle_spin_before[~active]
        environment.active_agent_decisions[~active] = decisions_before[~active]
        for world, evaluator in enumerate(evaluators):
            if not active[world]:
                continue
            outcome = evaluator.observe(
                skill_frame_from_native(
                    environment.states[world],
                    step=int(environment.steps[world]),
                    events=int(events[world]),
                    role_assignment=environment.role_assignments[world],
                    controlled_team=scenarios[world].parameters.controlled_team,
                )
            )
            if outcome.terminal:
                active[world] = False
                outcomes[world] = outcome
        if not active.any():
            break
    elapsed = time.perf_counter() - started
    trials = tuple(
        _trial(
            scenario,
            outcome
            or evaluators[index].observe(
                skill_frame_from_native(
                    environment.states[index],
                    step=scenario.context.horizon,
                    events=0,
                    role_assignment=environment.role_assignments[index],
                    controlled_team=scenario.parameters.controlled_team,
                )
            ),
        )
        for index, (scenario, outcome) in enumerate(zip(scenarios, outcomes, strict=True))
    )
    families = tuple(
        _family(family, trials) for family in sorted({trial.family for trial in trials})
    )
    resolved = sum(trial.status != SkillStatus.UNRESOLVED for trial in trials)
    return SemanticEvaluationReport(
        schema_version=1,
        control=control,
        attempts=len(trials),
        elapsed_seconds=elapsed,
        resolved_drills_per_second=resolved / elapsed if elapsed else 0.0,
        physical_validity_rate=1.0,
        mean_controlled_touches=sum(trial.controlled_touches for trial in trials) / len(trials),
        idle_spin_ratio=float(environment.idle_spin_steps.sum())
        / max(1, int(environment.active_agent_decisions.sum())),
        difficulty_bands=_difficulty_bands(trials),
        difficulty_levels=_difficulty_levels(trials),
        families=families,
        trials=trials,
    )


def _environment(
    config_json: str,
    state_json: str,
    worlds: int,
    *,
    action_parser: str = "continuous",
) -> VectorMarlMatchEnv:
    return VectorMarlMatchEnv(
        config_json,
        state_json,
        num_envs=worlds,
        stage=8,
        horizon=1_000_000,
        action_repeat=4,
        action_delta_coefficient=0.0,
        goal_coefficient=0.0,
        progress_coefficient=0.0,
        wheel_effort_coefficient=0.0,
        ball_direction_coefficient=0.0,
        useful_touch_impulse_coefficient=0.0,
        goal_geometry_coefficient=0.0,
        goal_geometry_discount=0.99,
        idle_spin_coefficient=0.0,
        idle_spin_grace_seconds=0.5,
        idle_spin_turn_threshold=0.13,
        idle_spin_drive_threshold=0.07,
        idle_spin_speed_threshold=0.08,
        idle_spin_ball_distance=0.12,
        attacker_alignment_coefficient=0.0,
        time_penalty_coefficient=0.0,
        movement_speed_threshold=0.03,
        teammate_spacing=0.14,
        teammate_congestion_coefficient=0.0,
        contact_distance=0.082,
        contact_grace_seconds=0.5,
        ally_deadlock_coefficient=0.0,
        opponent_deadlock_coefficient=0.0,
        defensive_coverage_coefficient=0.0,
        defensive_activation_x=0.15,
        draw_penalty=0.0,
        stagnation_penalty=0.0,
        stagnation_seconds=1_000.0,
        stagnation_ball_distance=0.02,
        action_parser=action_parser,
    )


def _evaluator(scenario: SemanticScenario, config: dict[str, object]) -> SkillEvaluator:
    robot = config["robot"]
    ball = config["ball"]
    field = config["field"]
    assert isinstance(robot, dict) and isinstance(ball, dict) and isinstance(field, dict)
    return SkillEvaluator(
        scenario.context,
        robot_radius=math.hypot(float(robot["length"]), float(robot["width"])) / 2,
        ball_radius=float(ball["radius"]),
        goal_half_width=float(field["goal_width"]) / 2,
    )


def _trial(scenario: SemanticScenario, outcome: SkillOutcome) -> SkillEvaluation:
    return SkillEvaluation(
        family=scenario.parameters.family,
        controlled_team=scenario.parameters.controlled_team,
        seed=scenario.parameters.seed,
        difficulty={
            key: float(value) for key, value in asdict(scenario.parameters.difficulty).items()
        },
        status=outcome.status.value,
        reason=outcome.reason.value,
        steps=outcome.step,
        parameter_hash=scenario.parameters.digest,
        state_hash=scenario.scenario.digest,
        controlled_touches=outcome.controlled_touches,
        opponent_touches=outcome.opponent_touches,
    )


def _family(family: str, trials: tuple[SkillEvaluation, ...]) -> FamilyEvaluation:
    selected = [trial for trial in trials if trial.family == family]
    successes = sum(trial.status == SkillStatus.SUCCESS for trial in selected)
    failures = sum(trial.status == SkillStatus.FAILURE for trial in selected)
    unresolved = len(selected) - successes - failures
    low, high = _wilson(successes, len(selected))
    resolved_steps = [trial.steps for trial in selected if trial.status != SkillStatus.UNRESOLVED]
    return FamilyEvaluation(
        family=family,
        attempts=len(selected),
        successes=successes,
        failures=failures,
        unresolved=unresolved,
        success_rate=successes / len(selected),
        confidence_low=low,
        confidence_high=high,
        mean_steps_to_resolution=(
            sum(resolved_steps) / len(resolved_steps) if resolved_steps else None
        ),
    )


def _wilson(successes: int, attempts: int, z: float = 1.96) -> tuple[float, float]:
    if attempts == 0:
        return 0.0, 1.0
    rate = successes / attempts
    denominator = 1 + z * z / attempts
    center = (rate + z * z / (2 * attempts)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / attempts + z * z / (4 * attempts * attempts))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _difficulty_bands(
    trials: tuple[SkillEvaluation, ...],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[SkillEvaluation]] = {
        "beginner": [],
        "developing": [],
        "advanced": [],
    }
    for trial in trials:
        mean = sum(trial.difficulty.values()) / len(trial.difficulty)
        band = "beginner" if mean < 0.33 else "developing" if mean < 0.66 else "advanced"
        grouped[band].append(trial)
    return {
        band: {
            "attempts": len(selected),
            "success_rate": (
                sum(trial.status == SkillStatus.SUCCESS for trial in selected) / len(selected)
                if selected
                else 0.0
            ),
            "unresolved_rate": (
                sum(trial.status == SkillStatus.UNRESOLVED for trial in selected) / len(selected)
                if selected
                else 0.0
            ),
        }
        for band, selected in grouped.items()
    }


def _difficulty_levels(
    trials: tuple[SkillEvaluation, ...],
) -> dict[str, dict[str, float | int]]:
    """Preserve exact uniform holdout levels instead of merging nearby bands."""
    grouped: dict[str, list[SkillEvaluation]] = {}
    for trial in trials:
        mean = sum(trial.difficulty.values()) / len(trial.difficulty)
        grouped.setdefault(f"{mean:.2f}", []).append(trial)
    return {
        level: {
            "attempts": len(selected),
            "success_rate": (
                sum(trial.status == SkillStatus.SUCCESS for trial in selected) / len(selected)
                if selected
                else 0.0
            ),
            "unresolved_rate": (
                sum(trial.status == SkillStatus.UNRESOLVED for trial in selected) / len(selected)
                if selected
                else 0.0
            ),
        }
        for level, selected in sorted(grouped.items())
    }
