"""Paired M18 PPO architecture and causal-reward screening."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean

from vsss_league.tournament import evaluate_checkpoint_scorecard
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.marl_ppo import MarlLearner
from vsss_train.semantic_evaluation import evaluate_semantic_skills
from vsss_train.semantic_scenarios import SemanticSkillCurriculum


@dataclass(frozen=True)
class Arm:
    name: str
    hidden_size: int
    activation: str = "tanh"
    layer_norm: bool = False
    epochs: int = 4
    useful_touch_impulse: float = 0.0
    ball_direction: float = 1.0


@dataclass(frozen=True)
class ArmResult:
    arm: str
    actor_parameters: int
    critic_parameters: int
    environment_steps: int
    frames_per_second: float
    mean_return: float
    mean_progress: float
    mean_approx_kl: float
    mean_clip_fraction: float
    mean_value_loss: float
    semantic_success_rate: float
    semantic_unresolved_rate: float
    semantic_family_success: dict[str, float]
    promotion_floors_passed: int
    terminal_score: float
    compute_seconds: float


ARMS = (
    Arm("128_tanh", 128),
    Arm("256_tanh", 256),
    Arm("256_tanh_ln", 256, layer_norm=True),
    Arm("256_relu_ln", 256, activation="relu", layer_norm=True),
    Arm("256_relu_ln_2epoch", 256, activation="relu", layer_norm=True, epochs=2),
    Arm(
        "256_relu_ln_impulse",
        256,
        activation="relu",
        layer_norm=True,
        useful_touch_impulse=0.25,
        ball_direction=0.25,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=64)
    arguments = parser.parse_args()
    compute_settings = (arguments.iterations, arguments.worlds, arguments.rollout_steps)
    if arguments.seeds < 2 or min(compute_settings) <= 0:
        parser.error("screening requires at least two seeds and positive compute settings")

    base = load_marl_config("experiments/configs/m17-mappo-coordination.toml")
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    seeds = tuple(base.seed + 1_800_000 + index for index in range(arguments.seeds))
    results = [
        run_arm(
            arm,
            replace(
                base,
                device=arguments.device,
                num_envs=arguments.worlds,
                rollout_steps=arguments.rollout_steps,
                minibatch_size=min(
                    base.minibatch_size,
                    arguments.worlds * arguments.rollout_steps * 3,
                ),
                curriculum_heuristic_iterations=10_000,
            ),
            seeds,
            arguments.iterations,
            config_json,
            state_json,
        )
        for arm in ARMS
    ]
    ranked = sorted(
        results,
        key=lambda result: (
            result.promotion_floors_passed,
            result.semantic_success_rate,
            result.terminal_score,
            -result.semantic_unresolved_rate,
            result.frames_per_second,
        ),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "protocol": {
            "paired_seeds": list(seeds),
            "iterations": arguments.iterations,
            "worlds": arguments.worlds,
            "rollout_steps": arguments.rollout_steps,
            "environment_steps_per_arm": results[0].environment_steps,
            "selection_order": [
                "promotion_floors_passed",
                "semantic_success_rate",
                "terminal_score",
                "semantic_unresolved_rate",
                "frames_per_second",
            ],
        },
        "arms": [asdict(result) for result in results],
        "ranking": [result.arm for result in ranked],
        "candidate": ranked[0].arm,
        "decision": (
            "candidate_requires_longer_confirmation"
            if ranked[0].promotion_floors_passed < len(base.semantic_promotion_floors)
            else "candidate_passed_short_screen"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def run_arm(
    arm: Arm,
    base: MarlConfig,
    seeds: tuple[int, ...],
    iterations: int,
    config_json: str,
    state_json: str,
) -> ArmResult:
    config = replace(
        base,
        hidden_size=arm.hidden_size,
        network_activation=arm.activation,
        layer_norm=arm.layer_norm,
        epochs=arm.epochs,
        useful_touch_impulse_coefficient=arm.useful_touch_impulse,
        ball_direction_coefficient=arm.ball_direction,
        policy_id=f"m18-{arm.name}",
    )
    started = time.perf_counter()
    returns: list[float] = []
    progress: list[float] = []
    losses: dict[str, list[float]] = {
        "approx_kl": [],
        "clip_fraction": [],
        "value_loss": [],
    }
    semantic_success: list[float] = []
    semantic_unresolved: list[float] = []
    family_rates: dict[str, list[float]] = {}
    terminal_scores: list[float] = []
    actor_parameters = critic_parameters = 0
    training_seconds = 0.0
    for seed in seeds:
        seeded = replace(config, seed=seed)
        learner = MarlLearner(seeded)
        actor_parameters = sum(parameter.numel() for parameter in learner.actor.parameters())
        critic_parameters = sum(parameter.numel() for parameter in learner.critic.parameters())
        session = create_rollout_session(seeded, config_json, state_json)
        training_started = time.perf_counter()
        for iteration in range(1, iterations + 1):
            result = train_iteration(
                learner,
                None,
                config_json,
                state_json,
                iteration=iteration,
                seed=seed + iteration,
                opponent_id="heuristic",
                checkpoint=None,
                session=session,
            )
            returns.append(result.return_total)
            progress.append(result.progress)
            for metric in losses:
                losses[metric].append(result.losses[metric])
        training_seconds += time.perf_counter() - training_started
        holdouts = SemanticSkillCurriculum(
            json.loads(state_json),
            json.loads(config_json),
            seed=seed,
        ).holdouts(seeds=(seed + 90_000,))
        semantic = evaluate_semantic_skills(
            learner.actor,
            holdouts,
            config_json,
            state_json,
            device=learner.device,
        )
        successes = sum(trial.status == "success" for trial in semantic.trials)
        semantic_success.append(successes / semantic.attempts)
        semantic_unresolved.append(
            sum(trial.status == "unresolved" for trial in semantic.trials) / semantic.attempts
        )
        for family in semantic.families:
            family_rates.setdefault(family.family, []).append(family.success_rate)
        scorecard = evaluate_checkpoint_scorecard(
            learner.actor,
            config_json,
            state_json,
            checkpoint=Path("in-memory.pt"),
            policy_version=learner.policy_version,
            seeds=(seed + 100_000,),
            ticks=60,
        )
        terminal_scores.append((scorecard.wins + 0.5 * scorecard.draws) / scorecard.matches)
    elapsed = time.perf_counter() - started
    environment_steps = len(seeds) * iterations * config.num_envs * config.rollout_steps
    averaged_families = {name: fmean(values) for name, values in sorted(family_rates.items())}
    floors_passed = sum(
        averaged_families.get(name, 0.0) >= floor
        for name, floor in config.semantic_promotion_floors.items()
    )
    return ArmResult(
        arm=arm.name,
        actor_parameters=actor_parameters,
        critic_parameters=critic_parameters,
        environment_steps=environment_steps,
        frames_per_second=environment_steps / training_seconds,
        mean_return=fmean(returns),
        mean_progress=fmean(progress),
        mean_approx_kl=fmean(losses["approx_kl"]),
        mean_clip_fraction=fmean(losses["clip_fraction"]),
        mean_value_loss=fmean(losses["value_loss"]),
        semantic_success_rate=fmean(semantic_success),
        semantic_unresolved_rate=fmean(semantic_unresolved),
        semantic_family_success=averaged_families,
        promotion_floors_passed=floors_passed,
        terminal_score=fmean(terminal_scores),
        compute_seconds=elapsed,
    )


if __name__ == "__main__":
    main()
