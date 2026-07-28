"""Matched-compute M15 curriculum and reward ablation."""

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
class AblationResult:
    arm: str
    environment_steps: int
    semantic_success_rate: float
    semantic_unresolved_rate: float
    terminal_score: float
    mean_return: float
    compute_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.seeds < 2 or arguments.iterations <= 0:
        parser.error("ablation requires at least two seeds and one iteration")
    m14 = load_marl_config("experiments/configs/m14-mappo-adaptive.toml")
    m15 = load_marl_config("experiments/configs/m15-mappo-semantic.toml")
    shared = {
        "device": arguments.device,
        "num_envs": 8,
        "rollout_steps": 8,
        "horizon": 60,
        "epochs": 1,
        "minibatch_size": 96,
        "curriculum_heuristic_iterations": 10_000,
    }
    arms = (
        ("m14_static", replace(m14, **shared)),
        ("semantic_predicates", replace(m15, semantic_terminal_reward=0.0, **shared)),
        ("semantic_terminal", replace(m15, semantic_terminal_reward=2.0, **shared)),
        (
            "semantic_dense",
            replace(
                m15,
                semantic_terminal_reward=2.0,
                progress_coefficient=0.5,
                **shared,
            ),
        ),
        (
            "full_match_heavy",
            replace(m15, semantic_full_match_fraction=0.75, **shared),
        ),
    )
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    seeds = tuple(m15.seed + 810_000 + index for index in range(arguments.seeds))
    results = tuple(
        _run_arm(
            name,
            config,
            seeds,
            arguments.iterations,
            config_json,
            state_json,
        )
        for name, config in arms
    )
    baseline = results[0]
    candidate = results[2]
    payload = {
        "schema_version": 1,
        "paired_seeds": list(seeds),
        "iterations": arguments.iterations,
        "matched_environment_steps": results[0].environment_steps,
        "arms": [asdict(result) for result in results],
        "probe_decision": (
            "continue_bounded_m15"
            if candidate.semantic_success_rate > baseline.semantic_success_rate
            and candidate.terminal_score >= baseline.terminal_score
            else "revise_before_large_run"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _run_arm(
    name: str,
    config: MarlConfig,
    seeds: tuple[int, ...],
    iterations: int,
    config_json: str,
    state_json: str,
) -> AblationResult:
    started = time.perf_counter()
    returns: list[float] = []
    semantic_success: list[float] = []
    semantic_unresolved: list[float] = []
    terminal_scores: list[float] = []
    for seed in seeds:
        seeded = replace(config, seed=seed)
        learner = MarlLearner(seeded)
        session = create_rollout_session(seeded, config_json, state_json)
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
        unresolved = sum(trial.status == "unresolved" for trial in semantic.trials)
        semantic_success.append(successes / semantic.attempts)
        semantic_unresolved.append(unresolved / semantic.attempts)
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
    return AblationResult(
        arm=name,
        environment_steps=len(seeds) * iterations * config.num_envs * config.rollout_steps,
        semantic_success_rate=fmean(semantic_success),
        semantic_unresolved_rate=fmean(semantic_unresolved),
        terminal_score=fmean(terminal_scores),
        mean_return=fmean(returns),
        compute_seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
