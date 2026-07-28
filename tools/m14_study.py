"""Run a bounded, resumable M14 reward/PPO search with real simulator rollouts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import fmean

from vsss_league.tournament import evaluate_checkpoint_scorecard
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import load_marl_config
from vsss_train.marl_ppo import MarlLearner
from vsss_train.search import (
    Fidelity,
    FidelityResult,
    SearchParameters,
    create_study,
    run_multifidelity_trial,
)

ITERATIONS: dict[Fidelity, int] = {"smoke": 1, "screen": 2, "confirm": 3}
HORIZONS: dict[Fidelity, int] = {"smoke": 30, "screen": 45, "confirm": 60}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/m14-mappo-adaptive.toml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/reports/m14-study"))
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    arguments = parser.parse_args()
    if arguments.trials <= 0:
        parser.error("--trials must be positive")
    output = arguments.output_dir
    study = create_study(name="m14-reward-search-v1", storage_path=output / "study.db")
    base = replace(load_marl_config(arguments.config), device=arguments.device)
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()

    def evaluator(
        parameters: SearchParameters,
        fidelity: Fidelity,
        seeds: tuple[int, ...],
    ) -> FidelityResult:
        started = time.perf_counter()
        terminal_scores: list[float] = []
        coordination: list[float] = []
        for seed in seeds:
            config = replace(
                base,
                seed=seed,
                num_envs=8,
                rollout_steps=8,
                horizon=HORIZONS[fidelity],
                epochs=1,
                minibatch_size=96,
                learning_rate=parameters.learning_rate,
                entropy_coefficient=parameters.entropy_coefficient,
                clip_epsilon=parameters.clip_epsilon,
                goal_coefficient=parameters.goal_coefficient,
                progress_coefficient=parameters.progress_coefficient,
                teammate_congestion_coefficient=parameters.congestion_coefficient,
                defensive_coverage_coefficient=parameters.defensive_coefficient,
            )
            learner = MarlLearner(config)
            session = create_rollout_session(config, config_json, state_json)
            saturation = 0.0
            for iteration in range(1, ITERATIONS[fidelity] + 1):
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
                saturation += result.losses["action_saturation"]
            scorecard = evaluate_checkpoint_scorecard(
                learner.actor,
                config_json,
                state_json,
                checkpoint=output / "in-memory.pt",
                policy_version=learner.policy_version,
                seeds=(seed + 500_000,),
                ticks=HORIZONS[fidelity],
            )
            terminal_scores.append((scorecard.wins + 0.5 * scorecard.draws) / scorecard.matches)
            coordination.append(saturation / ITERATIONS[fidelity])
        return FidelityResult(
            terminal_score=fmean(terminal_scores),
            coordination_failure=fmean(coordination),
            compute_seconds=time.perf_counter() - started,
        )

    for _ in range(arguments.trials):
        trial = run_multifidelity_trial(
            study,
            evaluator,
            lineage_path=output / "lineage.jsonl",
            confirmation_floor=0.0,
        )
        print(
            json.dumps(
                {
                    "trial": trial.number,
                    "state": trial.state.name,
                    "values": trial.values,
                },
                sort_keys=True,
            )
        )
    summary = {
        "schema_version": 1,
        "study": study.study_name,
        "directions": [direction.name for direction in study.directions],
        "trials": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "values": trial.values,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
        ],
        "pareto_trials": [trial.number for trial in study.best_trials],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
