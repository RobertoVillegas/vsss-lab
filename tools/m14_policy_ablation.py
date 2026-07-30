"""Run matched-budget MLP/GRU policy ablations under partial observability.

DEPRECATED: M14 is closed and this study is pinned to its configuration; it does
not evaluate the M24.2 parametric action space. See docs/tooling-status.md.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Literal

from vsss_league.tournament import evaluate_checkpoint_scorecard
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.marl_ppo import MarlLearner


@dataclass(frozen=True)
class ArchitectureResult:
    architecture: str
    seeds: tuple[int, ...]
    parameters: int
    wins: int
    draws: int
    losses: int
    terminal_score: float
    mean_return: float
    compute_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/m14-mappo-adaptive.toml")
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/reports/m14-policy.json"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.seeds < 2:
        parser.error("--seeds must be at least 2")
    base = replace(
        load_marl_config(arguments.config),
        device=arguments.device,
        adaptive_curriculum=False,
        scenario_suite="",
        observation_dropout=0.15,
        observation_noise_std=0.02,
        num_envs=8,
        rollout_steps=8,
        horizon=60,
        epochs=1,
        minibatch_size=96,
    )
    seeds = tuple(base.seed + 800_000 + index for index in range(arguments.seeds))
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    architectures: tuple[Literal["mlp", "gru"], ...] = ("mlp", "gru")
    results = tuple(
        _run_architecture(
            replace(base, policy_architecture=architecture),
            architecture,
            seeds,
            config_json,
            state_json,
        )
        for architecture in architectures
    )
    payload = {
        "schema_version": 1,
        "partial_observability": {"dropout": 0.15, "noise_std": 0.02},
        "paired_seeds": list(seeds),
        "results": [asdict(result) for result in results],
        "terminal_score_delta_gru_minus_mlp": (
            results[1].terminal_score - results[0].terminal_score
        ),
        "decision": (
            "adopt_gru" if results[1].terminal_score > results[0].terminal_score else "retain_mlp"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _run_architecture(
    config: MarlConfig,
    architecture: str,
    seeds: tuple[int, ...],
    config_json: str,
    state_json: str,
) -> ArchitectureResult:
    started = time.perf_counter()
    wins = draws = losses = 0
    returns: list[float] = []
    parameters = 0
    for seed in seeds:
        seeded = replace(config, seed=seed)
        learner = MarlLearner(seeded)
        parameters = sum(parameter.numel() for parameter in learner.actor.parameters())
        session = create_rollout_session(seeded, config_json, state_json)
        for iteration in range(1, 4):
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
        scorecard = evaluate_checkpoint_scorecard(
            learner.actor,
            config_json,
            state_json,
            checkpoint=Path("in-memory.pt"),
            policy_version=learner.policy_version,
            seeds=(seed + 900_000,),
            ticks=60,
            action_parser=learner.config.action_parser,
        )
        wins += scorecard.wins
        draws += scorecard.draws
        losses += scorecard.losses
    matches = wins + draws + losses
    return ArchitectureResult(
        architecture=architecture,
        seeds=seeds,
        parameters=parameters,
        wins=wins,
        draws=draws,
        losses=losses,
        terminal_score=(wins + 0.5 * draws) / matches,
        mean_return=fmean(returns),
        compute_seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
