"""Compare continuous and symmetric-lattice wheels at matched control frequency."""

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
class ActionResult:
    parser: str
    seeds: tuple[int, ...]
    action_repeat: int
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
    parser.add_argument("--output", type=Path, default=Path("experiments/reports/m14-action.json"))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()
    base = replace(
        load_marl_config(arguments.config),
        device=arguments.device,
        adaptive_curriculum=False,
        scenario_suite="",
        num_envs=8,
        rollout_steps=8,
        horizon=60,
        epochs=1,
        minibatch_size=96,
    )
    seeds = tuple(base.seed + 1_000_000 + index for index in range(arguments.seeds))
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    parsers: tuple[Literal["continuous", "lattice"], ...] = ("continuous", "lattice")
    results = tuple(
        _run(
            replace(base, action_parser=action_parser),
            action_parser,
            seeds,
            config_json,
            state_json,
        )
        for action_parser in parsers
    )
    payload = {
        "schema_version": 1,
        "matched_action_repeat": base.action_repeat,
        "paired_seeds": list(seeds),
        "results": [asdict(result) for result in results],
        "terminal_score_delta_lattice_minus_continuous": (
            results[1].terminal_score - results[0].terminal_score
        ),
        "decision": (
            "adopt_lattice"
            if results[1].terminal_score > results[0].terminal_score
            else "retain_continuous"
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _run(
    config: MarlConfig,
    parser: str,
    seeds: tuple[int, ...],
    config_json: str,
    state_json: str,
) -> ActionResult:
    started = time.perf_counter()
    wins = draws = losses = 0
    returns: list[float] = []
    for seed in seeds:
        seeded = replace(config, seed=seed)
        learner = MarlLearner(seeded)
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
            seeds=(seed + 1_100_000,),
            ticks=60,
            action_parser=learner.config.action_parser,
        )
        wins += scorecard.wins
        draws += scorecard.draws
        losses += scorecard.losses
    matches = wins + draws + losses
    return ActionResult(
        parser=parser,
        seeds=seeds,
        action_repeat=config.action_repeat,
        wins=wins,
        draws=draws,
        losses=losses,
        terminal_score=(wins + 0.5 * draws) / matches,
        mean_return=fmean(returns),
        compute_seconds=time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
