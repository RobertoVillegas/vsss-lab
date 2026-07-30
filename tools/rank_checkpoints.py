"""Rank selected league checkpoints against a fixed heuristic opponent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vsss_league.tournament import evaluate_checkpoint_scorecard
from vsss_train.config import load_marl_config
from vsss_train.marl_ppo import MarlLearner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iterations", required=True, help="comma-separated checkpoint iterations")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.seeds <= 0:
        parser.error("--seeds must be positive")
    iterations = tuple(int(value) for value in arguments.iterations.split(","))
    config = load_marl_config(arguments.config)
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    seeds = tuple(range(config.seed + 50_000, config.seed + 50_000 + arguments.seeds))
    scorecards = []
    for iteration in iterations:
        checkpoint = arguments.run_dir / "checkpoints" / f"iteration-{iteration:06d}.pt"
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
        learner = MarlLearner(config)
        learner.load(checkpoint)
        scorecards.append(
            evaluate_checkpoint_scorecard(
                learner.actor.eval(),
                config_json,
                state_json,
                checkpoint=checkpoint,
                policy_version=learner.policy_version,
                seeds=seeds,
                ticks=config.horizon,
                action_parser=config.action_parser,
            )
        )
    ranked = sorted(
        scorecards,
        key=lambda card: (
            card.wins - card.losses,
            card.goals_for - card.goals_against,
            card.mean_progress,
        ),
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "config": str(arguments.config.resolve()),
        "seeds": arguments.seeds,
        "ranking": [asdict(card) for card in ranked],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
