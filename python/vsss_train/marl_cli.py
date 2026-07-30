"""M6 shared-policy preparation and deterministic evaluation CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vsss_train.config import load_marl_config
from vsss_train.marl_env import distill_dynamic_teacher, evaluate_against_random
from vsss_train.marl_ppo import MarlLearner


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("prepare", "evaluate"))
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--match-config", type=Path, required=True)
    result.add_argument("--match-state", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--samples", type=int, default=2_048)
    result.add_argument("--distill-epochs", type=int, default=20)
    result.add_argument("--seeds", type=int, default=20)
    result.add_argument("--margin", type=float, default=0.05)
    return result


def main() -> None:
    arguments = parser().parse_args()
    config = load_marl_config(arguments.config)
    match_config = arguments.match_config.read_text()
    match_state = arguments.match_state.read_text()
    learner = MarlLearner(config)
    if arguments.command == "prepare":
        loss = distill_dynamic_teacher(
            learner.actor,
            match_config,
            match_state,
            seed=config.seed,
            samples=arguments.samples,
            epochs=arguments.distill_epochs,
        )
        learner.save(arguments.checkpoint)
        print(json.dumps({"algorithm": config.algorithm, "distillation_loss": loss}))
        return
    learner.load(arguments.checkpoint)
    result = evaluate_against_random(
        learner.actor,
        match_config,
        match_state,
        stage=config.curriculum_stage,
        seeds=range(config.seed + 30_000, config.seed + 30_000 + arguments.seeds),
        horizon=config.horizon,
        action_repeat=config.action_repeat,
        required_margin=arguments.margin,
        action_parser=config.action_parser,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
