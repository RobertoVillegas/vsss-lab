"""Measure the heterogeneity gain of a role-conditioned checkpoint (ADR 0026 addendum)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vsss_train.config import load_marl_config
from vsss_train.heterogeneity import ABLATION_MODES, measure_heterogeneity_gain
from vsss_train.marl_ppo import MarlLearner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--match-config", type=Path, required=True)
    parser.add_argument("--match-state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--ablation", choices=ABLATION_MODES, default="uniform")
    arguments = parser.parse_args()
    config = load_marl_config(arguments.config)
    learner = MarlLearner(config)
    learner.load(arguments.checkpoint)
    result = measure_heterogeneity_gain(
        learner.actor,
        arguments.match_config.read_text(),
        arguments.match_state.read_text(),
        stage=config.curriculum_stage,
        seeds=range(config.seed + 40_000, config.seed + 40_000 + arguments.seeds),
        horizon=config.horizon,
        action_repeat=config.action_repeat,
        action_parser=config.action_parser,
        ablation=arguments.ablation,
    )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
