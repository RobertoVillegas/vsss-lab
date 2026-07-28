"""Evaluate a checkpoint or control policy on immutable M15 skill holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from vsss_train.config import load_marl_config
from vsss_train.marl_ppo import load_policy_actor
from vsss_train.semantic_evaluation import evaluate_semantic_skills
from vsss_train.semantic_scenarios import SemanticSkillCurriculum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--match-config", type=Path, required=True)
    parser.add_argument("--match-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--control", choices=("policy", "random", "heuristic"), default="policy")
    parser.add_argument("--seeds", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.seeds < 5:
        raise ValueError("paired semantic evaluation requires at least five seeds")
    if arguments.control == "policy" and arguments.checkpoint is None:
        raise ValueError("--checkpoint is required for policy evaluation")
    config = load_marl_config(arguments.config)
    match_config_text = arguments.match_config.read_text()
    match_state_text = arguments.match_state.read_text()
    curriculum = SemanticSkillCurriculum(
        json.loads(match_state_text),
        json.loads(match_config_text),
        seed=config.seed,
    )
    seeds = tuple(10_007 + index * 30 for index in range(arguments.seeds))
    scenarios = curriculum.holdouts(seeds=seeds)
    device = torch.device(
        "cuda" if config.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    actor = (
        load_policy_actor(arguments.checkpoint, config, device)[0]
        if arguments.checkpoint is not None
        else None
    )
    report = evaluate_semantic_skills(
        actor,
        scenarios,
        match_config_text,
        match_state_text,
        control=arguments.control,
        device=device,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
