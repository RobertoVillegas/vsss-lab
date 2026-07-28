"""Command-line entry point for M5 training and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vsss_train.config import load_config
from vsss_train.ppo import ActorCritic, evaluate, load_checkpoint, train
from vsss_train.task import GoToTargetEnv


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("command", choices=("train", "evaluate"))
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--match-config", type=Path, required=True)
    result.add_argument("--match-state", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--metrics", type=Path, default=Path("reports/m5-metrics.jsonl"))
    result.add_argument("--resume", action="store_true")
    result.add_argument("--stage", type=int, default=5)
    result.add_argument("--episodes", type=int, default=100)
    return result


def main() -> None:
    arguments = parser().parse_args()
    config = load_config(arguments.config)
    env = GoToTargetEnv(
        arguments.match_config.read_text(),
        arguments.match_state.read_text(),
        stage=config.initial_stage if arguments.command == "train" else arguments.stage,
        max_steps=config.max_episode_steps,
        success_radius=config.success_radius,
    )
    if arguments.command == "train":
        train(
            env,
            config,
            checkpoint=arguments.checkpoint,
            metrics=arguments.metrics,
            resume=arguments.resume,
        )
        return
    device = torch.device(config.device)
    model = ActorCritic(env.observation_size, env.action_size, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    load_checkpoint(arguments.checkpoint, model, optimizer, config)
    result = evaluate(
        env, model, range(config.seed + 20_000, config.seed + 20_000 + arguments.episodes), device
    )
    result["threshold"] = 0.95
    result["passed"] = result["success_rate"] >= result["threshold"]
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
