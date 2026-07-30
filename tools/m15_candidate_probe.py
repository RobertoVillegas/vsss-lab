"""Train and screen one bounded M15 candidate before any large run."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import torch
from vsss_league.tournament import (
    evaluate_checkpoint_scorecard,
    evaluate_policy_pair_scorecard,
)
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import load_marl_config
from vsss_train.marl import SharedActor
from vsss_train.marl_env import distill_dynamic_teacher
from vsss_train.marl_ppo import MarlLearner
from vsss_train.semantic_evaluation import evaluate_semantic_skills
from vsss_train.semantic_scenarios import SemanticSkillCurriculum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    if arguments.iterations <= 0:
        parser.error("iterations must be positive")
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    base = load_marl_config("experiments/configs/m15-mappo-semantic.toml")
    config = replace(
        base,
        device=arguments.device,
        num_envs=64,
        rollout_steps=64,
        horizon=250,
        epochs=2,
        minibatch_size=12_288,
    )
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    learner = MarlLearner(config)
    distill_dynamic_teacher(
        learner.actor,
        config_json,
        state_json,
        seed=config.seed,
        samples=512,
        epochs=5,
    )
    session = create_rollout_session(config, config_json, state_json)
    started = time.perf_counter()
    frames = matches = 0
    returns = []
    for iteration in range(1, arguments.iterations + 1):
        result = train_iteration(
            learner,
            None,
            config_json,
            state_json,
            iteration=iteration,
            seed=config.seed + iteration,
            opponent_id="heuristic",
            checkpoint=None,
            session=session,
        )
        frames += result.frames
        matches += result.matches
        returns.append(result.return_total)
    elapsed = time.perf_counter() - started
    checkpoint = output / "candidate.pt"
    learner.save(checkpoint)
    holdouts = SemanticSkillCurriculum(
        json.loads(state_json),
        json.loads(config_json),
        seed=config.seed,
    ).holdouts()
    semantic = evaluate_semantic_skills(
        learner.actor,
        holdouts,
        config_json,
        state_json,
        device=learner.device,
        action_parser=config.action_parser,
    )
    paired_seeds = tuple(config.seed + 950_000 + index for index in range(5))
    heuristic = evaluate_checkpoint_scorecard(
        learner.actor,
        config_json,
        state_json,
        checkpoint=checkpoint,
        policy_version=learner.policy_version,
        seeds=paired_seeds,
        ticks=300,
        action_parser=config.action_parser,
    )
    frozen_m14 = _legacy_actor(
        Path("/home/rob/runs/vsss-training-run-0004/checkpoints/iteration-000425.pt"),
        learner.device,
    )
    historical = _legacy_actor(
        Path("/home/rob/runs/vsss-training-run-0003/checkpoints/iteration-001450.pt"),
        learner.device,
    )
    versus_m14 = evaluate_policy_pair_scorecard(
        cast(SharedActor, learner.actor),
        frozen_m14,
        config_json,
        state_json,
        candidate=f"semantic-shared@{learner.policy_version}",
        opponent="directional-shared@425",
        seeds=paired_seeds,
        ticks=300,
    )
    versus_history = evaluate_policy_pair_scorecard(
        cast(SharedActor, learner.actor),
        historical,
        config_json,
        state_json,
        candidate=f"semantic-shared@{learner.policy_version}",
        opponent="directional-shared@1450",
        seeds=paired_seeds,
        ticks=300,
    )
    payload = {
        "schema_version": 1,
        "candidate": f"semantic-shared@{learner.policy_version}",
        "checkpoint": str(checkpoint.resolve()),
        "training": {
            "iterations": arguments.iterations,
            "environment_steps": frames,
            "matches": matches,
            "elapsed_seconds": elapsed,
            "frames_per_second": frames / elapsed,
            "mean_return": sum(returns) / len(returns),
        },
        "semantic": semantic.as_dict(),
        "heuristic": asdict(heuristic),
        "frozen_m14": asdict(versus_m14),
        "historical": asdict(versus_history),
    }
    (output / "screen.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _legacy_actor(path: Path, device: torch.device) -> SharedActor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"legacy checkpoint lacks config: {path}")
    actor = SharedActor(hidden_size=int(config["hidden_size"])).to(device)
    actor.load_state_dict(payload["actor"])
    return actor.eval()


if __name__ == "__main__":
    main()
