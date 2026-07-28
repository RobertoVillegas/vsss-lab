"""Reproducible M6 observation and shared-policy microbenchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from vsss_env._native import BatchSimulator
from vsss_train.marl import SharedActor, build_team_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2_000)
    arguments = parser.parse_args()
    if arguments.iterations <= 0:
        raise ValueError("iterations must be positive")
    root = Path(__file__).parents[1]
    config = (root / "tests/golden/m1_match_config.json").read_text()
    state_json = (root / "tests/golden/m1_match_state.json").read_text()
    state = np.asarray(BatchSimulator(config, state_json, 1).reset()[0], dtype=np.float32)
    actor = SharedActor()
    observation = build_team_observation(state, team=0)
    torch.set_num_threads(1)

    started = time.perf_counter()
    for _ in range(arguments.iterations):
        build_team_observation(state, team=0)
    observation_seconds = time.perf_counter() - started

    with torch.inference_mode():
        actor.deterministic_action(observation)
        started = time.perf_counter()
        for _ in range(arguments.iterations):
            actor.deterministic_action(observation)
        actor_seconds = time.perf_counter() - started

    result = {
        "iterations": arguments.iterations,
        "observation_us": observation_seconds * 1e6 / arguments.iterations,
        "shared_actor_us": actor_seconds * 1e6 / arguments.iterations,
        "agents_per_call": 3,
        "actor_parameters": sum(parameter.numel() for parameter in actor.parameters()),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
