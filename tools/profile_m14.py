"""Profile the vector rollout boundary before considering alternate physics."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from vsss_league.training import create_rollout_session
from vsss_train.config import load_marl_config
from vsss_train.marl import SharedActor, build_team_observation, stack_team_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/m13-mappo-directional.toml")
    )
    parser.add_argument(
        "--match-config", type=Path, default=Path("tests/golden/m1_match_config.json")
    )
    parser.add_argument(
        "--match-state", type=Path, default=Path("tests/golden/m1_match_state.json")
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config = load_marl_config(arguments.config)
    config_json = arguments.match_config.read_text()
    state_json = arguments.match_state.read_text()
    device = torch.device(
        "cuda" if config.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    actor = SharedActor(config.hidden_size).to(device).eval()
    environment = create_rollout_session(config, config_json, state_json).environment
    for world in range(config.num_envs):
        environment.reset(world, config.seed + world)
    timings: dict[str, float] = defaultdict(float)
    actions = np.zeros((config.num_envs, 3, 2), dtype=np.float32)
    started = time.perf_counter()
    for _ in range(arguments.steps):
        phase = time.perf_counter()
        observation = stack_team_batches(
            [build_team_observation(state, team=0) for state in environment.states]
        )
        timings["observation_cpu"] += time.perf_counter() - phase
        phase = time.perf_counter()
        observation = observation.to(device)
        _synchronize(device)
        timings["host_to_device"] += time.perf_counter() - phase
        phase = time.perf_counter()
        with torch.inference_mode():
            action_tensor = actor.deterministic_action(observation)
        _synchronize(device)
        timings["inference"] += time.perf_counter() - phase
        phase = time.perf_counter()
        actions = action_tensor.cpu().numpy()
        _synchronize(device)
        timings["device_to_host"] += time.perf_counter() - phase
        phase = time.perf_counter()
        environment.step(actions, None)
        timings["physics_reward_reset"] += time.perf_counter() - phase
    elapsed = time.perf_counter() - started
    total = sum(timings.values())
    report = {
        "schema_version": 1,
        "device": str(device),
        "worlds": config.num_envs,
        "decision_steps": arguments.steps,
        "environment_frames": arguments.steps * config.num_envs * config.action_repeat,
        "elapsed_seconds": elapsed,
        "frames_per_second": arguments.steps * config.num_envs * config.action_repeat / elapsed,
        "phases": {
            name: {
                "seconds": seconds,
                "fraction": seconds / total if total else 0.0,
            }
            for name, seconds in timings.items()
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload)
    print(payload, end="")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
