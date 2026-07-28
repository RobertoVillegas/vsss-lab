"""Benchmark end-to-end M15 rollout/optimization throughput."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import load_marl_config
from vsss_train.marl_ppo import MarlLearner


@dataclass(frozen=True)
class Throughput:
    requested_device: str
    actual_device: str
    worlds: int
    iterations: int
    environment_steps: int
    matches: int
    resolved_drills: int
    elapsed_seconds: float
    frames_per_second: float
    matches_per_second: float
    resolved_drills_per_second: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.worlds <= 0 or arguments.iterations <= 0:
        parser.error("worlds and iterations must be positive")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results = [_benchmark(device, arguments.worlds, arguments.iterations) for device in devices]
    payload = {
        "schema_version": 1,
        "cuda_available": torch.cuda.is_available(),
        "results": [asdict(result) for result in results],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def _benchmark(device: str, worlds: int, iterations: int) -> Throughput:
    base = load_marl_config("experiments/configs/m15-mappo-semantic.toml")
    config = replace(
        base,
        device=device,  # type: ignore[arg-type]
        num_envs=worlds,
        rollout_steps=32,
        epochs=1,
        minibatch_size=worlds * 3 * 32,
        horizon=250,
    )
    config_json = Path("tests/golden/m1_match_config.json").read_text()
    state_json = Path("tests/golden/m1_match_state.json").read_text()
    learner = MarlLearner(config)
    session = create_rollout_session(config, config_json, state_json)
    train_iteration(
        learner,
        None,
        config_json,
        state_json,
        iteration=0,
        seed=config.seed,
        opponent_id="heuristic",
        checkpoint=None,
        session=session,
    )
    if learner.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    frames = matches = resolved = 0
    for iteration in range(1, iterations + 1):
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
        resolved += sum(
            result.terminations.get(f"skill_{status}", 0)
            for status in ("success", "failure", "unresolved")
        )
    if learner.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return Throughput(
        requested_device=device,
        actual_device=learner.device.type,
        worlds=worlds,
        iterations=iterations,
        environment_steps=frames,
        matches=matches,
        resolved_drills=resolved,
        elapsed_seconds=elapsed,
        frames_per_second=frames / elapsed,
        matches_per_second=matches / elapsed,
        resolved_drills_per_second=resolved / elapsed,
    )


if __name__ == "__main__":
    main()
