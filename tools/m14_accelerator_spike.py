"""Benchmark device kernels and falsify a minimal CUDA physics prototype."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vsss_env._native import BatchSimulator
from vsss_train.accelerator import compare_traces, decide_accelerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--match-config", type=Path, default=Path("tests/golden/m1_match_config.json")
    )
    parser.add_argument(
        "--match-state", type=Path, default=Path("tests/golden/m1_match_state.json")
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.worlds <= 0 or arguments.steps <= 1:
        raise ValueError("worlds must be positive and steps must exceed one")
    device = _device(arguments.device)
    config_json = arguments.match_config.read_text()
    state_json = arguments.match_state.read_text()
    config = json.loads(config_json)
    generator = np.random.default_rng(14)
    actions = generator.uniform(
        -float(config["max_wheel_speed"]),
        float(config["max_wheel_speed"]),
        size=(arguments.steps, arguments.worlds, 6, 2),
    ).astype(np.float32)
    actions[: min(50, arguments.steps), 1::4, 0, :] = float(config["max_wheel_speed"])

    simulator = BatchSimulator(config_json, state_json, arguments.worlds)
    simulator.reset()
    initial_rows = []
    template = json.loads(state_json)
    for world in range(arguments.worlds):
        scenario = json.loads(json.dumps(template))
        if world % 4 == 0:
            scenario["ball"].update(x=0.70, y=0.0, vx=2.0, vy=0.0)
        elif world % 4 == 1:
            scenario["ball"].update(x=0.0, y=0.0, vx=0.0, vy=0.0)
            scenario["robots"][0]["pose"].update(x=-0.08, y=0.0, theta=0.0)
        initial_rows.append(simulator.restore_state(world, json.dumps(scenario)))
    initial = np.stack(initial_rows)
    rapier_positions = np.empty((arguments.steps, arguments.worlds, 7, 2), np.float32)
    rapier_events = np.empty((arguments.steps, arguments.worlds), np.int64)
    started = time.perf_counter()
    for step in range(arguments.steps):
        state = simulator.step(actions[step])
        rapier_positions[step] = _positions(state)
        rapier_events[step] = state[:, -1].astype(np.int64)
    rapier_seconds = time.perf_counter() - started

    candidate = torch.from_numpy(initial).to(device)
    action_tensor = torch.from_numpy(actions).to(device)
    candidate_positions = torch.empty(
        (arguments.steps, arguments.worlds, 7, 2),
        dtype=torch.float32,
        device=device,
    )
    candidate_events = torch.zeros(
        (arguments.steps, arguments.worlds),
        dtype=torch.int64,
        device=device,
    )
    _sync(device)
    started = time.perf_counter()
    for step in range(arguments.steps):
        candidate = _kinematic_step(candidate, action_tensor[step], config)
        candidate_positions[step] = _torch_positions(candidate)
        candidate_events[step] = _goal_events(candidate, config)
    _sync(device)
    candidate_seconds = time.perf_counter() - started
    parity = compare_traces(
        rapier_positions.reshape(-1, 7, 2),
        candidate_positions.cpu().numpy().reshape(-1, 7, 2),
        rapier_events.reshape(-1),
        candidate_events.cpu().numpy().reshape(-1),
        position_tolerance=1e-3,
    )
    frames = arguments.steps * arguments.worlds
    decision = decide_accelerator(
        candidate_backend=f"torch-{device.type}-kinematic-spike",
        authoritative_fps=frames / rapier_seconds,
        candidate_fps=frames / candidate_seconds,
        parity=parity,
    )
    payload = {
        "schema_version": 1,
        "device": str(device),
        "worlds": arguments.worlds,
        "steps": arguments.steps,
        "trace_scenarios": {
            "goal_entry_worlds": sum(world % 4 == 0 for world in range(arguments.worlds)),
            "robot_ball_contact_worlds": sum(world % 4 == 1 for world in range(arguments.worlds)),
        },
        "device_kernels": _benchmark_kernels(initial, device),
        "decision": asdict(decision),
        "limitations": [
            "prototype omits rigid-body contact manifolds",
            "prototype omits chamfer and goal-wall collision geometry",
            "prototype is never selectable by training",
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")


def _kinematic_step(state: torch.Tensor, actions: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    result = state.clone()
    dt = float(config["timestep"])
    wheel_radius = float(config["wheel"]["radius"])
    axle_track = float(config["wheel"]["axle_track"])
    for robot in range(6):
        base = 10 + robot * 11
        theta = result[:, base + 4]
        left = actions[:, robot, 0]
        right = actions[:, robot, 1]
        linear = wheel_radius * (left + right) * 0.5
        angular = wheel_radius * (right - left) / axle_track
        result[:, base + 2] += linear * torch.cos(theta) * dt
        result[:, base + 3] += linear * torch.sin(theta) * dt
        result[:, base + 4] += angular * dt
        result[:, base + 5] = linear * torch.cos(theta)
        result[:, base + 6] = linear * torch.sin(theta)
        result[:, base + 7] = angular
        result[:, base + 8] = left
        result[:, base + 9] = right
    result[:, 5] += result[:, 7] * dt
    result[:, 6] += result[:, 8] * dt
    result[:, 1] += 1
    result[:, 2] += dt
    return result


def _benchmark_kernels(initial: np.ndarray[Any, np.dtype[np.float32]], device: torch.device) -> dict[str, Any]:
    state = torch.from_numpy(initial).to(device)
    iterations = 1_000
    timings: dict[str, float] = {}
    for name, operation in (
        ("observation", lambda: torch.cat((state[:, 5:10], state[:, 10:21]), dim=1)),
        (
            "reward",
            lambda: (state[:, 5] - state[:, 12])
            - 0.01 * state[:, 7:9].square().sum(dim=1),
        ),
        (
            "reset",
            lambda: torch.where((state[:, -1:] > 0), torch.zeros_like(state), state),
        ),
    ):
        _sync(device)
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        _sync(device)
        timings[name] = time.perf_counter() - started
    return {
        name: {
            "seconds": seconds,
            "world_operations_per_second": iterations * len(initial) / seconds,
        }
        for name, seconds in timings.items()
    }


def _positions(state: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    positions = [state[:, 5:7]]
    positions.extend(state[:, 12 + robot * 11 : 14 + robot * 11] for robot in range(6))
    return np.stack(positions, axis=1)


def _torch_positions(state: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [state[:, 5:7], *(state[:, 12 + robot * 11 : 14 + robot * 11] for robot in range(6))],
        dim=1,
    )


def _goal_events(state: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    half_length = float(config["field"]["length"]) / 2
    half_goal = float(config["field"]["goal_width"]) / 2
    radius = float(config["ball"]["radius"])
    in_mouth = state[:, 6].abs() + radius <= half_goal
    blue = in_mouth & (state[:, 5] - radius >= half_length)
    yellow = in_mouth & (state[:, 5] + radius <= -half_length)
    return blue.to(torch.int64) | (yellow.to(torch.int64) << 1)


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(
        "cuda" if requested == "cuda" or requested == "auto" and torch.cuda.is_available() else "cpu"
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
