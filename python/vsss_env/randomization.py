"""Seeded domain randomization and OOD evaluation."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vsss_env.backends import FloatArray, NativeBackend


@dataclass(frozen=True)
class RealizedDomain:
    """All sampled values needed to reproduce one perturbed episode."""

    seed: int
    friction: float
    restitution: float
    motor_strength: tuple[float, ...]
    latency_steps: int
    action_drop_probability: float
    position_noise_std: float
    heading_noise_std: float


class RandomizedBackend:
    """Fault-injection wrapper that keeps scoring ground truth separate."""

    def __init__(self, config_json: str, state_json: str, suite: dict[str, Any], seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        uniform = self._rng.uniform
        self.domain = RealizedDomain(
            seed=seed,
            friction=float(uniform(*suite["friction"])),
            restitution=float(uniform(*suite["restitution"])),
            motor_strength=tuple(
                float(value)
                for value in uniform(
                    float(suite["motor_strength"][0]),
                    float(suite["motor_strength"][1]),
                    size=6,
                )
            ),
            latency_steps=int(
                self._rng.integers(
                    suite["latency_steps"][0],
                    suite["latency_steps"][1] + 1,
                )
            ),
            action_drop_probability=float(uniform(*suite["action_drop_probability"])),
            position_noise_std=float(uniform(*suite["position_noise_std"])),
            heading_noise_std=float(uniform(*suite["heading_noise_std"])),
        )
        config = json.loads(config_json)
        config["friction"] = self.domain.friction
        config["restitution"] = self.domain.restitution
        self._backend = NativeBackend(json.dumps(config), state_json)
        self._queue: deque[FloatArray] = deque()
        self._delivered = np.zeros((6, 2), dtype=np.float32)
        self.ground_truth = np.zeros(77, dtype=np.float32)

    def reset(self) -> FloatArray:
        self._queue.clear()
        self._delivered.fill(0)
        self.ground_truth = self._backend.reset()
        return self._observe()

    def step(self, actions: FloatArray) -> FloatArray:
        scaled = np.asarray(actions, dtype=np.float32) * np.asarray(
            self.domain.motor_strength, dtype=np.float32
        )[:, None]
        self._queue.append(scaled)
        if len(self._queue) > self.domain.latency_steps:
            candidate = self._queue.popleft()
            if self._rng.random() >= self.domain.action_drop_probability:
                self._delivered = candidate
        self.ground_truth = self._backend.step(self._delivered)
        return self._observe()

    def _observe(self) -> FloatArray:
        observed = self.ground_truth.copy()
        position_indices = [5, 6]
        heading_indices: list[int] = []
        for robot in range(6):
            base = 10 + robot * 11
            position_indices.extend([base + 2, base + 3])
            heading_indices.append(base + 4)
        observed[position_indices] += self._rng.normal(
            0.0, self.domain.position_noise_std, len(position_indices)
        )
        observed[heading_indices] += self._rng.normal(
            0.0, self.domain.heading_noise_std, len(heading_indices)
        )
        return observed

    def close(self) -> None:
        self._backend.close()


class PursuitPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, state: FloatArray) -> FloatArray: ...


class NominalPursuit:
    """High-gain nominal ball pursuit with no temporal filtering."""

    def reset(self) -> None:
        pass

    def act(self, state: FloatArray) -> FloatArray:
        return _pursuit_action(state, heading_gain=11.0, speed=30.0)


class RobustPursuit:
    """Filtered, conservative pursuit designed for delayed noisy observations."""

    def __init__(self) -> None:
        self._filtered: FloatArray | None = None

    def reset(self) -> None:
        self._filtered = None

    def act(self, state: FloatArray) -> FloatArray:
        if self._filtered is None:
            self._filtered = state.copy()
        else:
            self._filtered = 0.55 * self._filtered + 0.45 * state
        return _pursuit_action(self._filtered, heading_gain=2.8, speed=24.0)


def _pursuit_action(state: FloatArray, heading_gain: float, speed: float) -> FloatArray:
    dx = float(state[5] - state[12])
    dy = float(state[6] - state[13])
    desired = math.atan2(dy, dx)
    error = math.atan2(math.sin(desired - float(state[14])), math.cos(desired - float(state[14])))
    forward = speed * max(0.0, math.cos(error))
    turn = heading_gain * error / 0.025
    actions = np.zeros((6, 2), dtype=np.float32)
    actions[0] = np.clip([forward - turn, forward + turn], -60.0, 60.0)
    return actions


def evaluate_ood(
    config_path: Path,
    state_path: Path,
    suite_path: Path,
) -> dict[str, Any]:
    """Compare nominal and robust policies on paired held-out domains."""
    config = config_path.read_text()
    state_data = json.loads(state_path.read_text())
    state_data.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    state_data["ball"].update(vx=0.0, vy=0.0, omega=0.0)
    state = json.dumps(state_data)
    suite: dict[str, Any] = json.loads(suite_path.read_text())
    pairs: list[dict[str, Any]] = []
    policies: tuple[PursuitPolicy, PursuitPolicy] = (NominalPursuit(), RobustPursuit())
    for seed in range(int(suite["seeds"])):
        scores: list[float] = []
        realized: dict[str, Any] | None = None
        for policy in policies:
            backend = RandomizedBackend(config, state, suite, seed)
            observation = backend.reset()
            start = _distance_to_ball(backend.ground_truth)
            policy.reset()
            for _ in range(int(suite["ticks"])):
                observation = backend.step(policy.act(observation))
            scores.append(start - _distance_to_ball(backend.ground_truth))
            realized = asdict(backend.domain)
            backend.close()
        pairs.append({"seed": seed, "nominal": scores[0], "robust": scores[1], "domain": realized})
    nominal = float(np.mean([pair["nominal"] for pair in pairs]))
    robust = float(np.mean([pair["robust"] for pair in pairs]))
    margin = robust - nominal
    return {
        "schema_version": 1,
        "suite": suite,
        "nominal_progress": nominal,
        "robust_progress": robust,
        "margin": margin,
        "passed": robust > 0.0 and margin >= float(suite["required_margin"]),
        "pairs": pairs,
    }


def _distance_to_ball(state: FloatArray) -> float:
    return math.hypot(float(state[5] - state[12]), float(state[6] - state[13]))
