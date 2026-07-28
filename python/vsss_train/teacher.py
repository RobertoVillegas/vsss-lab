"""Bounded exact-simulator planning and verified demonstration contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SkillResult:
    score: float
    success: bool
    physically_valid: bool
    terminal_reason: str


@dataclass(frozen=True)
class Demonstration:
    skill: str
    seed: int
    actions: tuple[tuple[float, float], ...]
    score: float
    terminal_reason: str


Rollout = Callable[[FloatArray], SkillResult]


def plan_atomic_skill(
    rollout: Rollout,
    *,
    skill: str,
    seed: int,
    horizon: int = 20,
    population: int = 128,
    elites: int = 16,
    generations: int = 6,
) -> Demonstration:
    """Use bounded CEM, then replay the winner through the exact verifier."""
    if horizon <= 0 or population <= 1 or not 1 <= elites < population or generations <= 0:
        raise ValueError("invalid bounded planner dimensions")
    generator = np.random.default_rng(seed)
    mean = np.zeros((horizon, 2), dtype=np.float32)
    deviation = np.ones((horizon, 2), dtype=np.float32)
    best: FloatArray | None = None
    for _ in range(generations):
        candidates = np.clip(
            generator.normal(mean, deviation, size=(population, horizon, 2)),
            -1.0,
            1.0,
        ).astype(np.float32)
        scored = [(rollout(candidate), candidate) for candidate in candidates]
        valid = [
            (result.score, candidate) for result, candidate in scored if result.physically_valid
        ]
        if not valid:
            raise ValueError("planner produced no physically valid trajectory")
        valid.sort(key=lambda item: item[0], reverse=True)
        selected = np.stack([candidate for _, candidate in valid[:elites]])
        mean = selected.mean(axis=0).astype(np.float32)
        deviation = np.maximum(selected.std(axis=0), 0.05).astype(np.float32)
        best = valid[0][1]
    if best is None:
        raise AssertionError("planner failed to select a trajectory")
    verified = rollout(best.copy())
    if not verified.physically_valid:
        raise ValueError("winning trajectory failed exact physics replay")
    if not verified.success:
        raise ValueError("winning trajectory failed skill predicate")
    return Demonstration(
        skill=skill,
        seed=seed,
        actions=tuple((float(action[0]), float(action[1])) for action in best),
        score=verified.score,
        terminal_reason=verified.terminal_reason,
    )


def write_demonstrations(path: Path, demonstrations: tuple[Demonstration, ...]) -> None:
    payload = {
        "schema_version": 1,
        "verified_exact_simulator": True,
        "demonstrations": [asdict(item) for item in demonstrations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
