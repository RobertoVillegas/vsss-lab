"""Equivalence between the native observation path and its Python reference.

A port of the hot loop fails quietly, not loudly: a channel in a different order, a term
that rounds differently, a threshold that flips. The native path is therefore accepted only
when it agrees with the implementation it replaces on recorded states, and the tolerance is
set below the smallest margin any comparison downstream depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from vsss_env._native import BatchSimulator
from vsss_train.marl import build_team_observation
from vsss_train.roles import assign_roles, role_features

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()
# Downstream comparisons act on quantities of order 1e-2; float32 arithmetic reordered
# between two implementations differs by ~1e-7, so this is far below any decision boundary.
TOLERANCE = 1e-5


def stirred_batch(worlds: int, steps: int = 12) -> BatchSimulator:
    """Advance a batch so the worlds differ from each other and from their template."""
    simulator = BatchSimulator(CONFIG, STATE, worlds)
    simulator.reset()
    generator = np.random.default_rng(19)
    for _ in range(steps):
        actions = generator.uniform(-1.0, 1.0, (worlds, 6, 2)).astype(np.float32) * 12.0
        simulator.step_repeated(actions, 4)
    return simulator


def geometry() -> tuple[float, float, float]:
    """Return field length, field width and match duration from the golden config."""
    config = json.loads(CONFIG)
    field = config["field"]
    return float(field["length"]), float(field["width"]), float(config["match_duration"])


def reference(states: np.ndarray, teams: np.ndarray) -> list[np.ndarray]:
    length, width, duration = geometry()
    groups: list[list[np.ndarray]] = [[], [], [], [], [], []]
    for state, team in zip(states, teams, strict=True):
        batch = build_team_observation(
            state,
            team=int(team),
            field_length=length,
            field_width=width,
            match_duration=duration,
            role_assignment=assign_roles(state, int(team)),
        )
        for index, tensor in enumerate(batch):
            groups[index].append(tensor.reshape(-1).numpy())
    return [np.stack(group) for group in groups]


def native(simulator: BatchSimulator, states: np.ndarray, teams: np.ndarray) -> list[np.ndarray]:
    length, width, duration = geometry()
    roles = np.stack(
        [
            role_features(assign_roles(state, int(team))).reshape(-1)
            for state, team in zip(states, teams, strict=True)
        ]
    ).astype(np.float32)
    return [
        np.asarray(group)
        for group in simulator.observations(teams.astype(np.int64), roles, length, width, duration)
    ]


@pytest.mark.parametrize("teams_pattern", ["all_blue", "all_yellow", "mixed"])
def test_native_observations_match_the_python_reference(teams_pattern: str) -> None:
    worlds = 8
    simulator = stirred_batch(worlds)
    states = np.asarray(simulator.step_repeated(np.zeros((worlds, 6, 2), dtype=np.float32), 1))
    if teams_pattern == "all_blue":
        teams = np.zeros(worlds, dtype=np.int64)
    elif teams_pattern == "all_yellow":
        teams = np.ones(worlds, dtype=np.int64)
    else:
        teams = np.arange(worlds, dtype=np.int64) % 2

    expected = reference(states, teams)
    actual = native(simulator, states, teams)

    names = ("self_features", "ball", "goals", "context", "teammates", "opponents")
    for name, want, got in zip(names, expected, actual, strict=True):
        assert got.shape == want.shape, name
        largest = float(np.abs(got - want).max())
        assert largest <= TOLERANCE, f"{name} differs by {largest}"


def test_native_observations_reject_a_malformed_request() -> None:
    simulator = stirred_batch(4)
    roles = np.zeros((4, 15), dtype=np.float32)

    with pytest.raises(ValueError, match="one team index per world"):
        simulator.observations(np.zeros(3, dtype=np.int64), roles, 1.5, 1.3, 600.0)
    with pytest.raises(ValueError, match="one role row per world"):
        simulator.observations(np.zeros(4, dtype=np.int64), roles[:2], 1.5, 1.3, 600.0)


def test_native_observations_feed_the_actor_unchanged() -> None:
    """The port is only useful if its output is accepted where the reference's was."""
    worlds = 4
    simulator = stirred_batch(worlds)
    states = np.asarray(simulator.step_repeated(np.zeros((worlds, 6, 2), dtype=np.float32), 1))
    teams = np.zeros(worlds, dtype=np.int64)
    groups = native(simulator, states, teams)

    shapes = [
        (worlds, 3, 8),
        (worlds, 3, 7),
        (worlds, 3, 4),
        (worlds, 3, 9),
        (worlds, 3, 2, 6),
        (worlds, 3, 3, 6),
    ]
    tensors = [
        torch.from_numpy(group).reshape(shape) for group, shape in zip(groups, shapes, strict=True)
    ]
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
