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
from vsss_train.marl_env import _goal_geometry_metrics, _idle_spin_flags
from vsss_train.primitives import circular_primitive_wheel_actions
from vsss_train.roles import DynamicRoleAssigner, assign_roles, role_features

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


def test_native_roles_match_the_python_reference_without_hysteresis() -> None:
    """The reward's assignment is stateless, so each state is judged on its own."""
    worlds = 8
    simulator = stirred_batch(worlds)
    for _ in range(24):
        states = np.asarray(
            simulator.step_repeated(
                np.random.default_rng(7).uniform(-1.0, 1.0, (worlds, 6, 2)).astype(np.float32) * 9,
                4,
            )
        )
        teams = np.arange(worlds, dtype=np.int64) % 2
        features, changed, uncovered, cost, names = simulator.team_roles(teams, False)

        for world, (state, team) in enumerate(zip(states, teams, strict=True)):
            want = assign_roles(state, int(team))
            assert tuple(names[world]) == want.roles, world
            np.testing.assert_array_equal(features[world], role_features(want).reshape(-1))
            np.testing.assert_array_equal(changed[world], np.asarray(want.changed, dtype=np.int64))
            assert bool(uncovered[world]) == want.uncovered, world
            assert cost[world] == pytest.approx(want.cost, abs=TOLERANCE), world


def test_native_roles_match_the_python_reference_with_hysteresis() -> None:
    """Hysteresis is a per-world history, so it only agrees if the whole sequence agrees."""
    worlds = 6
    simulator = stirred_batch(worlds)
    assigners = [DynamicRoleAssigner() for _ in range(worlds)]
    # A fresh simulator carries no role history, matching a fresh Python assigner.
    generator = np.random.default_rng(31)
    teams = np.zeros(worlds, dtype=np.int64)

    disagreements_with_stateless = 0
    for step in range(40):
        actions = generator.uniform(-1.0, 1.0, (worlds, 6, 2)).astype(np.float32) * 10.0
        states = np.asarray(simulator.step_repeated(actions, 4))
        _, changed, uncovered, cost, names = simulator.team_roles(teams, True)

        for world, state in enumerate(states):
            want = assigners[world].assign(state, 0)
            assert tuple(names[world]) == want.roles, f"step {step} world {world}"
            np.testing.assert_array_equal(changed[world], np.asarray(want.changed, dtype=np.int64))
            assert bool(uncovered[world]) == want.uncovered
            assert cost[world] == pytest.approx(want.cost, abs=TOLERANCE)
            if want.roles != assign_roles(state, 0).roles:
                disagreements_with_stateless += 1

    # If hysteresis never bit, this test would pass for the wrong reason: it would only be
    # re-checking the stateless path the previous test already covers.
    assert disagreements_with_stateless > 0


def test_restarting_roles_clears_the_history_and_seeds_the_next_decision() -> None:
    worlds = 4
    simulator = stirred_batch(worlds)
    teams = np.zeros(worlds, dtype=np.int64)
    simulator.team_roles(teams, True)
    _, changed, _, _, _ = simulator.team_roles(teams, True)
    assert changed.sum() == 0  # a repeated state cannot change a role it already holds

    # A restart both forgets and re-assigns, so the assignment it returns has nothing to have
    # changed from, and the decision after it compares against that assignment rather than none.
    _, restarted, _, _, names = simulator.restart_roles(0, 0)
    assert restarted.sum() == 0
    _, after, _, _, following = simulator.team_roles(teams, True)
    assert after[0].sum() == 0
    assert following[0] == names[0]

    with pytest.raises(ValueError, match="world index out of range"):
        simulator.restart_roles(worlds, 0)


def test_native_wheel_actions_match_the_python_reference() -> None:
    """Tokens the policy could emit, including the edges the decode rounds at."""
    worlds = 8
    simulator = stirred_batch(worlds)
    generator = np.random.default_rng(11)
    teams = np.arange(worlds, dtype=np.int64) % 2

    for step in range(20):
        states = np.asarray(
            simulator.step_repeated(
                generator.uniform(-1.0, 1.0, (worlds, 6, 2)).astype(np.float32) * 10.0, 4
            )
        )
        tokens = generator.uniform(-1.0, 1.0, (worlds, 3, 3)).astype(np.float32)
        if step == 0:
            # The skill index comes from rounding, so the halfway points decide a branch.
            tokens[:, :, 0] = np.float32(-0.5)
            tokens[:, 0, 0] = np.float32(0.5)
        actual = np.asarray(simulator.circular_wheel_actions(teams, tokens, 0.8))

        for world, (state, team) in enumerate(zip(states, teams, strict=True)):
            want = circular_primitive_wheel_actions(state, team=int(team), tokens=tokens[world])
            np.testing.assert_allclose(actual[world], want, rtol=0.0, atol=TOLERANCE)


def test_native_wheel_actions_reject_a_malformed_request() -> None:
    simulator = stirred_batch(4)
    tokens = np.zeros((4, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="one team index per world"):
        simulator.circular_wheel_actions(np.zeros(2, dtype=np.int64), tokens, 0.8)
    with pytest.raises(ValueError, match=r"tokens must have shape"):
        simulator.circular_wheel_actions(np.zeros(4, dtype=np.int64), tokens[:, :, :2], 0.8)


def test_native_goal_geometry_matches_the_python_reference_term_by_term() -> None:
    """The reward records this decomposition by name, so each term is asserted, not the sum."""
    worlds = 8
    simulator = stirred_batch(worlds)
    config = json.loads(CONFIG)
    generator = np.random.default_rng(13)
    teams = np.arange(worlds, dtype=np.int64) % 2
    terms = (
        "potential",
        "attacker_alignment",
        "goal_aperture",
        "controllable_proximity",
        "attacking_progress",
    )

    for _ in range(16):
        states = np.asarray(
            simulator.step_repeated(
                generator.uniform(-1.0, 1.0, (worlds, 6, 2)).astype(np.float32) * 10.0, 4
            )
        )
        actual = np.asarray(
            simulator.goal_geometry(
                teams,
                float(config["field"]["length"]),
                float(config["field"]["goal_width"]),
                float(config["ball"]["radius"]),
            )
        )
        for world, (state, team) in enumerate(zip(states, teams, strict=True)):
            want = _goal_geometry_metrics(state, config, int(team))
            for index, term in enumerate(terms):
                assert actual[world, index] == pytest.approx(want[term], abs=TOLERANCE), term


def test_native_idle_spin_matches_the_python_reference() -> None:
    """The flag gates a penalty, so a threshold that lands differently changes the reward."""
    worlds = 8
    simulator = stirred_batch(worlds)
    generator = np.random.default_rng(23)
    teams = np.arange(worlds, dtype=np.int64) % 2
    thresholds = {
        "angular_speed_threshold": 2.5,
        "drive_threshold": 0.25,
        "speed_threshold": 0.08,
        "ball_distance": 0.30,
    }

    flagged = 0
    for _ in range(16):
        states = np.asarray(
            simulator.step_repeated(
                generator.uniform(-1.0, 1.0, (worlds, 6, 2)).astype(np.float32) * 14.0, 4
            )
        )
        actions = generator.uniform(-1.0, 1.0, (worlds, 3, 2)).astype(np.float32)
        flags, intensity = simulator.idle_spin(teams, actions, *thresholds.values())

        for world, (state, team) in enumerate(zip(states, teams, strict=True)):
            want_flags, want_intensity = _idle_spin_flags(
                state, int(team), actions[world], **thresholds
            )
            np.testing.assert_array_equal(flags[world], want_flags)
            np.testing.assert_allclose(intensity[world], want_intensity, rtol=0.0, atol=TOLERANCE)
            flagged += int(want_flags.sum())

    # A comparison of all-false against all-false proves nothing about the threshold.
    assert flagged > 0


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
