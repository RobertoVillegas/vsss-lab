"""The carry gradient must pay for a scoring position, not for reaching the byline.

The requirement it exists for: a corner is close to the goal and a goal from there is possible
but not direct, so carrying the ball into one must not pay. Dragging along a touchline is the
failure mode this shape is chosen to make unprofitable — a carrier out there gets blocked or
stuck, and a reward that pays by proximity alone would teach exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vsss_train.marl_env import ROBOT_BASE, ROBOT_WIDTH, _goal_mouth_potential

CONFIG = json.loads((Path(__file__).parents[1] / "tests/golden/m1_match_config.json").read_text())


def state_with_ball(x: float, y: float) -> np.ndarray:
    state = np.zeros(ROBOT_BASE + 6 * ROBOT_WIDTH + 1, dtype=np.float32)
    state[5], state[6] = x, y
    for slot in range(6):
        base = ROBOT_BASE + slot * ROBOT_WIDTH
        state[base + 1] = 0.0 if slot < 3 else 1.0
        state[base + 10] = 1.0
    return state


def potential(x: float, y: float, team: int = 0) -> float:
    return _goal_mouth_potential(state_with_ball(x, y), CONFIG, team)


def test_a_corner_is_worth_less_than_midfield_despite_being_closer() -> None:
    corner = potential(0.70, 0.62)
    midfield = potential(0.10, 0.0)
    assert corner < midfield, f"corner {corner:.3f} must not beat midfield {midfield:.3f}"


def test_carrying_down_a_touchline_into_the_corner_does_not_pay() -> None:
    assert potential(0.70, 0.62) < potential(0.45, 0.60)


def test_carrying_to_the_front_of_the_goal_pays() -> None:
    assert potential(0.60, 0.0) - potential(0.10, 0.0) > 0.25


def test_the_maximum_is_in_front_of_the_goal() -> None:
    front = potential(0.72, 0.0)
    assert front > 0.9
    for x, y in ((0.72, 0.62), (0.0, 0.0), (-0.70, 0.0), (0.72, 0.45)):
        assert potential(x, y) < front


def test_it_is_bounded() -> None:
    for x in np.linspace(-0.74, 0.74, 21):
        for y in np.linspace(-0.64, 0.64, 17):
            assert 0.0 <= potential(float(x), float(y)) <= 1.0


def test_the_two_teams_see_mirrored_fields() -> None:
    for x, y in ((0.60, 0.0), (0.30, 0.25), (-0.40, -0.15)):
        assert potential(x, y, team=0) == pytest.approx(potential(-x, y, team=1), abs=1e-6)


def test_behind_the_goal_line_is_worth_nothing() -> None:
    assert potential(0.80, 0.0) == pytest.approx(0.0)
