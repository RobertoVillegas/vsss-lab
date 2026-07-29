from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vsss_env._native import BatchSimulator
from vsss_train.marl_env import ContactMetrics, _contact_deadlock_metrics

ROOT = Path(__file__).parents[1]
CONFIG_TEXT = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE_TEXT = (ROOT / "tests/golden/m1_match_state.json").read_text()
CONFIG = json.loads(CONFIG_TEXT)


def state() -> np.ndarray:
    return np.asarray(BatchSimulator(CONFIG_TEXT, STATE_TEXT, 1).reset()[0], dtype=np.float32)


def place(value: np.ndarray, slot: int, x: float, y: float) -> None:
    base = 10 + slot * 11
    value[base + 2 : base + 4] = (x, y)
    value[base + 10] = 1.0


def metrics(
    value: np.ndarray,
    *,
    ally: np.ndarray | None = None,
    opponent: np.ndarray | None = None,
    previous_ball: tuple[float, float] | None = None,
) -> ContactMetrics:
    return _contact_deadlock_metrics(
        value,
        0,
        contact_distance=0.082,
        grace_steps=5,
        ally_streaks=np.zeros(3, dtype=np.int64) if ally is None else ally,
        opponent_streaks=np.zeros(9, dtype=np.int64) if opponent is None else opponent,
        previous_ball=(
            value[5:7].copy()
            if previous_ball is None
            else np.asarray(previous_ball, dtype=np.float32)
        ),
        meaningful_ball_displacement=0.0024,
        config=CONFIG,
    )


def test_brief_ally_contact_has_grace_then_becomes_a_deadlock() -> None:
    value = state()
    value[5:7] = (0.25, 0.30)
    place(value, 0, -0.40, 0.0)
    place(value, 1, -0.32, 0.0)
    streaks = np.zeros(3, dtype=np.int64)
    result = None
    for _ in range(5):
        result = metrics(value, ally=streaks)
        streaks = result.ally_streaks
        assert result.ally_penalty == 0.0
    result = metrics(value, ally=streaks)
    assert result.ally_deadlocks == 1
    assert result.ally_penalty > 0.0


def test_productive_opponent_challenge_is_measured_but_not_penalized() -> None:
    value = state()
    place(value, 0, 0.0, 0.0)
    place(value, 3, 0.08, 0.0)
    value[5:7] = (0.20, 0.20)
    result = metrics(
        value,
        opponent=np.asarray([6] + [0] * 8, dtype=np.int64),
        previous_ball=(0.19, 0.20),
    )
    assert result.opponent_contacts == 1
    assert result.opponent_penalty == 0.0


def test_ally_contact_at_the_ball_is_not_penalized() -> None:
    value = state()
    place(value, 0, 0.0, 0.0)
    place(value, 1, 0.08, 0.0)
    value[5:7] = (0.04, 0.0)
    result = metrics(value, ally=np.asarray([6, 0, 0], dtype=np.int64))
    assert result.ally_contacts == 1
    assert result.ally_penalty == 0.0


def test_separating_after_sustained_contact_records_escape() -> None:
    value = state()
    place(value, 0, -0.4, 0.0)
    place(value, 1, 0.0, 0.0)
    result = metrics(value, ally=np.asarray([6, 0, 0], dtype=np.int64))
    assert result.escapes == 1
    assert result.ally_penalty == pytest.approx(0.0)
