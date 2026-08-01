"""Attempting a drill must never be worse than refusing to attempt it.

Run 0008 is the case this exists for. A drill paid +1 for success and -1 for failure, and a
drill that ran out of time paid nothing. Below a fifty per cent success rate that makes
abstaining the better move, and the policy found it: failures fell away as unresolved rose from
25 to 73 per cent, strikes fell from 56 to 11 per cent, and full-match scoring reached zero.
"""

from __future__ import annotations

import pytest
from vsss_train.config import MarlConfig


def resolved_timeout_penalty(config: MarlConfig) -> float:
    """What an unresolved drill costs, which defaults to what a failed one costs."""
    if config.semantic_timeout_penalty is None:
        return config.semantic_terminal_reward
    return config.semantic_timeout_penalty


def value_of_attempting(success_rate: float, config: MarlConfig) -> float:
    reward = config.semantic_terminal_reward
    return success_rate * reward - (1.0 - success_rate) * reward


def value_of_abstaining(config: MarlConfig) -> float:
    return -resolved_timeout_penalty(config)


@pytest.mark.parametrize("success_rate", [0.0, 0.1, 0.25, 0.3, 0.5, 0.75, 1.0])
def test_attempting_is_never_worse_than_abstaining(success_rate: float) -> None:
    config = MarlConfig(semantic_terminal_reward=1.0)
    assert value_of_attempting(success_rate, config) >= value_of_abstaining(config)


@pytest.mark.parametrize("success_rate", [0.25, 0.3, 0.4])
def test_the_old_setting_paid_a_policy_to_give_up(success_rate: float) -> None:
    """The behaviour being fixed, kept as the reason the default is what it is."""
    old = MarlConfig(semantic_terminal_reward=1.0, semantic_timeout_penalty=0.0)
    assert value_of_abstaining(old) > value_of_attempting(success_rate, old)


def test_an_unresolved_drill_costs_what_a_failed_one_costs_by_default() -> None:
    config = MarlConfig(semantic_terminal_reward=1.5)
    assert resolved_timeout_penalty(config) == pytest.approx(1.5)


def test_the_timeout_penalty_can_be_set_apart_from_the_failure_penalty() -> None:
    config = MarlConfig(semantic_terminal_reward=1.0, semantic_timeout_penalty=2.0)
    assert resolved_timeout_penalty(config) == pytest.approx(2.0)


def test_a_negative_timeout_penalty_is_rejected() -> None:
    with pytest.raises(ValueError, match="semantic_timeout_penalty"):
        MarlConfig(semantic_timeout_penalty=-1.0)
