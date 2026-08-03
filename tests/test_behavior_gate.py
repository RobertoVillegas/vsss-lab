"""The behaviour gate must reject a policy that has stopped playing.

Run 0008 is the case this exists for: it held its idle-spin ratio at 0.005, scored nothing for
eight hundred iterations, and passed all seventy-four evaluations of a gate that only forbade
spinning. A stopped robot has no angular speed, so standing still passed perfectly.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from vsss_league.gates import judge_behavior
from vsss_train.config import MarlConfig

CONFIGURED = MarlConfig(
    semantic_max_idle_spin_ratio=0.08,
    semantic_max_stop_fraction=0.15,
    semantic_min_goals_per_minute=0.2,
)


def test_the_run_0008_collapse_is_rejected() -> None:
    """Its measured numbers at iteration 1848, which the old gate passed."""
    verdict = judge_behavior(
        idle_spin_ratio=0.005,
        stop_fraction=0.32,
        goals_for_per_minute=0.0,
        config=CONFIGURED,
    )
    assert not verdict.passed
    assert set(verdict.failures) == {"stop_fraction", "goals_per_minute"}
    # The pathology the old gate watched for was genuinely absent, which is the whole point.
    assert "idle_spin" not in verdict.failures


def test_healthy_early_training_is_accepted() -> None:
    """Its measured numbers at iteration 25, which should still pass."""
    verdict = judge_behavior(
        idle_spin_ratio=0.079,
        stop_fraction=0.0004,
        goals_for_per_minute=0.6,
        config=CONFIGURED,
    )
    assert verdict.passed
    assert verdict.failures == ()


@pytest.mark.parametrize(
    ("idle", "stop", "goals", "expected"),
    [
        (0.20, 0.01, 0.6, "idle_spin"),
        (0.01, 0.90, 0.6, "stop_fraction"),
        (0.01, 0.01, 0.0, "goals_per_minute"),
    ],
)
def test_each_component_can_fail_alone(
    idle: float, stop: float, goals: float, expected: str
) -> None:
    verdict = judge_behavior(
        idle_spin_ratio=idle, stop_fraction=stop, goals_for_per_minute=goals, config=CONFIGURED
    )
    assert verdict.failures == (expected,)
    assert not verdict.passed


def test_the_new_components_are_inactive_by_default() -> None:
    """Existing configurations must keep the behaviour they had."""
    verdict = judge_behavior(
        idle_spin_ratio=0.0,
        stop_fraction=1.0,
        goals_for_per_minute=0.0,
        config=MarlConfig(),
    )
    assert verdict.passed


def test_the_verdict_records_what_it_judged() -> None:
    """A rejection that does not say which component failed is a boolean to guess at."""
    payload = judge_behavior(
        idle_spin_ratio=0.005, stop_fraction=0.32, goals_for_per_minute=0.0, config=CONFIGURED
    ).as_dict()
    assert payload["failures"] == ["stop_fraction", "goals_per_minute"]
    assert payload["stop_fraction_ceiling"] == pytest.approx(0.15)
    assert payload["goals_per_minute_floor"] == pytest.approx(0.2)


def test_a_stop_fraction_ceiling_of_one_cannot_reject() -> None:
    config = replace(CONFIGURED, semantic_max_stop_fraction=1.0)
    verdict = judge_behavior(
        idle_spin_ratio=0.0, stop_fraction=1.0, goals_for_per_minute=1.0, config=config
    )
    assert verdict.passed


def test_goal_throughput_does_not_block_motion_eligibility() -> None:
    verdict = judge_behavior(
        idle_spin_ratio=0.01,
        stop_fraction=0.05,
        goals_for_per_minute=0.0,
        config=CONFIGURED,
    )

    assert not verdict.passed
    assert verdict.failures == ("goals_per_minute",)
    assert verdict.motion_eligible


def test_motion_pathology_blocks_phase_eligibility() -> None:
    verdict = judge_behavior(
        idle_spin_ratio=0.01,
        stop_fraction=0.20,
        goals_for_per_minute=1.0,
        config=CONFIGURED,
    )

    assert not verdict.motion_eligible
