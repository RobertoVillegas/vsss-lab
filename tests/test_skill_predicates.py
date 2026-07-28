from __future__ import annotations

from vsss_train.semantic_scenarios import SkillContext
from vsss_train.skill_predicates import (
    SkillEvaluator,
    SkillFrame,
    SkillReason,
    SkillStatus,
)


def context(family: str, *, horizon: int = 20) -> SkillContext:
    return SkillContext(
        family=family,  # type: ignore[arg-type]
        controlled_team="blue",
        controlled_robot_id="R0",
        support_robot_id="R1" if family == "pass_receive" else None,
        target_goal_x=0.75,
        own_goal_x=-0.75,
        target_y=0.0,
        target_half_width=0.1,
        initial_ball_speed=0.4,
        initial_threat=family in ("interception", "save_deflection", "clearance"),
        horizon=horizon,
        parameter_hash="parameters",
        state_hash="state",
    )


def frame(
    step: int,
    *,
    ball: tuple[float, float] = (0.0, 0.0),
    velocity: tuple[float, float] = (0.0, 0.0),
    r0: tuple[float, float] = (0.4, 0.4),
    r1: tuple[float, float] = (-0.4, -0.4),
    opponent: tuple[float, float] = (0.4, -0.4),
    events: int = 0,
) -> SkillFrame:
    return SkillFrame(
        step,
        *ball,
        *velocity,
        {"R0": r0, "R1": r1, "R3": opponent},
        {"R0": "blue", "R1": "blue", "R3": "yellow"},
        events,
    )


def evaluator(family: str, *, horizon: int = 20) -> SkillEvaluator:
    return SkillEvaluator(
        context(family, horizon=horizon),
        robot_radius=0.04,
        ball_radius=0.0215,
        goal_half_width=0.2,
        confirmation_steps=2,
    )


def test_approach_counts_contact_entry_once_and_succeeds() -> None:
    subject = evaluator("approach")
    result = subject.observe(frame(1, r0=(0.03, 0.0)))
    assert result.status is SkillStatus.SUCCESS
    assert result.reason is SkillReason.CONTROLLED_CONTACT
    assert result.controlled_touches == 1
    assert subject.observe(frame(2, r0=(0.03, 0.0))) == result


def test_interception_requires_touch_and_confirmed_safe_trajectory() -> None:
    subject = evaluator("interception")
    assert subject.observe(frame(1, velocity=(-0.3, 0.0))).status is SkillStatus.RUNNING
    touched = subject.observe(frame(2, velocity=(0.2, 0.1), r0=(0.03, 0.0)))
    assert touched.status is SkillStatus.RUNNING
    result = subject.observe(frame(3, velocity=(0.2, 0.1)))
    assert (result.status, result.reason) == (
        SkillStatus.SUCCESS,
        SkillReason.THREAT_CLEARED,
    )


def test_near_miss_without_contact_times_out_unresolved() -> None:
    subject = evaluator("save_deflection", horizon=2)
    subject.observe(frame(1, velocity=(0.2, 0.3)))
    result = subject.observe(frame(2, velocity=(0.2, 0.3)))
    assert (result.status, result.reason) == (SkillStatus.UNRESOLVED, SkillReason.TIMEOUT)


def test_own_goal_is_failure() -> None:
    result = evaluator("clearance").observe(frame(1, events=2))
    assert (result.status, result.reason) == (
        SkillStatus.FAILURE,
        SkillReason.OPPONENT_GOAL,
    )


def test_shot_requires_prior_controlled_contact() -> None:
    subject = evaluator("shot")
    assert subject.observe(frame(1, events=1)).status is SkillStatus.RUNNING
    subject.observe(frame(2, r0=(0.03, 0.0)))
    result = subject.observe(frame(3, events=1))
    assert (result.status, result.reason) == (
        SkillStatus.SUCCESS,
        SkillReason.GOAL_SCORED,
    )


def test_pass_requires_ordered_uncontested_contact_chain() -> None:
    subject = evaluator("pass_receive")
    subject.observe(frame(1, r1=(0.03, 0.0)))
    subject.observe(frame(2))
    result = subject.observe(frame(3, r0=(0.03, 0.0)))
    assert (result.status, result.reason) == (
        SkillStatus.SUCCESS,
        SkillReason.PASS_RECEIVED,
    )


def test_pass_fails_after_opponent_intercepts() -> None:
    subject = evaluator("pass_receive")
    subject.observe(frame(1, r1=(0.03, 0.0)))
    subject.observe(frame(2))
    result = subject.observe(frame(3, opponent=(0.03, 0.0)))
    assert (result.status, result.reason) == (
        SkillStatus.FAILURE,
        SkillReason.OPPONENT_TOUCH,
    )


def test_rebound_resets_confirmation_window_until_trajectory_is_safe() -> None:
    subject = evaluator("save_deflection")
    subject.observe(frame(1, r0=(0.03, 0.0), velocity=(0.2, 0.2)))
    rebound = subject.observe(frame(2, velocity=(-0.3, 0.0)))
    assert rebound.status is SkillStatus.RUNNING
    subject.observe(frame(3, velocity=(0.2, 0.2)))
    result = subject.observe(frame(4, velocity=(0.2, 0.2)))
    assert result.status is SkillStatus.SUCCESS


def test_persistent_overlap_is_one_touch_not_a_farmable_contact_chain() -> None:
    subject = evaluator("pass_receive", horizon=4)
    first = subject.observe(frame(1, r1=(0.03, 0.0)))
    repeated = subject.observe(frame(2, r1=(0.03, 0.0)))
    assert first.controlled_touches == repeated.controlled_touches == 1
    subject.observe(frame(3))
    result = subject.observe(frame(4, r0=(0.03, 0.0)))
    assert result.status is SkillStatus.SUCCESS
    assert result.controlled_touches == 2
