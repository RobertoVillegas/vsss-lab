"""Causal, stateful outcome predicates for M15 semantic skill drills."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from vsss_train.semantic_scenarios import SkillContext


class SkillStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    UNRESOLVED = "unresolved"


class SkillReason(StrEnum):
    IN_PROGRESS = "in_progress"
    CONTROLLED_CONTACT = "controlled_contact"
    THREAT_CLEARED = "threat_cleared"
    BALL_CLEARED = "ball_cleared"
    GOAL_SCORED = "goal_scored"
    PASS_RECEIVED = "pass_received"
    OPPONENT_GOAL = "opponent_goal"
    OPPONENT_TOUCH = "opponent_touch"
    WRONG_TOUCH_ORDER = "wrong_touch_order"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class SkillFrame:
    step: int
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    robot_positions: Mapping[str, tuple[float, float]]
    robot_teams: Mapping[str, str]
    events: int = 0


@dataclass(frozen=True)
class SkillOutcome:
    status: SkillStatus
    reason: SkillReason
    step: int
    controlled_touches: int
    opponent_touches: int

    @property
    def terminal(self) -> bool:
        return self.status is not SkillStatus.RUNNING


class SkillEvaluator:
    """Evaluate causal skill completion without rewarding persistent overlap."""

    def __init__(
        self,
        context: SkillContext,
        *,
        robot_radius: float,
        ball_radius: float,
        goal_half_width: float,
        confirmation_steps: int = 3,
    ) -> None:
        if confirmation_steps <= 0:
            raise ValueError("confirmation_steps must be positive")
        self.context = context
        self.contact_distance = robot_radius + ball_radius + 1e-6
        self.goal_half_width = goal_half_width
        self.confirmation_steps = confirmation_steps
        self._contacts: set[str] = set()
        self._controlled_touches = 0
        self._opponent_touches = 0
        self._primary_touched = False
        self._support_touched = False
        self._opponent_after_support = False
        self._confirmation = 0
        self._outcome: SkillOutcome | None = None

    def observe(self, frame: SkillFrame) -> SkillOutcome:
        if self._outcome is not None:
            return self._outcome
        entered = self._new_contacts(frame)
        controlled = [
            robot_id
            for robot_id in entered
            if frame.robot_teams[robot_id] == self.context.controlled_team
        ]
        opponents = [robot_id for robot_id in entered if robot_id not in controlled]
        self._controlled_touches += len(controlled)
        self._opponent_touches += len(opponents)
        self._primary_touched |= self.context.controlled_robot_id in controlled
        if self.context.support_robot_id in controlled:
            self._support_touched = True
        if opponents and self._support_touched:
            self._opponent_after_support = True

        outcome = self._evaluate(frame, controlled, opponents)
        if outcome.status is SkillStatus.RUNNING and frame.step >= self.context.horizon:
            outcome = self._make(SkillStatus.UNRESOLVED, SkillReason.TIMEOUT, frame.step)
        if outcome.terminal:
            self._outcome = outcome
        return outcome

    def _evaluate(
        self,
        frame: SkillFrame,
        controlled: list[str],
        opponents: list[str],
    ) -> SkillOutcome:
        scored, conceded = self._goals(frame.events)
        if conceded:
            return self._make(SkillStatus.FAILURE, SkillReason.OPPONENT_GOAL, frame.step)
        family = self.context.family
        if family == "approach" and self.context.controlled_robot_id in controlled:
            return self._make(SkillStatus.SUCCESS, SkillReason.CONTROLLED_CONTACT, frame.step)
        if family in ("interception", "save_deflection"):
            safe = self._primary_touched and not self._threatens_own_goal(frame)
            if self._confirmed(safe):
                return self._make(SkillStatus.SUCCESS, SkillReason.THREAT_CLEARED, frame.step)
        elif family == "clearance":
            attack_sign = 1.0 if self.context.target_goal_x > 0 else -1.0
            safe = self._primary_touched and attack_sign * frame.ball_x > -0.10
            if self._confirmed(safe):
                return self._make(SkillStatus.SUCCESS, SkillReason.BALL_CLEARED, frame.step)
        elif family == "shot" and scored and self._primary_touched:
            return self._make(SkillStatus.SUCCESS, SkillReason.GOAL_SCORED, frame.step)
        elif family == "pass_receive":
            if self._primary_touched and not self._support_touched:
                return self._make(SkillStatus.FAILURE, SkillReason.WRONG_TOUCH_ORDER, frame.step)
            if self._opponent_after_support:
                return self._make(SkillStatus.FAILURE, SkillReason.OPPONENT_TOUCH, frame.step)
            if (
                self._support_touched
                and self.context.controlled_robot_id in controlled
                and not opponents
            ):
                return self._make(SkillStatus.SUCCESS, SkillReason.PASS_RECEIVED, frame.step)
        return self._make(SkillStatus.RUNNING, SkillReason.IN_PROGRESS, frame.step)

    def _new_contacts(self, frame: SkillFrame) -> set[str]:
        contacts = {
            robot_id
            for robot_id, position in frame.robot_positions.items()
            if math.dist(position, (frame.ball_x, frame.ball_y)) <= self.contact_distance
        }
        entered = contacts - self._contacts
        self._contacts = contacts
        return entered

    def _threatens_own_goal(self, frame: SkillFrame) -> bool:
        toward_goal = (self.context.own_goal_x - frame.ball_x) * frame.ball_vx > 0
        if not toward_goal or abs(frame.ball_vx) < 1e-8:
            return False
        time = (self.context.own_goal_x - frame.ball_x) / frame.ball_vx
        crossing_y = frame.ball_y + frame.ball_vy * time
        return time >= 0 and abs(crossing_y) <= self.goal_half_width

    def _goals(self, events: int) -> tuple[bool, bool]:
        blue_scored = bool(events & 1)
        yellow_scored = bool(events & 2)
        if self.context.controlled_team == "blue":
            return blue_scored, yellow_scored
        return yellow_scored, blue_scored

    def _confirmed(self, condition: bool) -> bool:
        self._confirmation = self._confirmation + 1 if condition else 0
        return self._confirmation >= self.confirmation_steps

    def _make(
        self,
        status: SkillStatus,
        reason: SkillReason,
        step: int,
    ) -> SkillOutcome:
        return SkillOutcome(
            status,
            reason,
            step,
            self._controlled_touches,
            self._opponent_touches,
        )
