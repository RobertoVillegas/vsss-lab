"""Whether a policy is playing, judged from what an evaluation measured.

A gate that only forbids a pathology can be passed by doing nothing. Run 0008 collapsed to a
third of its decisions being `stop`, scored zero goals per minute for eight hundred iterations,
and passed the spin-only gate on every one of seventy-four evaluations — because a stopped robot
has no angular speed. It promoted through every curriculum phase as it went. Absence of the
pathology is necessary and is not sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vsss_train.config import MarlConfig


@dataclass(frozen=True)
class BehaviorJudgement:
    """The gate's verdict, with every component that produced it."""

    passed: bool
    idle_spin_ratio: float
    idle_spin_ceiling: float
    stop_fraction: float
    stop_fraction_ceiling: float
    goals_for_per_minute: float
    goals_per_minute_floor: float

    @property
    def failures(self) -> tuple[str, ...]:
        """Which components failed, so a rejection says why rather than only that."""
        reasons = []
        if self.idle_spin_ratio > self.idle_spin_ceiling:
            reasons.append("idle_spin")
        if self.stop_fraction > self.stop_fraction_ceiling:
            reasons.append("stop_fraction")
        if self.goals_for_per_minute < self.goals_per_minute_floor:
            reasons.append("goals_per_minute")
        return tuple(reasons)

    @property
    def motion_eligible(self) -> bool:
        """Whether play is active enough to unlock teaching phases.

        Goal throughput remains mandatory for promotion, but a zero-goal short match sample
        must not prevent the curriculum from teaching the defensive skills that create an
        integrated scoring policy.  STOP and idle spin still reject a collapsed policy.
        """
        return (
            self.idle_spin_ratio <= self.idle_spin_ceiling
            and self.stop_fraction <= self.stop_fraction_ceiling
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "idle_spin_ratio": self.idle_spin_ratio,
            "idle_spin_ceiling": self.idle_spin_ceiling,
            "stop_fraction": self.stop_fraction,
            "stop_fraction_ceiling": self.stop_fraction_ceiling,
            "goals_for_per_minute": self.goals_for_per_minute,
            "goals_per_minute_floor": self.goals_per_minute_floor,
        }


def judge_behavior(
    *,
    idle_spin_ratio: float,
    stop_fraction: float,
    goals_for_per_minute: float,
    config: MarlConfig,
) -> BehaviorJudgement:
    """Require play to be present, not only the pathology to be absent."""
    judgement = BehaviorJudgement(
        passed=False,
        idle_spin_ratio=idle_spin_ratio,
        idle_spin_ceiling=config.semantic_max_idle_spin_ratio,
        stop_fraction=stop_fraction,
        stop_fraction_ceiling=config.semantic_max_stop_fraction,
        goals_for_per_minute=goals_for_per_minute,
        goals_per_minute_floor=config.semantic_min_goals_per_minute,
    )
    return replace(judgement, passed=not judgement.failures)
