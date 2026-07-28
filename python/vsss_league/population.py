"""Bounded behavioral league retention and distillation gates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vsss_league.evaluation import PairedEstimate


@dataclass(frozen=True)
class BehaviorCheckpoint:
    policy_id: str
    rating: float
    win_rate: float
    possession: float
    pressure: float
    congestion: float
    action_jerk: float

    @property
    def descriptor(self) -> tuple[float, ...]:
        return (self.possession, self.pressure, self.congestion, self.action_jerk)


@dataclass(frozen=True)
class ConsolidationDecision:
    selected: tuple[str, ...]
    specialist_score: float
    best_single_score: float
    lower_confidence_delta: float
    distill: bool
    reason: str


@dataclass(frozen=True)
class DistillationVerification:
    """Outcome of the positive-result gate and, when allowed, student checks."""

    attempted: bool
    preserved_terminal_outcomes: bool | None
    latency_ratio: float | None
    accepted: bool
    reason: str


def select_diverse_population(
    candidates: tuple[BehaviorCheckpoint, ...],
    *,
    limit: int,
) -> tuple[BehaviorCheckpoint, ...]:
    """Greedy quality-diversity retention with deterministic tie breaks."""
    if limit <= 0:
        raise ValueError("population limit must be positive")
    if not candidates:
        return ()
    remaining = sorted(candidates, key=lambda item: (-item.rating, item.policy_id))
    selected = [remaining.pop(0)]
    while remaining and len(selected) < limit:
        choice = max(
            remaining,
            key=lambda item: (
                min(_distance(item.descriptor, kept.descriptor) for kept in selected),
                item.rating,
                item.policy_id,
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)


def decide_consolidation(
    population: tuple[BehaviorCheckpoint, ...],
    *,
    specialist_score: float,
    best_single_score: float,
    lower_confidence_delta: float,
    minimum_delta: float = 0.0,
) -> ConsolidationDecision:
    """Permit distillation only after a confidence-resolved population win."""
    if len(population) < 2:
        return ConsolidationDecision(
            tuple(item.policy_id for item in population),
            specialist_score,
            best_single_score,
            lower_confidence_delta,
            False,
            "population_requires_multiple_specialists",
        )
    if specialist_score <= best_single_score or lower_confidence_delta < minimum_delta:
        return ConsolidationDecision(
            tuple(item.policy_id for item in population),
            specialist_score,
            best_single_score,
            lower_confidence_delta,
            False,
            "specialist_advantage_not_resolved",
        )
    return ConsolidationDecision(
        tuple(item.policy_id for item in population),
        specialist_score,
        best_single_score,
        lower_confidence_delta,
        True,
        "positive_population_result",
    )


def verify_distillation(
    decision: ConsolidationDecision,
    *,
    teacher: PairedEstimate | None = None,
    student: PairedEstimate | None = None,
    teacher_latency_ms: float | None = None,
    student_latency_ms: float | None = None,
    maximum_latency_ratio: float = 1.0,
) -> DistillationVerification:
    """Reject before training unless specialists won; otherwise check outcome and latency."""
    if not decision.distill:
        return DistillationVerification(False, None, None, False, decision.reason)
    if (
        teacher is None
        or student is None
        or teacher_latency_ms is None
        or student_latency_ms is None
    ):
        raise ValueError("positive distillation requires teacher/student evidence and latency")
    if teacher_latency_ms <= 0.0 or student_latency_ms <= 0.0:
        raise ValueError("latencies must be positive")
    latency_ratio = student_latency_ms / teacher_latency_ms
    preserved = student.lower_confidence >= teacher.lower_confidence
    accepted = preserved and latency_ratio <= maximum_latency_ratio
    return DistillationVerification(
        True,
        preserved,
        latency_ratio,
        accepted,
        "accepted"
        if accepted
        else "terminal_outcomes_regressed"
        if not preserved
        else "latency_gate_failed",
    )


def _distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.dist(first, second)
