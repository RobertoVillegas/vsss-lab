"""Bounded behavioral league retention and distillation gates."""

from __future__ import annotations

import math
from dataclasses import dataclass


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


def _distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.dist(first, second)
