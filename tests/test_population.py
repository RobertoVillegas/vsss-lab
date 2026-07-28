from __future__ import annotations

from vsss_league.population import (
    BehaviorCheckpoint,
    decide_consolidation,
    select_diverse_population,
)


def checkpoint(
    policy_id: str,
    rating: float,
    descriptor: tuple[float, float, float, float],
) -> BehaviorCheckpoint:
    return BehaviorCheckpoint(policy_id, rating, 0.5, *descriptor)


def test_population_is_bounded_reproducible_and_behaviorally_diverse() -> None:
    candidates = (
        checkpoint("best", 1200.0, (0.5, 0.5, 0.2, 0.2)),
        checkpoint("clone", 1190.0, (0.51, 0.49, 0.2, 0.21)),
        checkpoint("defender", 1100.0, (0.2, 0.1, 0.1, 0.1)),
        checkpoint("attacker", 1090.0, (0.8, 0.9, 0.4, 0.3)),
    )
    selected = select_diverse_population(candidates, limit=3)
    assert [item.policy_id for item in selected] == ["best", "attacker", "defender"]
    assert selected == select_diverse_population(tuple(reversed(candidates)), limit=3)


def test_distillation_requires_resolved_specialist_advantage() -> None:
    population = (
        checkpoint("attack", 1100, (0.8, 0.8, 0.2, 0.1)),
        checkpoint("defense", 1100, (0.2, 0.2, 0.1, 0.1)),
    )
    unresolved = decide_consolidation(
        population,
        specialist_score=0.55,
        best_single_score=0.53,
        lower_confidence_delta=-0.01,
    )
    assert not unresolved.distill
    resolved = decide_consolidation(
        population,
        specialist_score=0.60,
        best_single_score=0.53,
        lower_confidence_delta=0.02,
    )
    assert resolved.distill
