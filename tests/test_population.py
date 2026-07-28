from __future__ import annotations

from vsss_league.population import (
    BehaviorCheckpoint,
    decide_consolidation,
    select_diverse_population,
    verify_distillation,
)
from vsss_league.evaluation import PairedMatch, paired_estimate
from vsss_league.ratings import paired_elo_report


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


def test_paired_elo_reports_both_colors_and_confidence() -> None:
    matches = (
        PairedMatch(1, "routine", 1, 0),
        PairedMatch(2, "frontier", 1, 1),
        PairedMatch(3, "failure", 0, 1),
    )
    report = paired_elo_report("specialists", "single", matches, bootstrap_samples=500)
    assert report.updates == 6
    assert report.estimate.games == 6
    assert report.estimate.lower_confidence <= report.estimate.mean_score
    assert report.candidate_rating > report.opponent_rating


def test_negative_population_result_skips_distillation_entirely() -> None:
    population = (
        checkpoint("attack", 1100, (0.8, 0.8, 0.2, 0.1)),
        checkpoint("defense", 1100, (0.2, 0.2, 0.1, 0.1)),
    )
    decision = decide_consolidation(
        population,
        specialist_score=0.5,
        best_single_score=0.5,
        lower_confidence_delta=-0.25,
    )
    verification = verify_distillation(decision)
    assert not verification.attempted
    assert not verification.accepted


def test_positive_distillation_must_preserve_terminal_interval_and_latency() -> None:
    population = (
        checkpoint("attack", 1100, (0.8, 0.8, 0.2, 0.1)),
        checkpoint("defense", 1100, (0.2, 0.2, 0.1, 0.1)),
    )
    decision = decide_consolidation(
        population,
        specialist_score=0.75,
        best_single_score=0.5,
        lower_confidence_delta=0.1,
    )
    teacher = paired_estimate(
        (
            PairedMatch(1, "suite", 1, 0),
            PairedMatch(2, "suite", 1, 1),
        ),
        bootstrap_samples=500,
    )
    student = paired_estimate(
        (
            PairedMatch(3, "suite", 1, 0),
            PairedMatch(4, "suite", 1, 1),
        ),
        bootstrap_samples=500,
    )
    verification = verify_distillation(
        decision,
        teacher=teacher,
        student=student,
        teacher_latency_ms=0.4,
        student_latency_ms=0.2,
    )
    assert verification.accepted
