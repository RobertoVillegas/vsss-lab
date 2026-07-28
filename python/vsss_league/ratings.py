"""Small auditable Elo implementation."""

from __future__ import annotations

from dataclasses import dataclass

from vsss_league.evaluation import PairedEstimate, PairedMatch, paired_estimate


@dataclass(frozen=True)
class EloRating:
    first: float
    second: float


@dataclass(frozen=True)
class PairedEloReport:
    """Color-balanced historical rating plus uncertainty on terminal score."""

    candidate: str
    opponent: str
    candidate_rating: float
    opponent_rating: float
    estimate: PairedEstimate
    updates: int


def expected_score(first: float, second: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((second - first) / 400.0)))


def elo_update(
    first: float,
    second: float,
    *,
    first_score: float,
    k_factor: float = 32.0,
) -> EloRating:
    if first_score not in (0.0, 0.5, 1.0):
        raise ValueError("first_score must be 0, 0.5, or 1")
    delta = k_factor * (first_score - expected_score(first, second))
    return EloRating(first + delta, second - delta)


def paired_elo_report(
    candidate: str,
    opponent: str,
    matches: tuple[PairedMatch, ...],
    *,
    initial_rating: float = 1_000.0,
    k_factor: float = 32.0,
    bootstrap_samples: int = 10_000,
) -> PairedEloReport:
    """Rate both colors in stable seed order and retain a paired confidence interval."""
    estimate = paired_estimate(matches, bootstrap_samples=bootstrap_samples)
    candidate_rating = initial_rating
    opponent_rating = initial_rating
    updates = 0
    for match in sorted(matches, key=lambda item: (item.seed, item.scenario)):
        for outcome in (match.candidate_blue, match.candidate_yellow):
            rating = elo_update(
                candidate_rating,
                opponent_rating,
                first_score=(float(outcome) + 1.0) / 2.0,
                k_factor=k_factor,
            )
            candidate_rating, opponent_rating = rating.first, rating.second
            updates += 1
    return PairedEloReport(
        candidate=candidate,
        opponent=opponent,
        candidate_rating=candidate_rating,
        opponent_rating=opponent_rating,
        estimate=estimate,
        updates=updates,
    )
