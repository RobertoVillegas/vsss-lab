"""Small auditable Elo implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloRating:
    first: float
    second: float


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
