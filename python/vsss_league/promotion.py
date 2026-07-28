"""Deterministic promotion manifests and non-regression decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FixtureResult:
    opponent: str
    category: str
    margin: float
    regression_floor: float
    seeds: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return self.margin >= self.regression_floor


@dataclass(frozen=True)
class PromotionDecision:
    schema_version: int
    candidate: str
    current_main: str
    identity_gate: bool
    aggregate_margin: float
    required_margin: float
    fixtures: tuple[FixtureResult, ...]
    promoted: bool

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def decide_promotion(
    *,
    candidate: str,
    current_main: str,
    identity_gate: bool,
    fixtures: tuple[FixtureResult, ...],
    required_margin: float,
) -> PromotionDecision:
    if not fixtures:
        raise ValueError("promotion requires fixtures")
    required_categories = {"main", "historical", "heuristic"}
    categories = {fixture.category for fixture in fixtures}
    if not required_categories <= categories:
        raise ValueError("promotion requires main, historical, and heuristic fixtures")
    aggregate_margin = sum(fixture.margin for fixture in fixtures) / len(fixtures)
    promoted = (
        identity_gate
        and aggregate_margin >= required_margin
        and all(fixture.passed for fixture in fixtures)
    )
    return PromotionDecision(
        schema_version=1,
        candidate=candidate,
        current_main=current_main,
        identity_gate=identity_gate,
        aggregate_margin=aggregate_margin,
        required_margin=required_margin,
        fixtures=fixtures,
        promoted=promoted,
    )
