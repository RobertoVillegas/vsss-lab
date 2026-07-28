"""Deterministic promotion manifests and non-regression decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vsss_league.evaluation import PairedEstimate


@dataclass(frozen=True)
class FixtureResult:
    opponent: str
    category: str
    margin: float
    regression_floor: float
    seeds: tuple[int, ...]
    lower_confidence: float | None = None
    games: int | None = None

    @property
    def passed(self) -> bool:
        evidence = self.margin if self.lower_confidence is None else self.lower_confidence
        return evidence >= self.regression_floor

    @classmethod
    def from_estimate(
        cls,
        *,
        opponent: str,
        category: str,
        estimate: PairedEstimate,
        regression_floor: float,
    ) -> FixtureResult:
        return cls(
            opponent=opponent,
            category=category,
            margin=estimate.mean_score - 0.5,
            regression_floor=regression_floor,
            seeds=estimate.seeds,
            lower_confidence=estimate.lower_confidence - 0.5,
            games=estimate.games,
        )


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
    rejection_reasons: tuple[str, ...]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def write(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(self.canonical_json() + "\n", encoding="utf-8")
        temporary.replace(output)


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
    rejection_reasons = tuple(
        reason
        for condition, reason in (
            (not identity_gate, "identity_gate_failed"),
            (aggregate_margin < required_margin, "aggregate_margin_below_requirement"),
            (not all(fixture.passed for fixture in fixtures), "fixture_regression"),
        )
        if condition
    )
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
        rejection_reasons=rejection_reasons,
    )
