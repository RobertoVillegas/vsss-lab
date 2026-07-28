from __future__ import annotations

import json
from pathlib import Path

import pytest
from vsss_league.evaluation import (
    PairedMatch,
    paired_estimate,
    write_evaluation_artifact,
)
from vsss_league.promotion import FixtureResult, decide_promotion


def matches(outcome: int, *, count: int = 8) -> tuple[PairedMatch, ...]:
    return tuple(
        PairedMatch(
            seed=100 + index,
            scenario=f"holdout-{index % 2}",
            candidate_blue=outcome,  # type: ignore[arg-type]
            candidate_yellow=outcome,  # type: ignore[arg-type]
        )
        for index in range(count)
    )


def test_paired_interval_is_deterministic_and_color_balanced() -> None:
    values = (
        PairedMatch(1, "kickoff", 1, -1),
        PairedMatch(2, "defense", 1, 0),
        PairedMatch(3, "kickoff", 0, 0),
    )
    first = paired_estimate(values, bootstrap_samples=500)
    second = paired_estimate(values, bootstrap_samples=500)
    assert first == second
    assert first.games == 6
    assert first.mean_score == pytest.approx((0.5 + 0.75 + 0.5) / 3)


def test_promotion_uses_lower_confidence_not_point_estimate() -> None:
    strong = paired_estimate(matches(1), bootstrap_samples=500)
    uncertain = paired_estimate(
        (
            PairedMatch(1, "holdout", 1, 1),
            PairedMatch(2, "holdout", -1, -1),
        ),
        bootstrap_samples=500,
    )
    fixtures = (
        FixtureResult.from_estimate(
            opponent="main@0", category="main", estimate=strong, regression_floor=0.0
        ),
        FixtureResult.from_estimate(
            opponent="historical@0",
            category="historical",
            estimate=strong,
            regression_floor=0.0,
        ),
        FixtureResult.from_estimate(
            opponent="heuristic",
            category="heuristic",
            estimate=uncertain,
            regression_floor=0.0,
        ),
    )
    decision = decide_promotion(
        candidate="candidate@1",
        current_main="main@0",
        identity_gate=True,
        fixtures=fixtures,
        required_margin=0.0,
    )
    assert not decision.promoted
    assert decision.rejection_reasons == ("fixture_regression",)


def test_evaluation_and_decision_artifacts_are_complete_and_atomic(tmp_path: Path) -> None:
    paired = matches(1)
    estimate = paired_estimate(paired, bootstrap_samples=500)
    evidence = tmp_path / "evaluation.json"
    write_evaluation_artifact(
        evidence,
        candidate="candidate@1",
        baseline="main@0",
        suite="immutable-holdout-v1",
        matches=paired,
        estimate=estimate,
    )
    payload = json.loads(evidence.read_text())
    assert payload["estimate"]["games"] == 16
    assert len(payload["matches"]) == 8
    assert not tuple(tmp_path.glob("*.tmp"))

    fixture = FixtureResult.from_estimate(
        opponent="main@0", category="main", estimate=estimate, regression_floor=0.0
    )
    decision = decide_promotion(
        candidate="candidate@1",
        current_main="main@0",
        identity_gate=True,
        fixtures=(
            fixture,
            FixtureResult.from_estimate(
                opponent="historical@0",
                category="historical",
                estimate=estimate,
                regression_floor=0.0,
            ),
            FixtureResult.from_estimate(
                opponent="heuristic",
                category="heuristic",
                estimate=estimate,
                regression_floor=0.0,
            ),
        ),
        required_margin=0.1,
    )
    output = tmp_path / "decision.json"
    decision.write(output)
    assert json.loads(output.read_text())["promoted"]
