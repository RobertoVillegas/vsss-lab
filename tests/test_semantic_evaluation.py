from __future__ import annotations

import json
from pathlib import Path

from vsss_train.marl import ParametricPrimitiveRoleActor
from vsss_train.semantic_evaluation import evaluate_semantic_skills
from vsss_train.semantic_scenarios import (
    SemanticScenario,
    SkillDifficulty,
    SkillScenarioParameters,
    compile_skill_scenario,
)

ROOT = Path(__file__).parents[1]
CONFIG_TEXT = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE_TEXT = (ROOT / "tests/golden/m1_match_state.json").read_text()
CONFIG = json.loads(CONFIG_TEXT)
STATE = json.loads(STATE_TEXT)


def _scenarios() -> tuple[SemanticScenario, ...]:
    return tuple(
        compile_skill_scenario(
            SkillScenarioParameters(
                schema_version=1,
                family=family,  # type: ignore[arg-type]
                seed=seed,
                controlled_team=team,  # type: ignore[arg-type]
                difficulty=SkillDifficulty(0.2, 0.2, 0.2, 0.2, 0.2),
                horizon=20,
                holdout=True,
            ),
            STATE,
            CONFIG,
        )
        for family in ("approach", "interception")
        for team in ("blue", "yellow")
        for seed in (101, 103)
    )


def test_paired_semantic_evaluation_reports_trials_intervals_and_throughput() -> None:
    scenarios = _scenarios()
    report = evaluate_semantic_skills(
        None,
        scenarios,
        CONFIG_TEXT,
        STATE_TEXT,
        control="random",
    )
    assert report.schema_version == 1
    assert report.attempts == 8
    assert report.elapsed_seconds > 0.0
    assert report.physical_validity_rate == 1.0
    assert report.mean_controlled_touches >= 0.0
    assert 0.0 <= report.idle_spin_ratio <= 1.0
    assert report.difficulty_bands["beginner"]["attempts"] == 8
    assert report.difficulty_levels["0.20"]["attempts"] == 8
    assert {trial.controlled_team for trial in report.trials} == {"blue", "yellow"}
    assert {family.family for family in report.families} == {"approach", "interception"}
    assert all(family.attempts == 4 for family in report.families)
    assert all(
        0.0 <= family.confidence_low <= family.confidence_high <= 1.0 for family in report.families
    )
    assert {trial.status for trial in report.trials} <= {
        "success",
        "failure",
        "unresolved",
    }


def test_parametric_primitive_evaluation_accepts_wide_policy_and_baseline_actions() -> None:
    scenarios = _scenarios()
    actor = ParametricPrimitiveRoleActor(32)
    for control, evaluated in (
        ("policy", actor),
        ("random", None),
        ("heuristic", None),
    ):
        report = evaluate_semantic_skills(
            evaluated,
            scenarios,
            CONFIG_TEXT,
            STATE_TEXT,
            control=control,  # type: ignore[arg-type]
            action_parser="parametric_primitive",
        )
        assert report.attempts == 8
        assert 0.0 <= report.idle_spin_ratio <= 1.0
        assert {trial.status for trial in report.trials} <= {
            "success",
            "failure",
            "unresolved",
        }


def test_clearance_difficulty_moves_the_distance_it_scores() -> None:
    """The easy end of a drill must be easier at the thing the drill is scored on.

    Clearance succeeds when the ball leaves the defensive third, so its demand is the
    ball's depth. That depth used to be a coin flip between two deep positions, which left
    every difficulty level asking for the same displacement and made the family unlearnable
    from below: the difficulty curriculum could prioritize it forever without ever making
    it winnable.
    """
    depths = []
    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        scenario = compile_skill_scenario(
            SkillScenarioParameters(
                schema_version=1,
                family="clearance",
                seed=101,
                controlled_team="blue",
                difficulty=SkillDifficulty(level, level, level, level, level),
                horizon=240,
                holdout=True,
            ),
            STATE,
            CONFIG,
        )
        ball = scenario.scenario.state["ball"]
        assert isinstance(ball, dict)
        depths.append(abs(float(ball["x"])))

    assert depths == sorted(depths)
    # The easy end must ask for materially less displacement than the hard end.
    assert depths[0] < depths[-1] - 0.2
    # And it must still be a clearance: the ball starts inside the defensive third.
    assert depths[0] > 0.10
