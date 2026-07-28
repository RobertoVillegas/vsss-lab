from __future__ import annotations

import json
from pathlib import Path

from vsss_train.semantic_evaluation import evaluate_semantic_skills
from vsss_train.semantic_scenarios import (
    SkillDifficulty,
    SkillScenarioParameters,
    compile_skill_scenario,
)

ROOT = Path(__file__).parents[1]
CONFIG_TEXT = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE_TEXT = (ROOT / "tests/golden/m1_match_state.json").read_text()
CONFIG = json.loads(CONFIG_TEXT)
STATE = json.loads(STATE_TEXT)


def test_paired_semantic_evaluation_reports_trials_intervals_and_throughput() -> None:
    scenarios = tuple(
        compile_skill_scenario(
            SkillScenarioParameters(
                schema_version=1,
                family=family,  # type: ignore[arg-type]
                seed=seed,
                controlled_team=team,  # type: ignore[arg-type]
                difficulty=SkillDifficulty(0.2, 0.2, 0.2, 0.2, 0.2),
                horizon=2,
                holdout=True,
            ),
            STATE,
            CONFIG,
        )
        for family in ("approach", "interception")
        for team in ("blue", "yellow")
        for seed in (101, 103)
    )
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
    assert report.difficulty_bands["beginner"]["attempts"] == 8
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
