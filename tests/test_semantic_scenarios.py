from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from vsss_train.semantic_scenarios import (
    SKILL_FAMILIES,
    SemanticSkillCurriculum,
    SkillDifficulty,
    SkillScenarioParameters,
    compile_skill_scenario,
)

ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "tests/golden/m1_match_config.json").read_text())
STATE = json.loads((ROOT / "tests/golden/m1_match_state.json").read_text())
FAMILIES = (
    "approach",
    "interception",
    "save_deflection",
    "clearance",
    "shot",
    "pass_receive",
)


def parameters(family: str, seed: int, team: str = "blue") -> SkillScenarioParameters:
    return SkillScenarioParameters(
        schema_version=1,
        family=family,  # type: ignore[arg-type]
        seed=seed,
        controlled_team=team,  # type: ignore[arg-type]
        difficulty=SkillDifficulty(0.6, 0.5, 0.4, 0.3, 0.2),
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_semantic_compilation_is_deterministic_valid_and_seeded(family: str) -> None:
    first = compile_skill_scenario(parameters(family, 14), STATE, CONFIG)
    repeated = compile_skill_scenario(parameters(family, 14), STATE, CONFIG)
    different = compile_skill_scenario(parameters(family, 15), STATE, CONFIG)
    assert first.parameters.digest == repeated.parameters.digest
    assert first.scenario.digest == repeated.scenario.digest
    assert first.context.state_hash == first.scenario.digest
    assert first.scenario.digest != different.scenario.digest
    assert first.context.controlled_robot_id != ""
    assert math.isfinite(first.context.initial_ball_speed)


@pytest.mark.parametrize("family", FAMILIES)
def test_semantic_compilation_mirrors_ball_and_team_geometry(family: str) -> None:
    blue = compile_skill_scenario(parameters(family, 17, "blue"), STATE, CONFIG)
    yellow = compile_skill_scenario(parameters(family, 17, "yellow"), STATE, CONFIG)
    for key in ("x", "y", "vx", "vy"):
        assert yellow.scenario.state["ball"][key] == pytest.approx(  # type: ignore[index]
            -blue.scenario.state["ball"][key]  # type: ignore[index]
        )
    assert yellow.context.target_goal_x == -blue.context.target_goal_x
    assert yellow.context.own_goal_x == -blue.context.own_goal_x
    assert yellow.context.controlled_team == "yellow"


def test_all_families_cover_moving_and_static_ball_semantics() -> None:
    compiled = {
        family: compile_skill_scenario(parameters(family, 21), STATE, CONFIG) for family in FAMILIES
    }
    assert compiled["approach"].context.initial_ball_speed == 0.0
    for family in ("interception", "save_deflection", "clearance", "shot", "pass_receive"):
        assert compiled[family].context.initial_ball_speed > 0.0
    assert compiled["interception"].context.initial_threat
    assert compiled["save_deflection"].context.initial_threat
    assert compiled["clearance"].context.initial_threat


def test_difficulty_rejects_out_of_range_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="ball_speed"):
        SkillDifficulty(ball_speed=1.01)
    with pytest.raises(ValueError, match="ball_angle"):
        SkillDifficulty(ball_angle=float("nan"))


def test_compiler_remains_valid_across_families_colors_and_many_seeds() -> None:
    digests = set()
    scenario_ids = set()
    for family in FAMILIES:
        for team in ("blue", "yellow"):
            for seed in range(40):
                compiled = compile_skill_scenario(parameters(family, seed, team), STATE, CONFIG)
                digests.add(compiled.scenario.digest)
                scenario_ids.add(compiled.scenario.scenario_id)
    expected = len(FAMILIES) * 2 * 40
    assert len(scenario_ids) == expected
    # Some 180-degree mirrors are physically identical, but seeds still provide
    # broad state diversity without relying on an identifier in canonical state.
    assert len(digests) >= int(expected * 0.8)


def test_curriculum_mirrors_colors_and_allocates_every_family() -> None:
    curriculum = SemanticSkillCurriculum(STATE, CONFIG, seed=11, full_match_fraction=0.0)
    selections = [curriculum.select_training(index) for index in range(48)]
    scenarios = [selection.scenario for selection in selections]
    assert all(scenario is not None for scenario in scenarios)
    compiled = [scenario for scenario in scenarios if scenario is not None]
    assert {scenario.parameters.family for scenario in compiled} == set(SKILL_FAMILIES)
    assert {scenario.parameters.controlled_team for scenario in compiled} == {
        "blue",
        "yellow",
    }
    assert {selection.source for selection in selections} >= {"routine", "frontier"}


def test_curriculum_advances_mastered_family_and_deduplicates_failures() -> None:
    curriculum = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=13,
        full_match_fraction=0.0,
        window=4,
    )
    selection = curriculum.select_training(0)
    assert selection.scenario is not None
    scenario = selection.scenario
    curriculum.record(scenario, success=False)
    curriculum.record(scenario, success=False)
    assert curriculum.telemetry()["failure_count"] == 1
    for _ in range(8):
        curriculum.record(scenario, success=True)
    assert curriculum.levels[scenario.parameters.family] > 0.05
    assert curriculum.telemetry()["failure_count"] == 0


def test_curriculum_can_reserve_full_matches_outside_skill_gradients() -> None:
    curriculum = SemanticSkillCurriculum(STATE, CONFIG, seed=17, full_match_fraction=1.0)
    selection = curriculum.select_training(0)
    assert selection.scenario is None
    assert selection.source == "full_match"
