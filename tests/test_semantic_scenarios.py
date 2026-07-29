from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

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
    "rotation_recovery",
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
    for family in (
        "interception",
        "save_deflection",
        "clearance",
        "shot",
        "pass_receive",
        "rotation_recovery",
    ):
        assert compiled[family].context.initial_ball_speed > 0.0
    assert compiled["interception"].context.initial_threat
    assert compiled["save_deflection"].context.initial_threat
    assert compiled["clearance"].context.initial_threat


def test_difficulty_rejects_out_of_range_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="ball_speed"):
        SkillDifficulty(ball_speed=1.01)
    with pytest.raises(ValueError, match="ball_angle"):
        SkillDifficulty(ball_angle=float("nan"))


def test_compiler_rejects_a_skill_horizon_that_is_physically_unreachable() -> None:
    with pytest.raises(ValueError, match="cannot reach"):
        compile_skill_scenario(
            SkillScenarioParameters(
                schema_version=1,
                family="approach",
                seed=9,
                controlled_team="blue",
                difficulty=SkillDifficulty(spawn_distance=1.0),
                horizon=1,
            ),
            STATE,
            CONFIG,
        )


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
    assert {scenario.parameters.roster for scenario in compiled} >= {
        "1v0",
        "1v1",
        "2v1",
        "2v2",
        "3v2",
        "3v3",
    }


def test_phased_curriculum_focuses_then_advances_after_consecutive_holdouts() -> None:
    curriculum = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=41,
        full_match_fraction=0.20,
        phased=True,
        phase_patience=2,
    )
    selections = [curriculum.select_training(index) for index in range(80)]
    families = {
        selection.scenario.parameters.family
        for selection in selections
        if selection.scenario is not None
    }
    assert curriculum.phase_name == "foundation"
    assert families == {"approach", "shot", "interception"}
    passing = {"approach": 0.8, "shot": 0.75, "interception": 0.4}
    assert not curriculum.observe_holdout_rates(passing)
    assert curriculum.phase_name == "foundation"
    assert curriculum.observe_holdout_rates(passing)
    assert curriculum.phase_name == "defense"
    assert curriculum.training_families == (
        "interception",
        "save_deflection",
        "clearance",
    )


def test_phased_curriculum_resets_promotion_streak_and_persists_phase() -> None:
    curriculum = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=42,
        phased=True,
        phase_patience=2,
    )
    passing = {"approach": 0.8, "shot": 0.75, "interception": 0.4}
    curriculum.observe_holdout_rates(passing)
    curriculum.observe_holdout_rates({"approach": 0.7, "shot": 0.75, "interception": 0.4})
    assert curriculum.phase_gate_streak == 0
    curriculum.observe_holdout_rates(passing)
    assert curriculum.observe_holdout_rates(passing)
    restored = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=42,
        phased=True,
        phase_patience=2,
    )
    restored.load_state_dict(curriculum.state_dict())
    assert restored.phase_name == "defense"
    assert restored.phase_gate_streak == 0


@pytest.mark.parametrize(
    ("family", "roster", "controlled_count", "opponent_count"),
    (
        ("approach", "1v0", 1, 0),
        ("shot", "1v1", 1, 1),
        ("pass_receive", "2v1", 2, 1),
        ("clearance", "2v2", 2, 2),
        ("rotation_recovery", "3v2", 3, 2),
        ("rotation_recovery", "3v3", 3, 3),
    ),
)
def test_roster_ladder_enables_only_required_participants(
    family: str,
    roster: str,
    controlled_count: int,
    opponent_count: int,
) -> None:
    scenario = compile_skill_scenario(
        SkillScenarioParameters(
            schema_version=1,
            family=family,  # type: ignore[arg-type]
            seed=91,
            controlled_team="blue",
            difficulty=SkillDifficulty(0.2, 0.2, 0.2, 0.2, 0.2),
            roster=roster,  # type: ignore[arg-type]
        ),
        STATE,
        CONFIG,
    )
    robots = cast(list[dict[str, Any]], scenario.scenario.state["robots"])
    blue = [robot for robot in robots if robot["team"] == "blue"]
    yellow = [robot for robot in robots if robot["team"] == "yellow"]
    assert sum(bool(robot["enabled"]) for robot in blue) == controlled_count
    assert sum(bool(robot["enabled"]) for robot in yellow) == opponent_count


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
    levels = curriculum.levels[scenario.parameters.family]
    assert max(levels.values()) > 0.05
    assert curriculum.telemetry()["failure_count"] == 0


def test_curriculum_prioritizes_weak_skills_and_rehearses_mastered_skills() -> None:
    curriculum = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=31,
        full_match_fraction=0.0,
        window=4,
    )
    for family in SKILL_FAMILIES:
        scenario = compile_skill_scenario(parameters(family, 71), STATE, CONFIG)
        for _ in range(8):
            curriculum.record(scenario, success=family != "rotation_recovery")
    for index in range(400):
        curriculum.select_training(index)
    allocation = curriculum.telemetry()["allocation_by_family"]
    assert isinstance(allocation, dict)
    assert allocation["rotation_recovery"] > max(
        allocation[family] for family in SKILL_FAMILIES if family != "rotation_recovery"
    )
    assert all(allocation[family] > 0 for family in SKILL_FAMILIES)


def test_curriculum_never_reduces_a_difficulty_axis_below_training_floor() -> None:
    curriculum = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=32,
        full_match_fraction=0.0,
        window=4,
    )
    scenario = compile_skill_scenario(parameters("rotation_recovery", 73), STATE, CONFIG)
    for _ in range(40):
        curriculum.record(scenario, success=False)
    assert min(curriculum.levels["rotation_recovery"].values()) == pytest.approx(0.05)


def test_curriculum_can_reserve_full_matches_outside_skill_gradients() -> None:
    curriculum = SemanticSkillCurriculum(STATE, CONFIG, seed=17, full_match_fraction=1.0)
    selection = curriculum.select_training(0)
    assert selection.scenario is None
    assert selection.source == "full_match"


def test_curriculum_enforces_observed_full_match_floor() -> None:
    curriculum = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=18,
        full_match_fraction=0.25,
    )
    for index in range(40):
        curriculum.select_training(index)
        telemetry = curriculum.telemetry()
        assert telemetry["allocation_valid"] is True
        observed = telemetry["observed_full_match_fraction"]
        assert isinstance(observed, float)
        assert observed >= 0.25


def test_holdouts_are_paired_immutable_and_excluded_from_training_feedback() -> None:
    curriculum = SemanticSkillCurriculum(STATE, CONFIG, seed=19)
    holdouts = curriculum.holdouts(seeds=(101,))
    assert len(holdouts) == len(SKILL_FAMILIES) * 2 * 4
    assert {scenario.parameters.controlled_team for scenario in holdouts} == {
        "blue",
        "yellow",
    }
    assert {scenario.parameters.difficulty.ball_speed for scenario in holdouts} == {
        0.10,
        0.25,
        0.40,
        0.65,
    }
    assert all(
        scenario.parameters.holdout
        and scenario.scenario.immutable
        and scenario.scenario.role == "holdout"
        for scenario in holdouts
    )
    with pytest.raises(ValueError, match="holdouts"):
        curriculum.record(holdouts[0], success=True)


def test_curriculum_state_round_trip_preserves_difficulty_history_and_failures() -> None:
    original = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=23,
        full_match_fraction=0.0,
        window=4,
    )
    selection = original.select_training(0)
    assert selection.scenario is not None
    original.record(selection.scenario, success=False)
    for _ in range(8):
        original.record(selection.scenario, success=True)
    restored = SemanticSkillCurriculum(
        STATE,
        CONFIG,
        seed=23,
        full_match_fraction=0.0,
        window=4,
    )
    restored.load_state_dict(original.state_dict())
    assert restored.state_dict() == original.state_dict()
