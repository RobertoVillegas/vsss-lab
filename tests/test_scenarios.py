from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest
from vsss_league.training import create_rollout_session, train_iteration
from vsss_train.config import MarlConfig
from vsss_train.marl_ppo import MarlLearner
from vsss_train.scenarios import (
    BucketProgress,
    Scenario,
    ScenarioCurriculum,
    allocate_scenarios,
    load_suite,
    mutate_scenario,
    validate_scenario,
    write_suite,
)

ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "tests/golden/m1_match_config.json").read_text())
STATE = json.loads((ROOT / "tests/golden/m1_match_state.json").read_text())


def scenario(**overrides: object) -> Scenario:
    values = {
        "scenario_id": "kickoff-001",
        "kind": "kickoff",
        "role": "routine",
        "state": copy.deepcopy(STATE),
    }
    values.update(overrides)
    return Scenario(**values)


def test_validation_rejects_ball_and_robot_overlap() -> None:
    invalid = scenario()
    state = invalid.state
    state["ball"]["x"] = state["robots"][0]["pose"]["x"]  # type: ignore[index]
    state["ball"]["y"] = state["robots"][0]["pose"]["y"]  # type: ignore[index]
    with pytest.raises(ValueError, match="overlaps ball"):
        validate_scenario(invalid, CONFIG)


def test_curriculum_retains_mixture_and_prioritizes_progress() -> None:
    allocation = allocate_scenarios(
        (
            BucketProgress("kickoff", 0.55, 0.30, 10),
            BucketProgress("defense", 0.40, 0.38, 10),
            BucketProgress("shot", 0.99, 0.50, 10),
        ),
        batch_size=100,
    )
    assert (allocation.routine, allocation.frontier, allocation.failure, allocation.holdout) == (
        20,
        50,
        20,
        10,
    )
    assert allocation.frontier_kinds == ("kickoff", "defense")


def test_mutation_is_reproducible_valid_and_protects_holdout() -> None:
    first = mutate_scenario(scenario(), CONFIG, seed=7)
    second = mutate_scenario(scenario(), CONFIG, seed=7)
    assert first.digest == second.digest
    validate_scenario(first, CONFIG)
    with pytest.raises(ValueError, match="immutable"):
        mutate_scenario(scenario(role="holdout", immutable=True), CONFIG, seed=7)


def test_suite_rejects_duplicate_states_and_writes_atomically(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        write_suite(tmp_path / "suite.json", (scenario(), scenario(scenario_id="duplicate")))
    output = tmp_path / "suite.json"
    write_suite(output, (scenario(),))
    assert json.loads(output.read_text())["scenarios"][0]["role"] == "routine"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_curriculum_deduplicates_failures_and_never_trains_on_holdout() -> None:
    routine = scenario(scenario_id="routine")
    frontier = scenario(scenario_id="frontier", role="frontier")
    holdout = scenario(scenario_id="holdout", role="holdout", immutable=True)
    teacher = ScenarioCurriculum((routine, frontier, holdout), CONFIG, seed=14)
    teacher.record(frontier, success=False)
    teacher.record(frontier, success=False)
    selections = [teacher.select_training(index) for index in range(90)]
    assert all(selection.scenario.role != "holdout" for selection in selections)
    assert {selection.source for selection in selections} == {"routine", "frontier", "failure"}
    assert teacher.failure_count == 1
    telemetry = teacher.telemetry()
    assert telemetry["mixture"] == {
        "routine": 20,
        "frontier": 50,
        "failure": 20,
        "holdout": 1,
    }
    teacher.record(frontier, success=True)
    assert teacher.failure_count == 0


def test_replay_failure_descriptor_is_deduplicated_and_never_changes_reward() -> None:
    routine = scenario(scenario_id="routine")
    frontier = scenario(scenario_id="frontier", kind="defense", role="frontier")
    holdout = scenario(scenario_id="holdout", role="holdout", immutable=True)
    teacher = ScenarioCurriculum((routine, frontier, holdout), CONFIG, seed=14)
    assert teacher.ingest_failure_descriptor(kind="defense", digest="failure-1")
    assert not teacher.ingest_failure_descriptor(kind="defense", digest="failure-1")
    assert teacher.failure_count == 1
    assert teacher.telemetry()["replay_failure_descriptors"] == 1


def test_committed_m14_suite_is_valid_and_covers_all_skill_kinds() -> None:
    suite = load_suite(ROOT / "experiments/scenarios/m14-v1.json", CONFIG)
    assert len(suite) == 9
    assert {item.kind for item in suite} == {
        "kickoff",
        "approach",
        "interception",
        "clearance",
        "defense",
        "pass_receive",
        "shot",
        "congestion_recovery",
        "mixed",
    }


def test_adaptive_curriculum_drives_real_vector_rollout_and_telemetry() -> None:
    config = MarlConfig(
        device="cpu",
        adaptive_curriculum=True,
        scenario_suite=str(ROOT / "experiments/scenarios/m14-v1.json"),
        num_envs=2,
        hidden_size=8,
        rollout_steps=2,
        horizon=1,
        action_repeat=1,
        epochs=1,
        minibatch_size=6,
    )
    learner = MarlLearner(config)
    session = create_rollout_session(config, json.dumps(CONFIG), json.dumps(STATE))
    result = train_iteration(
        learner,
        None,
        json.dumps(CONFIG),
        json.dumps(STATE),
        iteration=1,
        seed=14,
        opponent_id="heuristic",
        checkpoint=None,
        session=session,
    )
    assert result.curriculum is not None
    mixture = cast(dict[str, int], result.curriculum["mixture"])
    assert sum(mixture[source] for source in ("routine", "frontier", "failure")) >= 2
    assert mixture["holdout"] == 2
