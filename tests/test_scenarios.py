from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from vsss_train.scenarios import (
    BucketProgress,
    Scenario,
    allocate_scenarios,
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
