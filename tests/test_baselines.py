"""M4 scripted controller and match regression tests."""

from pathlib import Path

import numpy as np
from vsss_baselines import DynamicTeamController, go_to_target
from vsss_env._native import BatchSimulator
from vsss_eval import inspect_replay, run_scripted_match

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def initial_row() -> np.ndarray:
    """Return the canonical M3 kickoff tensor."""
    return np.asarray(BatchSimulator(CONFIG, STATE, 1).reset()[0], dtype=np.float32)


def test_target_ahead_drives_straight_and_actions_are_bounded() -> None:
    action = go_to_target((0.0, 0.0, 0.0), (0.5, 0.0))
    assert action[0] == action[1] > 0.0
    assert np.all(np.abs(action) <= 1.0)


def test_dynamic_assignment_ignores_physical_ids() -> None:
    state = initial_row()
    controller = DynamicTeamController(0, 1)
    expected_roles = controller.assign(state)
    expected_actions = controller.actions(state)
    renamed = state.copy()
    renamed[[10, 21, 32]] = renamed[[32, 10, 21]]
    assert controller.assign(renamed) == expected_roles
    np.testing.assert_array_equal(controller.actions(renamed), expected_actions)


def test_scripted_match_and_replay_are_byte_reproducible(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = run_scripted_match(CONFIG, STATE, 120, first_path, seed=7)
    second = run_scripted_match(CONFIG, STATE, 120, second_path, seed=7)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    inspected = inspect_replay(first_path)
    assert inspected["ticks"] == 120
    assert inspected["final_checksum"] == first.final_checksum
