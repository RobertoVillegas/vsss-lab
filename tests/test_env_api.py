"""M3 native binding and standard environment contract tests."""

from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test
from vsss_env import ParallelVSSSEnv, SingleRobotEnv, TeamEnv
from vsss_env._native import BatchSimulator

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def test_native_batch_is_contiguous_and_replayable() -> None:
    simulator = BatchSimulator(CONFIG, STATE, 2)
    initial = simulator.reset()
    assert initial.shape == (2, BatchSimulator.state_width())
    assert initial.flags.c_contiguous
    snapshot = simulator.snapshots()[0]
    actions = np.zeros((2, 6, 2), dtype=np.float32)
    expected = simulator.step(actions)[0].copy()
    simulator.restore(0, snapshot)
    actual = simulator.step(actions)[0]
    np.testing.assert_array_equal(actual, expected)


def test_pettingzoo_parallel_contract() -> None:
    parallel_api_test(ParallelVSSSEnv(CONFIG, STATE), num_cycles=25)


def test_gymnasium_contracts() -> None:
    check_env(TeamEnv(CONFIG, STATE), skip_render_check=True)
    check_env(SingleRobotEnv(CONFIG, STATE), skip_render_check=True)
