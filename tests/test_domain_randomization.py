import json
from pathlib import Path

import numpy as np
from vsss_env.randomization import RandomizedBackend, evaluate_ood

ROOT = Path(__file__).resolve().parents[1]


def test_seed_reproduces_domain_and_trajectory() -> None:
    config = (ROOT / "tests/golden/m1_match_config.json").read_text()
    state = (ROOT / "tests/golden/m1_match_state.json").read_text()
    suite = json.loads((ROOT / "experiments/configs/m11-ood.json").read_text())
    first = RandomizedBackend(config, state, suite, 9)
    second = RandomizedBackend(config, state, suite, 9)
    np.testing.assert_array_equal(first.reset(), second.reset())
    actions = np.full((6, 2), 4.0, dtype=np.float32)
    for _ in range(12):
        np.testing.assert_array_equal(first.step(actions), second.step(actions))
    assert first.domain == second.domain


def test_robust_policy_beats_nominal_on_paired_ood_suite() -> None:
    report = evaluate_ood(
        ROOT / "tests/golden/m1_match_config.json",
        ROOT / "tests/golden/m1_match_state.json",
        ROOT / "experiments/configs/m11-ood.json",
    )
    assert report["passed"], report
