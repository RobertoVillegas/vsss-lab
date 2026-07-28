import sys
from pathlib import Path

import numpy as np
from vsss_env.backends import JsonLineBackend, NativeBackend, run_policy

ROOT = Path(__file__).resolve().parents[1]


def test_same_policy_uses_native_and_process_bridge_without_api_change() -> None:
    config = (ROOT / "tests/golden/m1_match_config.json").read_text()
    state = (ROOT / "tests/golden/m1_match_state.json").read_text()
    native = NativeBackend(config, state)
    with JsonLineBackend(
        [sys.executable, "-m", "tools.canonical_backend_sidecar"],
        config,
        state,
    ) as bridged:
        native_states = run_policy(native, 8)
        bridged_states = run_policy(bridged, 8)
    assert len(native_states) == len(bridged_states)
    for expected, actual in zip(native_states, bridged_states, strict=True):
        np.testing.assert_array_equal(actual, expected)
