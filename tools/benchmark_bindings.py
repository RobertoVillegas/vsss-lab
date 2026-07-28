"""Measure M3 Python-to-Rust batch-call overhead."""

import json
import time
from pathlib import Path

import numpy as np
from vsss_env._native import BatchSimulator

root = Path(__file__).parents[1]
config = (root / "tests/golden/m1_match_config.json").read_text()
state = (root / "tests/golden/m1_match_state.json").read_text()
worlds = 64
calls = 2_000
simulator = BatchSimulator(config, state, worlds)
actions = np.zeros((worlds, 6, 2), dtype=np.float32)

for _ in range(20):
    simulator.step(actions)
start = time.perf_counter()
for _ in range(calls):
    simulator.step(actions)
seconds = time.perf_counter() - start
print(
    json.dumps(
        {
            "calls": calls,
            "microseconds_per_call": seconds * 1e6 / calls,
            "seconds": seconds,
            "world_steps_per_second": calls * worlds / seconds,
            "worlds": worlds,
        },
        sort_keys=True,
    )
)
