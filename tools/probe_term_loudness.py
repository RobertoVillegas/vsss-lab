"""How loud is each reward term per step, which is what a policy gradient hears.

Comparing accumulated sums was the wrong instrument for a shaping term: with a zero terminal
potential its episode total is a constant, so the sum says nothing about its strength. What a
gradient responds to is the per-step variation.
"""

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from vsss_league.training import _reset_world, create_rollout_session
from vsss_train.config import load_marl_config
from vsss_train.marl_env import team_action_width

config = replace(load_marl_config("experiments/configs/m24-3-mappo-circular.toml"), num_envs=32)
session = create_rollout_session(
    config,
    Path("tests/golden/m1_match_config.json").read_text(),
    Path("tests/golden/m1_match_state.json").read_text(),
)
environment = session.environment
for world in range(environment.num_envs):
    _reset_world(session, world, 4400 + world)
session.initialized = True

samples: dict[str, list[float]] = defaultdict(list)
generator = np.random.default_rng(4)
width = team_action_width(config.action_parser)
for _ in range(600):
    environment.reset_reward_terms()
    environment.step(
        generator.uniform(-1.0, 1.0, (environment.num_envs, 3, width)).astype(np.float32), None
    )
    for name, value in environment.reward_terms.items():
        samples[name].append(value / max(1, environment.reward_decisions))

print(f"{'term':<22}{'mean':>12}{'std':>12}{'mean |x|':>12}")
rows = sorted(samples.items(), key=lambda kv: -float(np.abs(kv[1]).mean()))
for name, values in rows[:9]:
    array = np.asarray(values)
    print(f"{name:<22}{array.mean():>12.6f}{array.std():>12.6f}{np.abs(array).mean():>12.6f}")
