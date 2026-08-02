"""How much reward a drill episode carries against a full-match episode.

Cycle 2 showed the policy selecting primitives correctly inside drills and navigating through
matches. If the drill terminal is an order of magnitude louder than anything a match pays, the
match is not losing the argument on merit — it is being outvoted, and the fix is balance rather
than a different primitive or a different penalty.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from vsss_league.training import _reset_world, create_rollout_session
from vsss_train.config import load_marl_config
from vsss_train.marl_env import team_action_width
from vsss_train.marl_ppo import MarlLearner

config = load_marl_config("experiments/configs/m24-3-mappo-circular.toml")
match_config = Path("tests/golden/m1_match_config.json").read_text()
match_state = Path("tests/golden/m1_match_state.json").read_text()

learner = MarlLearner(config)
session = create_rollout_session(config, match_config, match_state)
environment = session.environment

# Accumulate per world and attribute each finished episode to the kind it was running.
totals: dict[str, list[float]] = defaultdict(list)
lengths: dict[str, list[int]] = defaultdict(list)
running = np.zeros(environment.num_envs)
steps = np.zeros(environment.num_envs, dtype=int)
kinds = [
    "full_match" if session.semantic_scenarios[w] is None else "drill"
    for w in range(environment.num_envs)
]


def kind_of(world: int) -> str:
    return "full_match" if session.semantic_scenarios[world] is None else "drill"


generator = np.random.default_rng(5)
width = team_action_width(config.action_parser)
for world in range(environment.num_envs):
    _reset_world(session, world, 900 + world)
    kinds[world] = kind_of(world)
session.initialized = True

for _ in range(1200):
    actions = generator.uniform(-1.0, 1.0, (environment.num_envs, 3, width)).astype(np.float32)
    _, rewards, done, _, _ = environment.step(actions, None)
    running += rewards
    steps += 1
    for world in np.flatnonzero(done):
        totals[kinds[world]].append(float(running[world]))
        lengths[kinds[world]].append(int(steps[world]))
        running[world] = 0.0
        steps[world] = 0
        _reset_world(session, int(world), int(generator.integers(0, 10**6)))
        kinds[world] = kind_of(int(world))

print("reward accumulated per finished episode, environment terms only")
print(f"{'kind':<14}{'episodes':>10}{'mean':>10}{'|mean|':>10}{'steps':>9}")
for kind in ("drill", "full_match"):
    values = totals.get(kind, [])
    if not values:
        print(f"{kind:<14}{'none':>10}")
        continue
    mean = sum(values) / len(values)
    magnitude = sum(abs(value) for value in values) / len(values)
    length = sum(lengths[kind]) / len(lengths[kind])
    print(f"{kind:<14}{len(values):>10}{mean:>10.3f}{magnitude:>10.3f}{length:>9.0f}")

terminal = config.semantic_terminal_reward
timeout = terminal if config.semantic_timeout_penalty is None else config.semantic_timeout_penalty
print()
print(
    f"drill terminal applied on top, outside these numbers: +/-{terminal} on success or "
    f"failure, -{timeout} on timeout"
)
print(f"a goal pays {config.goal_coefficient}")
