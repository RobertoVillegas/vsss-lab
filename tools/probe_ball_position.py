"""How often does a policy get the ball into a position its primitives can convert from?

The carry gradient's claim is not that it scores — finishing is capped by the action set — but
that it brings the ball to where finishing is possible. That is what has to be measured to judge
it, and it is not the same thing as goals per minute.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from vsss_league.training import create_rollout_session
from vsss_train.config import load_marl_config
from vsss_train.marl_env import _goal_mouth_potential, team_action_width
from vsss_train.marl_ppo import load_policy_actor

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()
MATCH = json.loads(CONFIG)

trained = load_marl_config(ROOT / "experiments/configs/m24-3-mappo-circular.toml")
config = replace(
    trained,
    num_envs=32,
    semantic_curriculum=False,
    semantic_phased_curriculum=False,
    adaptive_curriculum=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def occupancy(path: Path, steps: int = 300) -> tuple[float, float, float]:
    """Mean potential, and the share of time the ball is convertible or nearly so."""
    actor, _ = load_policy_actor(path, trained, device)
    session = create_rollout_session(config, CONFIG, STATE)
    environment = session.environment
    for world in range(environment.num_envs):
        environment.reset(world, 6000 + world)
    width = team_action_width(config.action_parser)
    values: list[float] = []
    for _ in range(steps):
        observation = environment.current_observations().to(device)
        with torch.no_grad():
            tokens = actor.deterministic_action(observation).cpu().numpy()
        for world in range(environment.num_envs):
            values.append(
                _goal_mouth_potential(
                    environment.states[world], MATCH, int(environment.controlled_teams[world])
                )
            )
        environment.step(tokens.astype(np.float32).reshape(-1, 3, width), None)
    array = np.asarray(values)
    return float(array.mean()), float((array > 0.5).mean()), float((array > 0.3).mean())


print(f"{'checkpoint':<26}{'Phi medio':>11}{'Phi>0.5':>10}{'Phi>0.3':>10}")
for label, path in [tuple(pair.split("=", 1)) for pair in sys.argv[1:]]:
    mean, high, mid = occupancy(Path(path))
    print(f"{label:<26}{mean:>11.3f}{high:>10.3f}{mid:>10.3f}")
