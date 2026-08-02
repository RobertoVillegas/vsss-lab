"""How much of a ball-position shaping term is signal and how much is discount drift.

Potential shaping applied as `c*(gamma*Phi' - Phi)` carries a standing charge of
`c*(gamma-1)*Phi` on every step. The existing goal-geometry term was measured to be 92 per cent
that charge. Before choosing a coefficient for a new potential, the same split has to be known
for it: the part that pays for carrying the ball, against the part that is paid for existing.
"""

import json
import math
from pathlib import Path

import numpy as np
from vsss_league.training import _reset_world, create_rollout_session
from vsss_train.config import load_marl_config
from vsss_train.marl_env import team_action_width

cfg = json.loads(Path("tests/golden/m1_match_config.json").read_text())
GOAL_X = cfg["field"]["length"] / 2.0
HALF_GOAL = cfg["field"]["goal_width"] / 2.0
PEAK = None


def subtended(x: float, y: float, attack_sign: float) -> float:
    goal_x = attack_sign * GOAL_X
    near = (
        math.atan2(HALF_GOAL - y, goal_x - x)
        if attack_sign > 0
        else math.atan2(HALF_GOAL - y, x - goal_x)
    )
    far = (
        math.atan2(-HALF_GOAL - y, goal_x - x)
        if attack_sign > 0
        else math.atan2(-HALF_GOAL - y, x - goal_x)
    )
    return abs(near - far)


PEAK = subtended(GOAL_X - 0.02, 0.0, 1.0)


def potential(state, team: int) -> float:
    sign = 1.0 if team == 0 else -1.0
    return min(1.0, subtended(float(state[5]) * sign, float(state[6]), 1.0) / PEAK)


config = load_marl_config("experiments/configs/m24-3-mappo-circular.toml")
match_config = Path("tests/golden/m1_match_config.json").read_text()
match_state = Path("tests/golden/m1_match_state.json").read_text()
session = create_rollout_session(config, match_config, match_state)
environment = session.environment
gamma = config.gamma

for world in range(environment.num_envs):
    _reset_world(session, world, 1500 + world)
session.initialized = True

drift = np.zeros(environment.num_envs)
telescope = np.zeros(environment.num_envs)
steps = np.zeros(environment.num_envs, dtype=int)
previous = np.array(
    [
        potential(environment.states[w], int(environment.controlled_teams[w]))
        for w in range(environment.num_envs)
    ]
)
episodes = []
generator = np.random.default_rng(9)
width = team_action_width(config.action_parser)

for _ in range(1500):
    actions = generator.uniform(-1.0, 1.0, (environment.num_envs, 3, width)).astype(np.float32)
    _, _, done, _, _ = environment.step(actions, None)
    current = np.array(
        [
            potential(environment.states[w], int(environment.controlled_teams[w]))
            for w in range(environment.num_envs)
        ]
    )
    # gamma*Phi' - Phi  ==  (Phi' - Phi)  +  (gamma - 1)*Phi'
    telescope += current - previous
    drift += (gamma - 1.0) * current
    steps += 1
    for world in np.flatnonzero(done):
        episodes.append((float(telescope[world]), float(drift[world]), int(steps[world])))
        telescope[world] = drift[world] = 0.0
        steps[world] = 0
        _reset_world(session, int(world), int(generator.integers(0, 10**6)))
        current[world] = potential(
            environment.states[world], int(environment.controlled_teams[world])
        )
    previous = current

signal = sum(abs(row[0]) for row in episodes) / len(episodes)
charge = sum(abs(row[1]) for row in episodes) / len(episodes)
length = sum(row[2] for row in episodes) / len(episodes)
print(f"{len(episodes)} episodios, {length:.0f} pasos de media")
print("por episodio, en unidades del coeficiente c:")
print(f"  parte que paga llevar la pelota (telescopio)  {signal:.3f} c")
print(f"  deriva del descuento (cobro por existir)      {charge:.3f} c")
print(f"  la deriva es {charge / max(signal, 1e-9):.1f}x la senal")
print()
for name, shaping_gamma in (("gamma = descuento (0.99)", gamma), ("gamma = 1 (sin deriva)", 1.0)):
    total = signal + abs(shaping_gamma - 1.0) / max(1e-9, abs(gamma - 1.0)) * charge
    print(f"  {name:<26} total por episodio ~ {total:.3f} c")
