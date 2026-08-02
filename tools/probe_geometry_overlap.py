"""Does the shot drill's geometry occur in a match?

Cycles 2 and 3 leave generalization as the only surviving explanation: equal reward, correct
per-drill selection, and navigate everywhere in a match. If the drill places the striker in a
configuration play never produces, the policy learned a shape rather than a situation.

Three quantities describe a finishing chance, measured for the controlled robot nearest the
ball: how far it is from the ball, how far the ball is from the goal, and how far off the
shooting line it is — the angle between where it must push and where it stands.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from vsss_env._native import BatchSimulator
from vsss_league.training import create_rollout_session
from vsss_train.config import load_marl_config
from vsss_train.marl_env import team_action_width
from vsss_train.marl_ppo import load_policy_actor
from vsss_train.semantic_scenarios import (
    GENERATOR_REVISION,
    SkillDifficulty,
    SkillScenarioParameters,
    compile_skill_scenario,
)

cfgj = Path("tests/golden/m1_match_config.json").read_text()
stj = Path("tests/golden/m1_match_state.json").read_text()
base, cfg = json.loads(stj), json.loads(cfgj)
GOAL_X = cfg["field"]["length"] / 2.0


def geometry(state, slot: int) -> tuple[float, float, float]:
    """Distance to the ball, ball's distance to goal, and degrees off the shooting line."""
    base_index = 10 + slot * 11
    robot = (float(state[base_index + 2]), float(state[base_index + 3]))
    ball = (float(state[5]), float(state[6]))
    to_ball = (ball[0] - robot[0], ball[1] - robot[1])
    ball_to_goal = (GOAL_X - ball[0], -ball[1])
    reach = math.hypot(*to_ball)
    range_to_goal = math.hypot(*ball_to_goal)
    if reach < 1e-6 or range_to_goal < 1e-6:
        return reach, range_to_goal, 0.0
    cosine = (to_ball[0] * ball_to_goal[0] + to_ball[1] * ball_to_goal[1]) / (reach * range_to_goal)
    return reach, range_to_goal, math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def nearest_blue(state) -> int:
    return min(
        range(3),
        key=lambda s: math.hypot(
            state[5] - state[10 + s * 11 + 2], state[6] - state[10 + s * 11 + 3]
        ),
    )


drill = []
for level in (0.0, 0.25, 0.5, 0.75, 1.0):
    for seed in range(24):
        parameters = SkillScenarioParameters(
            schema_version=1,
            family="shot",
            seed=seed,
            controlled_team="blue",
            difficulty=SkillDifficulty(spawn_distance=0.3, ball_speed=0.1, ball_angle=level),
            roster="3v3",
            horizon=240,
            holdout=False,
            generator_revision=GENERATOR_REVISION,
        )
        scenario = compile_skill_scenario(parameters, base, cfg)
        simulator = BatchSimulator(cfgj, json.dumps(scenario.scenario.state), 1)
        state = np.asarray(simulator.reset())[0]
        drill.append(geometry(state, int(scenario.context.controlled_robot_id[1:])))

trained = load_marl_config("experiments/configs/m24-3-mappo-circular.toml")
config = replace(
    trained,
    num_envs=32,
    semantic_curriculum=False,
    semantic_phased_curriculum=False,
    adaptive_curriculum=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor, _ = load_policy_actor(
    Path("/home/rob/runs/vsss-m24-3-run-0009/checkpoints/iteration-001500.pt"), trained, device
)
session = create_rollout_session(config, cfgj, stj)
environment = session.environment
for world in range(environment.num_envs):
    environment.reset(world, 7000 + world)

match = []
width = team_action_width(config.action_parser)
for _ in range(200):
    observation = environment.current_observations().to(device)
    with torch.no_grad():
        tokens = actor.deterministic_action(observation).cpu().numpy()
    for world in range(environment.num_envs):
        state = environment.states[world]
        if float(state[5]) > 0.2:  # ball in the attacking third, where a chance exists
            match.append(geometry(state, nearest_blue(state)))
    environment.step(tokens.astype(np.float32).reshape(-1, 3, width), None)


def spread(rows, index):
    values = sorted(row[index] for row in rows)
    if not values:
        return "—"
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return f"{pick(0.10):.2f} {pick(0.50):.2f} {pick(0.90):.2f}"


print(f"drill states {len(drill)}, match states in the attacking third {len(match)}")
print(f"{'quantity':<26}{'drill p10/p50/p90':>22}{'match p10/p50/p90':>22}")
for index, name in enumerate(("robot to ball (m)", "ball to goal (m)", "off shooting line (deg)")):
    print(f"{name:<26}{spread(drill, index):>22}{spread(match, index):>22}")

behind = lambda rows: sum(1 for row in rows if row[2] < 45.0) / max(1, len(rows))  # noqa: E731
close = lambda rows: sum(1 for row in rows if row[0] < 0.20) / max(1, len(rows))  # noqa: E731
print()
print(f"within 45 deg of the shooting line: drill {behind(drill):.2f}, match {behind(match):.2f}")
print(f"within 20 cm of the ball:           drill {close(drill):.2f}, match {close(match):.2f}")
