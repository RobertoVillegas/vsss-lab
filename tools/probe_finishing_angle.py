"""Can a primitive finish from an angle? Measured on the primitive, not on a drill.

The drill generator only ever places the striker on the shooting line, so sweeping a drill's
difficulty cannot answer this. Here the robot is placed at a chosen angle around the ball
directly, the goal is ahead, and one intent is driven on a loop until the ball is in or the
clock runs out.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from vsss_baselines.controllers import robot_pose
from vsss_env._native import BatchSimulator
from vsss_train.primitives import circular_primitive_wheel_actions

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()
cfg = json.loads(CONFIG)
GOAL_X = cfg["field"]["length"] / 2.0


def scenario(angle: float, ball: tuple[float, float], reach: float) -> str:
    """One striker placed `angle` radians off the shooting line, everyone else out of the way."""
    state = json.loads(STATE)
    state.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    state["ball"].update(x=ball[0], y=ball[1], vx=0.0, vy=0.0, omega=0.0)
    shooting = math.atan2(-ball[1], GOAL_X - ball[0])
    bearing = shooting + math.pi + angle
    for index, robot in enumerate(state["robots"]):
        robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
        robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0)
        if index == 0:
            robot["pose"].update(
                x=ball[0] + reach * math.cos(bearing),
                y=ball[1] + reach * math.sin(bearing),
                theta=bearing + math.pi,
            )
        else:  # parked far behind, so only the striker and the ball are in play
            robot["pose"].update(x=-0.65, y=-0.45 + 0.2 * index, theta=0.0)
    return json.dumps(state)


def token(skill: str, heading: float) -> list[float]:
    index = {"navigate": 0.0, "strike": 1.0}[skill]
    wrapped = (heading + math.pi) % (2.0 * math.pi) - math.pi
    return [index, wrapped / math.pi, 1.0]


def attempt(skill: str, angle: float, ball: tuple[float, float], reach: float) -> bool:
    simulator = BatchSimulator(CONFIG, scenario(angle, ball, reach), 1)
    state = np.asarray(simulator.reset())[0]
    for _ in range(240):
        if int(state[-1]) & 1:
            return True
        pose = robot_pose(state, 0)
        if skill == "navigate_ball":
            heading = math.atan2(state[6] - pose[1], state[5] - pose[0])
            intent = "navigate"
        else:
            heading = math.atan2(-state[6], GOAL_X - state[5])
            intent = "navigate" if skill == "navigate_goal" else "strike"
        tokens = np.zeros((3, 3), dtype=np.float32)
        tokens[:, 0] = -1.0
        tokens[0] = token(intent, heading)
        wheels = circular_primitive_wheel_actions(state, team=0, tokens=tokens)
        command = np.zeros((1, 6, 2), dtype=np.float32)
        command[0, :3] = wheels * 12.0
        state = np.asarray(simulator.step_repeated(command, 4))[0]
    return bool(int(state[-1]) & 1)


BALLS = ((0.30, 0.00), (0.45, 0.10), (0.20, -0.15), (0.50, -0.05))
REACHES = (0.14, 0.20, 0.26)
DEGREES = (0, 30, 60, 90, 120, 150)

print("goles desde un ángulo, colocando al rematador directamente")
print(f"{'intent':<16}" + "".join(f"{d:>8}°" for d in DEGREES))
for skill in ("navigate_ball", "navigate_goal", "strike"):
    cells = []
    for degrees in DEGREES:
        angle = math.radians(degrees)
        scored = sum(
            attempt(skill, side * angle, ball, reach)
            for ball in BALLS
            for reach in REACHES
            for side in (1.0, -1.0)
        )
        cells.append(f"{scored / (len(BALLS) * len(REACHES) * 2):>9.2f}")
    print(f"{skill:<16}" + "".join(cells))
