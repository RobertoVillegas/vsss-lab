"""Throwaway: per-decision trace of a 60° strike attempt, clearing on vs off."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/rob/src/vsss-lab/python")

from vsss_baselines.controllers import robot_pose
from vsss_env._native import BatchSimulator
from vsss_train.primitives import _strike_target, circular_primitive_wheel_actions

ROOT = Path("/home/rob/src/vsss-lab")
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()
cfg = json.loads(CONFIG)
GOAL_X = cfg["field"]["length"] / 2.0


def scenario(angle: float, ball: tuple[float, float], reach: float) -> str:
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
        else:
            robot["pose"].update(x=-0.65, y=-0.45 + 0.2 * index, theta=0.0)
    return json.dumps(state)


def token(skill: str, heading: float) -> list[float]:
    index = {"navigate": 0.0, "strike": 1.0}[skill]
    wrapped = (heading + math.pi) % (2.0 * math.pi) - math.pi
    return [index, wrapped / math.pi, 1.0]


def attempt(
    angle: float,
    ball: tuple[float, float],
    reach: float,
    clearing: bool,
) -> tuple[bool, list[dict[str, float]]]:
    simulator = BatchSimulator(CONFIG, scenario(angle, ball, reach), 1)
    state = np.asarray(simulator.reset())[0]
    trace: list[dict[str, float]] = []
    for step in range(480):
        if int(state[-1]) & 1:
            return True, trace
        pose = robot_pose(state, 0)
        heading = math.atan2(-state[6], GOAL_X - state[5])
        tokens = np.zeros((3, 3), dtype=np.float32)
        tokens[:, 0] = -1.0
        tokens[0] = token("strike", heading)
        bx, by = float(state[5]), float(state[6])
        exit_x, exit_y = math.cos(heading), math.sin(heading)
        target, driving = _strike_target(
            state,
            pose,
            direction=(exit_x, exit_y),
            ball_deceleration=0.8,
            authority=1.0,
            strike_clearing_enabled=clearing,
            strike_clearing_distance=0.16,
        )
        bvx, bvy = bx - pose[0], by - pose[1]
        bd = math.hypot(bvx, bvy)
        align = (bvx * exit_x + bvy * exit_y) / bd if bd > 1e-8 else 0.0
        heading_off = math.degrees(
            abs(
                (math.atan2(pose[2] and math.sin(pose[2]), math.cos(pose[2])) - heading + math.pi)
                % (2 * math.pi)
                - math.pi
            )
        )
        trace.append(
            {
                "step": step,
                "rx": round(pose[0], 3),
                "ry": round(pose[1], 3),
                "heading_off_exit": round(heading_off, 1),
                "robot_ball": round(bd, 3),
                "align_cos": round(align, 2),
                "gate": (
                    1
                    if (
                        math.hypot(target[0] - pose[0], target[1] - pose[1]) <= 0.11
                        and align >= math.cos(0.6)
                    )
                    else 0
                ),
                "driving": int(driving),
                "tx": round(target[0], 3),
                "ty": round(target[1], 3),
                "ball_x": round(bx, 3),
                "ball_y": round(by, 3),
            }
        )
        wheels = circular_primitive_wheel_actions(
            state, team=0, tokens=tokens, strike_clearing_enabled=clearing
        )
        command = np.zeros((1, 6, 2), dtype=np.float32)
        command[0, :3] = wheels * SCALE
        state = np.asarray(simulator.step_repeated(command, 4))[0]
    return bool(int(state[-1]) & 1), trace


SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

for clearing in (True, False):
    scored, trace = attempt(math.radians(60), (0.30, 0.00), 0.26, clearing)
    print(f"=== scale={SCALE} clearing={clearing} scored={scored} steps={len(trace)}")
    for row in trace[::12] + trace[-3:]:
        print("  ", row)
