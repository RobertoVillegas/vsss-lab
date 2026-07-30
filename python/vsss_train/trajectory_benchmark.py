"""Reward-independent exact-simulator benchmarks for soccer primitives."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from vsss_train.marl_env import MarlMatchEnv
from vsss_train.primitives import SoccerPrimitiveSet


@dataclass(frozen=True)
class TrajectoryTrial:
    case: str
    contacted: bool
    contact_seconds: float | None
    minimum_distance_m: float
    maximum_ball_speed_mps: float
    exit_direction_error_deg: float | None
    translational_motion_ratio: float
    physically_valid: bool


def benchmark_primitives(config_json: str, state_json: str) -> dict[str, object]:
    """Run fixed stationary/moving-ball strike cases through exact Rapier physics."""
    trials = tuple(
        _run_trial(config_json, state_json, case, initial_velocity, lateral)
        for case, initial_velocity, lateral in (
            ("stationary-center", (0.0, 0.0), 0.0),
            ("moving-forward", (0.18, 0.0), 0.12),
            ("moving-lateral", (0.05, -0.20), 0.18),
        )
    )
    return {
        "schema_version": 1,
        "backend": "rapier-exact",
        "trials": [asdict(trial) for trial in trials],
        "contact_rate": sum(trial.contacted for trial in trials) / len(trials),
        "directional_rate": sum(
            trial.exit_direction_error_deg is not None and trial.exit_direction_error_deg <= 45.0
            for trial in trials
        )
        / len(trials),
        "physically_valid": all(trial.physically_valid for trial in trials),
    }


def _run_trial(
    config_json: str,
    state_json: str,
    case: str,
    initial_velocity: tuple[float, float],
    lateral: float,
) -> TrajectoryTrial:
    snapshot = copy.deepcopy(json.loads(state_json))
    snapshot.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    snapshot["ball"].update(
        x=-0.02,
        y=lateral,
        vx=initial_velocity[0],
        vy=initial_velocity[1],
        omega=0.0,
    )
    for index, robot in enumerate(snapshot["robots"]):
        robot["enabled"] = index == 0
        if index == 0:
            robot["pose"].update(x=-0.42, y=lateral - 0.06, theta=0.0)
        robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
        robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0)
    environment = MarlMatchEnv(
        config_json,
        state_json,
        stage=7,
        horizon=400,
        action_repeat=4,
        action_parser="primitive",
        stagnation_seconds=8.0,
    )
    environment.reset_state(snapshot)
    tokens = SoccerPrimitiveSet.encode(torch.tensor([9, 0, 0])).numpy()
    minimum_distance = float("inf")
    contact_step: int | None = None
    maximum_ball_speed = 0.0
    exit_velocity: tuple[float, float] | None = None
    moving_steps = 0
    executed_steps = 0
    for step in range(400):
        environment.step(tokens)
        executed_steps += 1
        robot_x, robot_y = float(environment.state[12]), float(environment.state[13])
        distance = math.hypot(
            float(environment.state[5]) - robot_x,
            float(environment.state[6]) - robot_y,
        )
        minimum_distance = min(minimum_distance, distance)
        robot_speed = math.hypot(float(environment.state[15]), float(environment.state[16]))
        moving_steps += int(robot_speed >= 0.05)
        ball_velocity = (float(environment.state[7]), float(environment.state[8]))
        ball_speed = math.hypot(*ball_velocity)
        maximum_ball_speed = max(maximum_ball_speed, ball_speed)
        if contact_step is None and distance <= 0.0775:
            contact_step = step
        directed_impulse = ball_velocity[0] - initial_velocity[0]
        if contact_step is not None and directed_impulse >= 0.05:
            exit_velocity = ball_velocity
            break
    direction_error = None
    if exit_velocity is not None:
        direction_error = abs(math.degrees(math.atan2(exit_velocity[1], exit_velocity[0])))
    return TrajectoryTrial(
        case=case,
        contacted=contact_step is not None,
        contact_seconds=(contact_step + 1) * environment.decision_period
        if contact_step is not None
        else None,
        minimum_distance_m=minimum_distance,
        maximum_ball_speed_mps=maximum_ball_speed,
        exit_direction_error_deg=direction_error,
        translational_motion_ratio=moving_steps / executed_steps,
        physically_valid=bool(np.isfinite(environment.state).all()),
    )
