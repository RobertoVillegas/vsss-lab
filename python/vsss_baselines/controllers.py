"""Differential-drive skills and identity-free dynamic role assignment."""

from itertools import permutations
from math import atan2, cos, hypot, pi

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
ROBOT_BASE = 10
ROBOT_WIDTH = 11


def robot_pose(state: FloatArray, robot_index: int) -> tuple[float, float, float]:
    """Decode one robot pose from the stable M3 row."""
    offset = ROBOT_BASE + robot_index * ROBOT_WIDTH
    return float(state[offset + 2]), float(state[offset + 3]), float(state[offset + 4])


def go_to_target(pose: tuple[float, float, float], target: tuple[float, float]) -> FloatArray:
    """Return bounded wheel commands driving a robot toward a target."""
    x, y, theta = pose
    dx, dy = target[0] - x, target[1] - y
    distance = hypot(dx, dy)
    error = (atan2(dy, dx) - theta + pi) % (2.0 * pi) - pi
    forward = min(1.0, 2.0 * distance) * max(0.0, cos(error))
    # Wheel actions are normalized against the physical wheel-speed limit.
    # A small differential is already a fast yaw command on a 60 mm axle.
    turn = 0.08 * max(-1.0, min(1.0, error / (pi / 2.0)))
    return np.asarray(
        [np.clip(forward - turn, -1.0, 1.0), np.clip(forward + turn, -1.0, 1.0)],
        dtype=np.float32,
    )


def go_to_ball(state: FloatArray, robot_index: int) -> FloatArray:
    """Drive one robot toward the current ball position."""
    return go_to_target(robot_pose(state, robot_index), (float(state[5]), float(state[6])))


def goalie(state: FloatArray, robot_index: int, attack_sign: int) -> FloatArray:
    """Track the ball laterally near the team's own goal."""
    target = (-0.68 * attack_sign, float(np.clip(state[6], -0.18, 0.18)))
    return go_to_target(robot_pose(state, robot_index), target)


class DynamicTeamController:
    """Assign goalie, pressor, and support from geometry on every tick."""

    roles = ("goalie", "pressor", "support")

    def __init__(self, team_offset: int, attack_sign: int) -> None:
        if team_offset not in (0, 3) or attack_sign not in (-1, 1):
            raise ValueError("invalid team geometry")
        self.team_offset = team_offset
        self.attack_sign = attack_sign

    def assign(self, state: FloatArray) -> tuple[str, str, str]:
        """Return a role for each current team slot."""
        ball = (float(state[5]), float(state[6]))
        targets = (
            (-0.68 * self.attack_sign, float(np.clip(state[6], -0.18, 0.18))),
            ball,
            (ball[0] - 0.28 * self.attack_sign, -0.5 * ball[1]),
        )
        poses = [robot_pose(state, self.team_offset + index) for index in range(3)]
        role_order = min(
            permutations(range(3)),
            key=lambda order: sum(
                hypot(poses[index][0] - targets[role][0], poses[index][1] - targets[role][1])
                for index, role in enumerate(order)
            ),
        )
        return tuple(self.roles[role] for role in role_order)  # type: ignore[return-value]

    def actions(self, state: FloatArray) -> FloatArray:
        """Return actions ordered by the team's current slots."""
        result = np.zeros((3, 2), dtype=np.float32)
        ball = (float(state[5]), float(state[6]))
        for local_index, role in enumerate(self.assign(state)):
            robot_index = self.team_offset + local_index
            if role == "goalie":
                result[local_index] = goalie(state, robot_index, self.attack_sign)
            elif role == "pressor":
                result[local_index] = go_to_target(robot_pose(state, robot_index), ball)
            else:
                support = (ball[0] - 0.28 * self.attack_sign, -0.5 * ball[1])
                result[local_index] = go_to_target(robot_pose(state, robot_index), support)
        return result
