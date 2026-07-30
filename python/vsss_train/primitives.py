"""Causal deterministic motion primitives for hierarchical soccer policies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from vsss_baselines.controllers import go_to_target, robot_pose

FloatArray = NDArray[np.float32]
ROBOT_BASE = 10
ROBOT_WIDTH = 11


@dataclass(frozen=True)
class PrimitiveCommand:
    """Decoded policy intent independent from physical robot identity."""

    skill: str
    direction_index: int | None


class SoccerPrimitiveSet:
    """Stop plus eight-way navigation and directed-strike actions."""

    directions = 8
    action_count = 1 + 2 * directions

    @classmethod
    def encode(cls, indices: Tensor) -> Tensor:
        """Encode categorical actions into a bounded rollout transport token."""
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("primitive actions must be integer indices")
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= cls.action_count)):
            raise ValueError("primitive action index out of range")
        skill = torch.where(
            indices == 0,
            torch.full_like(indices, -1, dtype=torch.float32),
            torch.where(
                indices <= cls.directions,
                torch.zeros_like(indices, dtype=torch.float32),
                torch.ones_like(indices, dtype=torch.float32),
            ),
        )
        direction_index = torch.where(
            indices == 0,
            torch.zeros_like(indices),
            (indices - 1) % cls.directions,
        )
        direction = direction_index.to(torch.float32) * (2.0 / (cls.directions - 1)) - 1.0
        direction = torch.where(indices == 0, torch.zeros_like(direction), direction)
        return torch.stack((skill, direction), dim=-1)

    @classmethod
    def decode(cls, token: FloatArray) -> PrimitiveCommand:
        """Decode a bounded token without relying on exact floating-point equality."""
        if token.shape != (2,):
            raise ValueError("primitive token must contain skill and direction")
        skill_value = float(token[0])
        if skill_value < -0.5:
            return PrimitiveCommand("stop", None)
        direction = round((float(np.clip(token[1], -1.0, 1.0)) + 1.0) * 0.5 * 7)
        return PrimitiveCommand("navigate" if skill_value < 0.5 else "strike", direction)


def canonical_direction(index: int, team: int) -> tuple[float, float]:
    """Return one eight-way direction reflected into the team's world frame."""
    if not 0 <= index < SoccerPrimitiveSet.directions:
        raise ValueError("direction index out of range")
    if team not in (0, 1):
        raise ValueError("team must be blue or yellow")
    angle = index * math.tau / SoccerPrimitiveSet.directions
    canonical = (math.cos(angle), math.sin(angle))
    sign = 1.0 if team == 0 else -1.0
    return sign * canonical[0], sign * canonical[1]


def nearest_canonical_direction(vector: tuple[float, float], team: int) -> int:
    """Quantize a world-frame vector into the shared canonical direction set."""
    if math.hypot(*vector) <= 1e-9:
        return 0
    return max(
        range(SoccerPrimitiveSet.directions),
        key=lambda index: sum(
            first * second
            for first, second in zip(
                canonical_direction(index, team),
                vector,
                strict=True,
            )
        ),
    )


def primitive_wheel_actions(
    state: FloatArray,
    *,
    team: int,
    tokens: FloatArray,
    ball_deceleration: float = 0.8,
) -> FloatArray:
    """Convert policy intents to bounded differential-drive wheel commands."""
    if tokens.shape != (3, 2):
        raise ValueError("primitive team actions must have shape (3, 2)")
    if ball_deceleration <= 0.0:
        raise ValueError("ball deceleration must be positive")
    result = np.zeros((3, 2), dtype=np.float32)
    offset = 0 if team == 0 else 3
    for local_slot, token in enumerate(tokens):
        slot = offset + local_slot
        if not bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10])):
            continue
        command = SoccerPrimitiveSet.decode(token)
        if command.skill == "stop":
            continue
        assert command.direction_index is not None
        direction = canonical_direction(command.direction_index, team)
        pose = robot_pose(state, slot)
        if command.skill == "navigate":
            target = (pose[0] + 0.4 * direction[0], pose[1] + 0.4 * direction[1])
        else:
            target = _strike_target(
                state,
                pose,
                direction,
                ball_deceleration=ball_deceleration,
            )
        result[local_slot] = go_to_target(pose, target)
    return result


def _strike_target(
    state: FloatArray,
    pose: tuple[float, float, float],
    direction: tuple[float, float],
    *,
    ball_deceleration: float,
) -> tuple[float, float]:
    """Select a reachable behind-ball point, then drive through contact."""
    ball = np.asarray(state[5:7], dtype=np.float64)
    velocity = np.asarray(state[7:9], dtype=np.float64)
    robot = np.asarray(pose[:2], dtype=np.float64)
    exit_direction = np.asarray(direction, dtype=np.float64)
    contact_offset = 0.10
    selected_ball = ball
    selected_acquisition = ball - contact_offset * exit_direction
    maximum_robot_speed = 0.62
    maximum_turn_rate = 5.0
    heading = np.asarray((math.cos(pose[2]), math.sin(pose[2])), dtype=np.float64)

    for elapsed in np.linspace(0.0, 0.6, 7):
        if elapsed == 0.0:
            candidate_ball = ball
        else:
            speed = float(np.linalg.norm(velocity))
            if speed <= 1e-8:
                candidate_ball = ball
            else:
                travel = min(speed * elapsed, speed * speed / (2.0 * ball_deceleration))
                candidate_ball = ball + travel * velocity / speed
        acquisition = candidate_ball - contact_offset * exit_direction
        displacement = acquisition - robot
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-8:
            heading_error = 0.0
        else:
            cosine = float(np.clip(np.dot(heading, displacement / distance), -1.0, 1.0))
            heading_error = math.acos(cosine)
        arrival = distance / maximum_robot_speed + heading_error / maximum_turn_rate
        selected_ball = candidate_ball
        selected_acquisition = acquisition
        if arrival <= elapsed + 0.08:
            break

    acquisition_error = float(np.linalg.norm(selected_acquisition - robot))
    ball_vector = selected_ball - robot
    ball_distance = float(np.linalg.norm(ball_vector))
    aligned = ball_distance > 1e-8 and float(
        np.dot(ball_vector / ball_distance, exit_direction)
    ) >= math.cos(0.60)
    # Differential drive cannot settle on a point with millimetric precision
    # without oscillation. Enter the drive-through phase once the robot is
    # inside a body-scale acquisition envelope and faces the exit half-plane.
    if acquisition_error <= 0.11 and aligned:
        drive_through = selected_ball + 0.28 * exit_direction
        return float(drive_through[0]), float(drive_through[1])
    return float(selected_acquisition[0]), float(selected_acquisition[1])
