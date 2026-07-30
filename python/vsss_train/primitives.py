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
    direction_labels = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")

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


@dataclass(frozen=True)
class ParametricPrimitiveCommand:
    """A semantic skill with continuous heading and drive intensity."""

    skill: str
    direction: float
    intensity: float


class ParametricPrimitiveSet:
    """Stop, navigate, and strike with continuous geometric parameters."""

    action_count = 3
    skill_labels = ("stop", "navigate", "strike")

    @classmethod
    def encode(cls, skill_indices: Tensor, parameters: Tensor) -> Tensor:
        """Encode categorical skills and bounded direction/intensity parameters."""
        if skill_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("parametric primitive skills must be integer indices")
        if parameters.shape != (*skill_indices.shape, 3):
            raise ValueError(
                "parametric primitive parameters must end in direction x/y and intensity"
            )
        if bool(torch.any(skill_indices < 0)) or bool(torch.any(skill_indices >= cls.action_count)):
            raise ValueError("parametric primitive skill index out of range")
        skill = skill_indices.to(torch.float32) - 1.0
        bounded = parameters.clamp(-1.0, 1.0)
        return torch.cat((skill.unsqueeze(-1), bounded), dim=-1)

    @classmethod
    def decode(cls, token: FloatArray) -> ParametricPrimitiveCommand:
        """Decode a bounded transport token into physical controller parameters."""
        if token.shape != (4,):
            raise ValueError(
                "parametric primitive token must contain skill, direction x/y, and intensity"
            )
        skill_index = int(np.clip(round(float(token[0]) + 1.0), 0, 2))
        direction_vector = np.asarray(token[1:3], dtype=np.float64)
        norm = float(np.linalg.norm(direction_vector))
        direction = (
            math.atan2(float(direction_vector[1]), float(direction_vector[0]))
            if norm > 1e-6
            else 0.0
        )
        return ParametricPrimitiveCommand(
            cls.skill_labels[skill_index],
            direction,
            float(np.clip((token[3] + 1.0) * 0.5, 0.0, 1.0)),
        )


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


def parametric_primitive_wheel_actions(
    state: FloatArray,
    *,
    team: int,
    tokens: FloatArray,
    ball_deceleration: float = 0.8,
) -> FloatArray:
    """Execute continuously directed skills with curved, speed-aware arrival."""
    if tokens.shape != (3, 4):
        raise ValueError("parametric primitive team actions must have shape (3, 4)")
    if ball_deceleration <= 0.0:
        raise ValueError("ball deceleration must be positive")
    result = np.zeros((3, 2), dtype=np.float32)
    offset = 0 if team == 0 else 3
    for local_slot, token in enumerate(tokens):
        slot = offset + local_slot
        if not bool(float(state[ROBOT_BASE + slot * ROBOT_WIDTH + 10])):
            continue
        command = ParametricPrimitiveSet.decode(token)
        if command.skill == "stop" or command.intensity <= 1e-4:
            continue
        sign = 1.0 if team == 0 else -1.0
        direction = (sign * math.cos(command.direction), sign * math.sin(command.direction))
        pose = robot_pose(state, slot)
        if command.skill == "navigate":
            target = (pose[0] + 0.4 * direction[0], pose[1] + 0.4 * direction[1])
            arrival_scale = 1.0
        else:
            target = _strike_target(
                state,
                pose,
                direction,
                ball_deceleration=ball_deceleration,
            )
            ball = np.asarray(state[5:7], dtype=np.float64)
            target_vector = np.asarray(target, dtype=np.float64) - ball
            arrival_scale = 1.0 if float(np.dot(target_vector, direction)) > 0.0 else 0.72
        wheels = go_to_target(pose, target)
        result[local_slot] = wheels * np.float32(command.intensity * arrival_scale)
    return result


def describe_parametric_primitive_actions(
    state: FloatArray,
    *,
    team: int,
    tokens: FloatArray,
) -> list[dict[str, object]]:
    """Describe continuously parameterized intent for replay inspection."""
    if tokens.shape != (3, 4):
        raise ValueError("parametric primitive team actions must have shape (3, 4)")
    offset = 0 if team == 0 else 3
    descriptions: list[dict[str, object]] = []
    for local_slot, token in enumerate(tokens):
        slot = offset + local_slot
        command = ParametricPrimitiveSet.decode(token)
        pose = robot_pose(state, slot)
        sign = 1.0 if team == 0 else -1.0
        direction = (
            (sign * math.cos(command.direction), sign * math.sin(command.direction))
            if command.skill != "stop"
            else (0.0, 0.0)
        )
        ball = (float(state[5]), float(state[6]))
        if command.skill == "stop":
            target = (pose[0], pose[1])
            phase = "stop"
        elif command.skill == "navigate":
            target = (pose[0] + 0.4 * direction[0], pose[1] + 0.4 * direction[1])
            phase = "navigate"
        else:
            target = _strike_target(state, pose, direction, ball_deceleration=0.8)
            exit_dot = (target[0] - ball[0]) * direction[0] + (target[1] - ball[1]) * direction[1]
            phase = "strike" if exit_dot > 0.0 else "acquire"
        descriptions.append(
            {
                "skill": command.skill,
                "direction_index": None,
                "direction": f"{math.degrees(command.direction):+.1f}°",
                "direction_radians": command.direction,
                "intensity": command.intensity,
                "phase": phase,
                "target": {"x": target[0], "y": target[1]},
                "exit_direction": {"x": direction[0], "y": direction[1]},
                "ball_distance": math.hypot(pose[0] - ball[0], pose[1] - ball[1]),
            }
        )
    return descriptions


def describe_primitive_actions(
    state: FloatArray,
    *,
    team: int,
    tokens: FloatArray,
) -> list[dict[str, object]]:
    """Describe the exact deterministic plan used for replay inspection."""
    if tokens.shape != (3, 2):
        raise ValueError("primitive team actions must have shape (3, 2)")
    offset = 0 if team == 0 else 3
    descriptions: list[dict[str, object]] = []
    for local_slot, token in enumerate(tokens):
        slot = offset + local_slot
        command = SoccerPrimitiveSet.decode(token)
        pose = robot_pose(state, slot)
        ball = (float(state[5]), float(state[6]))
        direction = (
            canonical_direction(command.direction_index, team)
            if command.direction_index is not None
            else (0.0, 0.0)
        )
        if command.skill == "stop":
            target = (pose[0], pose[1])
            phase = "stop"
        elif command.skill == "navigate":
            target = (pose[0] + 0.4 * direction[0], pose[1] + 0.4 * direction[1])
            phase = "navigate"
        else:
            target = _strike_target(state, pose, direction, ball_deceleration=0.8)
            exit_dot = (target[0] - ball[0]) * direction[0] + (target[1] - ball[1]) * direction[1]
            phase = "strike" if exit_dot > 0.0 else "acquire"
        descriptions.append(
            {
                "skill": command.skill,
                "direction_index": command.direction_index,
                "direction": (
                    SoccerPrimitiveSet.direction_labels[command.direction_index]
                    if command.direction_index is not None
                    else None
                ),
                "phase": phase,
                "target": {"x": target[0], "y": target[1]},
                "exit_direction": {"x": direction[0], "y": direction[1]},
                "ball_distance": math.hypot(pose[0] - ball[0], pose[1] - ball[1]),
            }
        )
    return descriptions


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
