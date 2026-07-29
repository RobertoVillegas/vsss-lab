"""Permutation-safe shared-policy models for M6."""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from vsss_train.roles import RoleAssignment, assign_roles, role_features

FloatArray = NDArray[np.float32]
ROBOT_BASE = 10
ROBOT_WIDTH = 11
TEAM_SIZE = 3


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported network activation: {name}")


def _hidden_block(size: int, activation: str, layer_norm: bool) -> list[nn.Module]:
    modules: list[nn.Module] = [nn.Linear(size, size)]
    if layer_norm:
        modules.append(nn.LayerNorm(size))
    modules.append(_activation(activation))
    return modules


class TeamBatch(NamedTuple):
    """Structured team observations with explicit set axes."""

    self_features: Tensor
    ball: Tensor
    goals: Tensor
    context: Tensor
    teammates: Tensor
    opponents: Tensor

    def to(self, device: torch.device) -> TeamBatch:
        return TeamBatch(*(field.to(device) for field in self))

    def permute_agents(self, order: Tensor) -> TeamBatch:
        return TeamBatch(*(field.index_select(0, order) for field in self))

    def select_batch(self, indices: Tensor) -> TeamBatch:
        """Select leading batch entries while retaining the complete agent group."""
        return TeamBatch(*(field.index_select(0, indices) for field in self))

    def permute_entities(self, teammate_order: Tensor, opponent_order: Tensor) -> TeamBatch:
        return TeamBatch(
            self.self_features,
            self.ball,
            self.goals,
            self.context,
            self.teammates.index_select(1, teammate_order),
            self.opponents.index_select(1, opponent_order),
        )


def stack_team_batches(observations: list[TeamBatch]) -> TeamBatch:
    """Stack observations along a new leading batch dimension."""
    if not observations:
        raise ValueError("cannot stack an empty observation list")
    return TeamBatch(
        *(
            torch.stack([getattr(observation, name) for observation in observations])
            for name in TeamBatch._fields
        )
    )


def _robot(state: FloatArray, slot: int) -> FloatArray:
    start = ROBOT_BASE + slot * ROBOT_WIDTH
    return state[start : start + ROBOT_WIDTH]


def _canonical_robot(robot: FloatArray, attack_sign: float) -> tuple[float, ...]:
    theta = float(robot[4]) + (0.0 if attack_sign > 0 else math.pi)
    return (
        attack_sign * float(robot[2]),
        attack_sign * float(robot[3]),
        math.cos(theta),
        math.sin(theta),
        attack_sign * float(robot[5]),
        attack_sign * float(robot[6]),
        float(robot[7]),
        float(robot[8]),
        float(robot[9]),
        float(robot[10]),
    )


def build_team_observation(
    state: FloatArray,
    *,
    team: int,
    field_length: float = 1.5,
    field_width: float = 1.3,
    match_duration: float = 600.0,
    role_assignment: RoleAssignment | None = None,
) -> TeamBatch:
    """Build three agent observations without IDs or identity-ordered entity slots."""
    if team not in (0, 1):
        raise ValueError("team must be 0 (blue) or 1 (yellow)")
    attack_sign = 1.0 if team == 0 else -1.0
    controlled = [slot for slot in range(6) if int(_robot(state, slot)[1]) == team]
    opponents = [slot for slot in range(6) if int(_robot(state, slot)[1]) != team]
    if len(controlled) != TEAM_SIZE or len(opponents) != TEAM_SIZE:
        raise ValueError("canonical state must contain exactly three robots per team")
    robots = {slot: _canonical_robot(_robot(state, slot), attack_sign) for slot in range(6)}
    ball_x = attack_sign * float(state[5])
    ball_y = attack_sign * float(state[6])
    ball_vx = attack_sign * float(state[7])
    ball_vy = attack_sign * float(state[8])
    score_for = float(state[3 if team == 0 else 4])
    score_against = float(state[4 if team == 0 else 3])
    common_context = (
        max(0.0, 1.0 - float(state[2]) / match_duration),
        (score_for - score_against) / 10.0,
        float(bool(int(state[-1]) & 1)),
        float(bool(int(state[-1]) & 2)),
    )
    tactical = role_features(role_assignment or assign_roles(state, team))

    self_rows: list[tuple[float, ...]] = []
    ball_rows: list[tuple[float, ...]] = []
    goal_rows: list[tuple[float, ...]] = []
    teammate_rows: list[list[tuple[float, ...]]] = []
    opponent_rows: list[list[tuple[float, ...]]] = []
    for slot in controlled:
        x, y, cos_theta, sin_theta, vx, vy, omega, left, right, enabled = robots[slot]
        self_rows.append((cos_theta, sin_theta, vx, vy, omega, left, right, enabled))
        dx, dy = ball_x - x, ball_y - y
        dvx, dvy = ball_vx - vx, ball_vy - vy
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - math.atan2(sin_theta, cos_theta)
        ball_rows.append(
            (
                dx / field_length,
                dy / field_width,
                dvx,
                dvy,
                distance / field_length,
                math.cos(bearing),
                math.sin(bearing),
            )
        )
        goal_rows.append(
            (
                (-field_length / 2 - x) / field_length,
                -y / field_width,
                (field_length / 2 - x) / field_length,
                -y / field_width,
            )
        )
        teammate_rows.append(
            [
                _relative_entity(robots[other], robots[slot], field_length, field_width)
                for other in controlled
                if other != slot
            ]
        )
        opponent_rows.append(
            [
                _relative_entity(robots[other], robots[slot], field_length, field_width)
                for other in opponents
            ]
        )
    return TeamBatch(
        torch.tensor(self_rows, dtype=torch.float32),
        torch.tensor(ball_rows, dtype=torch.float32),
        torch.tensor(goal_rows, dtype=torch.float32),
        torch.cat(
            (
                torch.tensor([common_context] * TEAM_SIZE, dtype=torch.float32),
                torch.from_numpy(tactical),
            ),
            dim=-1,
        ),
        torch.tensor(teammate_rows, dtype=torch.float32),
        torch.tensor(opponent_rows, dtype=torch.float32),
    )


def _relative_entity(
    other: tuple[float, ...],
    current: tuple[float, ...],
    field_length: float,
    field_width: float,
) -> tuple[float, ...]:
    other_theta = math.atan2(other[3], other[2])
    current_theta = math.atan2(current[3], current[2])
    delta_theta = other_theta - current_theta
    return (
        (other[0] - current[0]) / field_length,
        (other[1] - current[1]) / field_width,
        other[4] - current[4],
        other[5] - current[5],
        math.cos(delta_theta),
        math.sin(delta_theta),
    )


class AgentEncoder(nn.Module):
    """Deep Sets encoder shared over agents and entity set members."""

    def __init__(
        self,
        hidden_size: int,
        *,
        activation: str = "tanh",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        entity: list[nn.Module] = [nn.Linear(6, hidden_size)]
        if layer_norm:
            entity.append(nn.LayerNorm(hidden_size))
        entity.append(_activation(activation))
        self.entity = nn.Sequential(*entity)
        fusion_input = nn.Linear(8 + 7 + 4 + 4 + 2 * hidden_size, hidden_size)
        fusion: list[nn.Module] = [fusion_input]
        if layer_norm:
            fusion.append(nn.LayerNorm(hidden_size))
        fusion.append(_activation(activation))
        fusion.extend(_hidden_block(hidden_size, activation, layer_norm))
        self.fusion = nn.Sequential(
            *fusion,
        )

    def forward(self, observation: TeamBatch) -> Tensor:
        teammates = self.entity(observation.teammates).mean(dim=-2)
        opponents = self.entity(observation.opponents).mean(dim=-2)
        features = torch.cat(
            (
                observation.self_features,
                observation.ball,
                observation.goals,
                observation.context[..., :4],
                teammates,
                opponents,
            ),
            dim=-1,
        )
        return cast(Tensor, self.fusion(features))


class SharedActor(nn.Module):
    """One decentralized actor applied equivariantly to all team agents."""

    def __init__(
        self,
        hidden_size: int = 64,
        *,
        activation: str = "tanh",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AgentEncoder(
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.action_head = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(self, observation: TeamBatch) -> tuple[Tensor, Tensor]:
        mean = self.action_head(self.encoder(observation))
        return mean, self.log_std.expand_as(mean)

    def deterministic_action(self, observation: TeamBatch) -> Tensor:
        mean, _ = self(observation)
        return torch.tanh(mean)


class RoleSharedActor(nn.Module):
    """Shared actor conditioned on transient responsibility, never robot identity."""

    def __init__(
        self,
        hidden_size: int = 64,
        *,
        activation: str = "tanh",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        entity: list[nn.Module] = [nn.Linear(6, hidden_size)]
        if layer_norm:
            entity.append(nn.LayerNorm(hidden_size))
        entity.append(_activation(activation))
        self.entity = nn.Sequential(*entity)
        fusion: list[nn.Module] = [nn.Linear(8 + 7 + 4 + 9 + 2 * hidden_size, hidden_size)]
        if layer_norm:
            fusion.append(nn.LayerNorm(hidden_size))
        fusion.append(_activation(activation))
        fusion.extend(_hidden_block(hidden_size, activation, layer_norm))
        self.fusion = nn.Sequential(
            *fusion,
        )
        self.action_head = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(self, observation: TeamBatch) -> tuple[Tensor, Tensor]:
        teammates = self.entity(observation.teammates).mean(dim=-2)
        opponents = self.entity(observation.opponents).mean(dim=-2)
        encoded = self.fusion(
            torch.cat(
                (
                    observation.self_features,
                    observation.ball,
                    observation.goals,
                    observation.context,
                    teammates,
                    opponents,
                ),
                dim=-1,
            )
        )
        mean = self.action_head(encoded)
        return mean, self.log_std.expand_as(mean)

    def deterministic_action(self, observation: TeamBatch) -> Tensor:
        mean, _ = self(observation)
        return torch.tanh(mean)


class LocalCritic(nn.Module):
    """Shared local critic used by IPPO."""

    def __init__(
        self,
        hidden_size: int = 64,
        *,
        activation: str = "tanh",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AgentEncoder(
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observation: TeamBatch) -> Tensor:
        return cast(Tensor, self.value_head(self.encoder(observation)).squeeze(-1))


class CentralizedCritic(nn.Module):
    """Permutation-equivariant team critic used only while training MAPPO."""

    def __init__(
        self,
        hidden_size: int = 64,
        *,
        activation: str = "tanh",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AgentEncoder(
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        value: list[nn.Module] = [nn.Linear(2 * hidden_size, hidden_size)]
        if layer_norm:
            value.append(nn.LayerNorm(hidden_size))
        value.extend((_activation(activation), nn.Linear(hidden_size, 1)))
        self.value_head = nn.Sequential(*value)

    def forward(self, observation: TeamBatch) -> Tensor:
        local = self.encoder(observation)
        team = local.mean(dim=-2, keepdim=True).expand_as(local)
        return cast(
            Tensor,
            self.value_head(torch.cat((local, team), dim=-1)).squeeze(-1),
        )
