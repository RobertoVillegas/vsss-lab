"""Explicit policy-memory, entity-attention, and action-parser ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from vsss_train.marl import AgentEncoder, TeamBatch


@dataclass
class RecurrentState:
    """Hidden state indexed explicitly by [world, agent, feature]."""

    hidden: Tensor

    @classmethod
    def zeros(
        cls,
        *,
        worlds: int,
        agents: int,
        hidden_size: int,
        device: torch.device | None = None,
    ) -> RecurrentState:
        return cls(torch.zeros(worlds, agents, hidden_size, device=device))

    def reset_worlds(self, done: Tensor) -> None:
        if done.ndim != 1 or done.shape[0] != self.hidden.shape[0]:
            raise ValueError("done mask must contain one value per world")
        self.hidden[done] = 0.0


class RecurrentSharedActor(nn.Module):
    """GRU actor whose state cannot leak across vectorized worlds."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = AgentEncoder(hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def forward(
        self,
        observation: TeamBatch,
        state: RecurrentState,
    ) -> tuple[Tensor, Tensor, RecurrentState]:
        encoded = self.encoder(observation)
        if encoded.shape[:-1] != state.hidden.shape[:-1]:
            raise ValueError("recurrent state does not match world and agent axes")
        hidden = self.cell(
            encoded.reshape(-1, self.hidden_size),
            state.hidden.reshape(-1, self.hidden_size),
        ).reshape_as(state.hidden)
        mean = self.action_head(hidden)
        return mean, self.log_std.expand_as(mean), RecurrentState(hidden)


class EntityAttentionActor(nn.Module):
    """Batched attention over teammate/opponent entities with visible telemetry."""

    def __init__(self, hidden_size: int = 64, heads: int = 4) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by attention heads")
        self.entity = nn.Linear(6, hidden_size)
        self.query = nn.Linear(8 + 7 + 4 + 4, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.latest_attention: Tensor | None = None

    def forward(self, observation: TeamBatch) -> tuple[Tensor, Tensor]:
        common = torch.cat(
            (
                observation.self_features,
                observation.ball,
                observation.goals,
                observation.context,
            ),
            dim=-1,
        )
        query = self.query(common).unsqueeze(-2)
        entities = self.entity(torch.cat((observation.teammates, observation.opponents), dim=-2))
        prefix = query.shape[:-2]
        flat_query = query.reshape(-1, 1, query.shape[-1])
        flat_entities = entities.reshape(-1, entities.shape[-2], entities.shape[-1])
        attended, weights = self.attention(
            flat_query,
            flat_entities,
            flat_entities,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = attended.reshape(*prefix, 1, attended.shape[-1])
        self.latest_attention = weights.reshape(*prefix, *weights.shape[1:]).detach()
        fused = self.fusion(torch.cat((query.squeeze(-2), attended.squeeze(-2)), dim=-1))
        mean = self.action_head(fused)
        return mean, self.log_std.expand_as(mean)


class SymmetricWheelLattice:
    """Small symmetric differential-drive action abstraction."""

    values: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (0.5, 1.0),
        (1.0, 0.5),
        (-0.5, -1.0),
        (-1.0, -0.5),
    )

    def parse(self, actions: Tensor) -> Tensor:
        if actions.dtype not in (torch.int32, torch.int64):
            raise ValueError("lattice actions must be integer indices")
        if bool(torch.any(actions < 0)) or bool(torch.any(actions >= len(self.values))):
            raise ValueError("lattice action index out of range")
        table = torch.tensor(self.values, dtype=torch.float32, device=actions.device)
        return table[actions]
