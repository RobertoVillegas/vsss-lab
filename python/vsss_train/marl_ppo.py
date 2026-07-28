"""Explicit synchronous shared-parameter IPPO and MAPPO losses."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.distributions import Normal

from vsss_train.config import MarlConfig
from vsss_train.marl import CentralizedCritic, LocalCritic, SharedActor, TeamBatch
from vsss_train.ppo import seed_everything

MARL_CHECKPOINT_SCHEMA = 1
TRAJECTORY_SCHEMA = 1


@dataclass(frozen=True)
class TrajectoryMetadata:
    schema_version: int
    run_id: str
    episode_id: int
    world_id: int
    team: int
    policy_id: str
    policy_version: int
    global_state_ref: str


@dataclass(frozen=True)
class TeamTrajectory:
    metadata: TrajectoryMetadata
    data: TensorDict

    def validate(self) -> None:
        if self.metadata.schema_version != TRAJECTORY_SCHEMA:
            raise ValueError("incompatible trajectory schema")
        required = {
            "tick",
            "self_features",
            "ball",
            "goals",
            "context",
            "teammates",
            "opponents",
            "action",
            "sample_log_prob",
            "reward_total",
            "terminated",
            "truncated",
            "state_value",
        }
        missing = required - set(self.data.keys())
        if missing:
            raise ValueError(f"trajectory missing fields: {sorted(missing)}")


def observation_from_trajectory(data: TensorDict) -> TeamBatch:
    return TeamBatch(
        data["self_features"],
        data["ball"],
        data["goals"],
        data["context"],
        data["teammates"],
        data["opponents"],
    )


def _team_gae(
    reward: Tensor,
    value: Tensor,
    done: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantage = torch.zeros_like(reward)
    estimate = torch.zeros_like(reward[-1])
    for index in range(reward.shape[0] - 1, -1, -1):
        continuation = 1.0 - done[index]
        next_value = torch.zeros_like(value[index]) if index == len(value) - 1 else value[index + 1]
        delta = reward[index] + gamma * next_value * continuation - value[index]
        estimate = delta + gamma * gae_lambda * continuation * estimate
        advantage[index] = estimate
    return advantage, advantage + value


class MarlLearner:
    """One synchronous learner supporting IPPO and MAPPO critic variants."""

    def __init__(self, config: MarlConfig) -> None:
        self.config = config
        seed_everything(config.seed)
        self.actor = SharedActor(config.hidden_size)
        self.critic: LocalCritic | CentralizedCritic
        self.critic = (
            LocalCritic(config.hidden_size)
            if config.algorithm == "ippo"
            else CentralizedCritic(config.hidden_size)
        )
        self.optimizer = torch.optim.Adam(
            (*self.actor.parameters(), *self.critic.parameters()),
            lr=config.learning_rate,
        )
        self.policy_version = 0

    def optimize(self, trajectory: TeamTrajectory) -> dict[str, float]:
        trajectory.validate()
        if trajectory.metadata.policy_id != self.config.policy_id:
            raise ValueError("trajectory policy ID mismatch")
        if trajectory.metadata.policy_version != self.policy_version:
            raise ValueError("stale trajectory policy version")
        data = trajectory.data
        observation = observation_from_trajectory(data)
        with torch.no_grad():
            advantage, value_target = _team_gae(
                data["reward_total"],
                data["state_value"],
                (data["terminated"] | data["truncated"]).float(),
                self.config.gamma,
                self.config.gae_lambda,
            )
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        steps = 0
        generator = torch.Generator().manual_seed(self.config.seed + self.policy_version)
        time_steps_per_batch = max(1, self.config.minibatch_size // 3)
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(data), generator=generator)
            for indices in permutation.split(time_steps_per_batch):  # type: ignore[no-untyped-call]
                sample = data[indices]
                sample_observation = observation.select_batch(indices)
                sample_advantage = advantage[indices]
                mean, log_std = self.actor(sample_observation)
                distribution = Normal(mean, log_std.exp())
                log_probability = distribution.log_prob(sample["action"]).sum(-1)  # type: ignore[no-untyped-call]
                ratio = (log_probability - sample["sample_log_prob"]).exp()
                clipped = ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                policy_loss = -torch.minimum(
                    ratio * sample_advantage,
                    clipped * sample_advantage,
                ).mean()
                value = self.critic(sample_observation)
                value_loss = 0.5 * (value - value_target[indices]).square().mean()
                entropy = distribution.entropy().sum(-1).mean()  # type: ignore[no-untyped-call]
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (*self.actor.parameters(), *self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach())
                totals["value_loss"] += float(value_loss.detach())
                totals["entropy"] += float(entropy.detach())
                steps += 1
        self.policy_version += 1
        return {name: value / steps for name, value in totals.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = cast(
            tuple[str, Any, int, int, float],
            np.random.get_state(),
        )
        torch.save(
            {
                "schema_version": MARL_CHECKPOINT_SCHEMA,
                "algorithm": self.config.algorithm,
                "config": asdict(self.config),
                "config_fingerprint": self.config.fingerprint(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "policy_version": self.policy_version,
                "curriculum_stage": self.config.curriculum_stage,
                "python_rng": random.getstate(),
                "numpy_rng": (
                    numpy_name,
                    torch.from_numpy(numpy_keys.copy()),
                    numpy_position,
                    numpy_has_gauss,
                    numpy_cached,
                ),
                "torch_rng": torch.get_rng_state(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != MARL_CHECKPOINT_SCHEMA:
            raise ValueError("incompatible MARL checkpoint schema")
        if payload.get("algorithm") != self.config.algorithm:
            raise ValueError("MARL checkpoint algorithm mismatch")
        if payload.get("config_fingerprint") != self.config.fingerprint():
            raise ValueError("MARL checkpoint configuration fingerprint mismatch")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.policy_version = int(payload["policy_version"])
        random.setstate(payload["python_rng"])
        numpy_rng = payload["numpy_rng"]
        np.random.set_state(
            (numpy_rng[0], numpy_rng[1].numpy(), numpy_rng[2], numpy_rng[3], numpy_rng[4])
        )
        torch.set_rng_state(payload["torch_rng"])
