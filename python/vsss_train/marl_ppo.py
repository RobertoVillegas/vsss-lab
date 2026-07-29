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
from torch.distributions import Categorical, Normal

from vsss_train.ablations import (
    EntityAttentionActor,
    LatticeSharedActor,
    RecurrentSharedActor,
    RecurrentState,
)
from vsss_train.config import MarlConfig
from vsss_train.marl import CentralizedCritic, LocalCritic, RoleSharedActor, SharedActor, TeamBatch
from vsss_train.ppo import seed_everything

MARL_CHECKPOINT_SCHEMA = 1
TRAJECTORY_SCHEMA = 1
PolicyActor = (
    SharedActor | RoleSharedActor | RecurrentSharedActor | EntityAttentionActor | LatticeSharedActor
)
LEGACY_NEUTRAL_CONFIG = {
    "minimum_log_std": -5.0,
    "maximum_log_std": 0.0,
    "wheel_effort_coefficient": 0.0,
    "ball_direction_coefficient": 0.0,
    "attacker_alignment_coefficient": 0.0,
    "time_penalty_coefficient": 0.0,
    "movement_speed_threshold": 0.03,
    "curriculum_heuristic_iterations": 0,
    "league_self_play_weight": 1.0,
    "league_historical_weight": 0.0,
    "league_heuristic_weight": 0.0,
    "league_history_window": 16,
    "policy_architecture": "mlp",
    "action_parser": "continuous",
    "adaptive_curriculum": False,
    "scenario_suite": "",
    "observation_dropout": 0.0,
    "observation_noise_std": 0.0,
    "goal_coefficient": 10.0,
    "progress_coefficient": 0.0,
    "semantic_curriculum": False,
    "semantic_full_match_fraction": 0.25,
    "semantic_terminal_reward": 2.0,
    "semantic_regression_patience": 0,
    "semantic_regression_warmup_evaluations": 0,
}
ACTION_EPSILON = 1e-6


def sample_bounded_action(distribution: Normal) -> tuple[Tensor, Tensor]:
    """Sample a tanh-bounded action and its transformed log probability."""
    latent = distribution.sample()  # type: ignore[no-untyped-call]
    action = torch.tanh(latent)
    return action, bounded_action_log_prob(distribution, action)


def bounded_action_log_prob(distribution: Normal, action: Tensor) -> Tensor:
    """Evaluate a tanh-transformed Gaussian in the action domain."""
    bounded = action.clamp(-1.0 + ACTION_EPSILON, 1.0 - ACTION_EPSILON)
    latent = torch.atanh(bounded)
    correction = torch.log1p(-bounded.square() + ACTION_EPSILON)
    return cast(
        Tensor,
        (distribution.log_prob(latent) - correction).sum(-1),  # type: ignore[no-untyped-call]
    )


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
        self.device = resolve_device(config.device)
        self.actor = _build_actor(config).to(self.device)
        self.critic: LocalCritic | CentralizedCritic
        self.critic = (
            LocalCritic(config.hidden_size)
            if config.algorithm == "ippo"
            else CentralizedCritic(config.hidden_size)
        ).to(self.device)
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

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        steps = 0
        generator = torch.Generator().manual_seed(self.config.seed + self.policy_version)
        agents_per_step = self.config.num_envs * 3
        time_steps_per_batch = max(1, self.config.minibatch_size // agents_per_step)
        for _ in range(self.config.epochs):
            permutation = torch.randperm(len(data), generator=generator)
            for indices in permutation.split(time_steps_per_batch):  # type: ignore[no-untyped-call]
                indices = indices.to(self.device)
                sample = data[indices]
                sample_observation = observation.select_batch(indices)
                sample_advantage = advantage[indices]
                if isinstance(self.actor, LatticeSharedActor):
                    logits, _ = self.actor(sample_observation)
                    distribution_discrete = Categorical(logits=logits)
                    log_probability = distribution_discrete.log_prob(  # type: ignore[no-untyped-call]
                        sample["action_index"]
                    )
                    ratio = (log_probability - sample["sample_log_prob"]).exp()
                    entropy = distribution_discrete.entropy().mean()  # type: ignore[no-untyped-call]
                elif isinstance(self.actor, RecurrentSharedActor):
                    mean, log_std, _ = self.actor.forward_with_state(
                        sample_observation,
                        RecurrentState(sample["recurrent_hidden"]),
                    )
                else:
                    mean, log_std = self.actor(sample_observation)
                if not isinstance(self.actor, LatticeSharedActor):
                    distribution = Normal(mean, log_std.exp())
                    log_probability = bounded_action_log_prob(distribution, sample["action"])
                    ratio = (log_probability - sample["sample_log_prob"]).exp()
                    entropy_action = torch.tanh(distribution.rsample())
                    entropy = -bounded_action_log_prob(distribution, entropy_action).mean()
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
                log_ratio = log_probability - sample["sample_log_prob"]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()
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
                with torch.no_grad():
                    self.actor.log_std.clamp_(
                        min=self.config.minimum_log_std,
                        max=self.config.maximum_log_std,
                    )
                totals["policy_loss"] += float(policy_loss.detach())
                totals["value_loss"] += float(value_loss.detach())
                totals["entropy"] += float(entropy.detach())
                totals["approx_kl"] += float(approx_kl.detach())
                totals["clip_fraction"] += float(clip_fraction.detach())
                steps += 1
        result = {name: value / steps for name, value in totals.items()}
        result["mean_abs_action"] = float(data["action"].abs().mean())
        result["action_saturation"] = float((data["action"].abs() > 0.95).float().mean())
        self.policy_version += 1
        return result

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
                "torch_cuda_rng": (
                    torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
                ),
            },
            path,
        )

    def initialize_policy(self, path: Path) -> int:
        """Warm-start actor and critic while retaining a fresh optimizer and RNG."""
        payload = cast(
            dict[str, Any],
            torch.load(path, map_location="cpu", weights_only=True),
        )
        if payload.get("schema_version") != MARL_CHECKPOINT_SCHEMA:
            raise ValueError("incompatible MARL warm-start checkpoint schema")
        if payload.get("algorithm") != self.config.algorithm:
            raise ValueError("MARL warm-start checkpoint algorithm mismatch")
        stored = payload.get("config")
        if not isinstance(stored, dict):
            raise ValueError("MARL warm-start checkpoint lacks configuration")
        current = asdict(self.config)
        architectural = ("hidden_size", "policy_architecture", "action_parser")
        mismatch = [
            key
            for key in architectural
            if stored.get(key, LEGACY_NEUTRAL_CONFIG.get(key)) != current[key]
        ]
        if mismatch:
            raise ValueError(
                f"MARL warm-start architecture mismatch: {', '.join(sorted(mismatch))}"
            )
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.policy_version = 0
        return int(payload["policy_version"])

    def load(self, path: Path) -> None:
        payload = _load_checkpoint_payload(path, self.config)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.optimizer.load_state_dict(payload["optimizer"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(self.device)
        self.policy_version = int(payload["policy_version"])
        random.setstate(payload["python_rng"])
        numpy_rng = payload["numpy_rng"]
        np.random.set_state(
            (numpy_rng[0], numpy_rng[1].numpy(), numpy_rng[2], numpy_rng[3], numpy_rng[4])
        )
        torch.set_rng_state(payload["torch_rng"])
        if self.device.type == "cuda" and payload.get("torch_cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng"])


def load_policy_actor(
    path: Path,
    config: MarlConfig,
    device: torch.device,
) -> tuple[PolicyActor, int]:
    """Load an inference-only historical actor without mutating trainer RNG state."""
    payload = _load_checkpoint_payload(path, config)
    with torch.random.fork_rng():
        actor = _build_actor(config).to(device)
    actor.load_state_dict(payload["actor"])
    return actor.eval(), int(payload["policy_version"])


def _load_checkpoint_payload(path: Path, config: MarlConfig) -> dict[str, Any]:
    payload = cast(
        dict[str, Any],
        torch.load(path, map_location="cpu", weights_only=True),
    )
    if payload.get("schema_version") != MARL_CHECKPOINT_SCHEMA:
        raise ValueError("incompatible MARL checkpoint schema")
    if payload.get("algorithm") != config.algorithm:
        raise ValueError("MARL checkpoint algorithm mismatch")
    if not _checkpoint_config_compatible(payload, config):
        raise ValueError("MARL checkpoint configuration fingerprint mismatch")
    return payload


def _checkpoint_config_compatible(payload: dict[str, Any], config: MarlConfig) -> bool:
    if payload.get("config_fingerprint") == config.fingerprint():
        return True
    stored = payload.get("config")
    if not isinstance(stored, dict):
        return False
    current = asdict(config)
    if any(key not in current or current[key] != value for key, value in stored.items()):
        return False
    missing = set(current) - set(stored)
    return all(
        key in LEGACY_NEUTRAL_CONFIG and current[key] == LEGACY_NEUTRAL_CONFIG[key]
        for key in missing
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda without silently ignoring an explicit CUDA request."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_actor(config: MarlConfig) -> PolicyActor:
    if config.action_parser == "lattice":
        return LatticeSharedActor(config.hidden_size)
    if config.policy_architecture == "gru":
        return RecurrentSharedActor(config.hidden_size)
    if config.policy_architecture == "attention":
        return EntityAttentionActor(config.hidden_size)
    if config.policy_architecture == "role_mlp":
        return RoleSharedActor(config.hidden_size)
    return SharedActor(config.hidden_size)
