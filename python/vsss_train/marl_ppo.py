"""Explicit synchronous shared-parameter IPPO and MAPPO losses."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.distributions import Categorical, Normal, VonMises

from vsss_train.ablations import (
    EntityAttentionActor,
    LatticeSharedActor,
    RecurrentSharedActor,
    RecurrentState,
)
from vsss_train.config import MarlConfig
from vsss_train.marl import (
    CentralizedCritic,
    CircularPrimitiveRoleActor,
    LocalCritic,
    ParametricPrimitiveRoleActor,
    PrimitiveRoleActor,
    RoleSharedActor,
    SharedActor,
    TeamBatch,
)
from vsss_train.ppo import seed_everything

MARL_CHECKPOINT_SCHEMA = 1
TRAJECTORY_SCHEMA = 1
PolicyActor = (
    SharedActor
    | RoleSharedActor
    | PrimitiveRoleActor
    | ParametricPrimitiveRoleActor
    | CircularPrimitiveRoleActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor
)
LEGACY_NEUTRAL_CONFIG = {
    "minimum_log_std": -5.0,
    "maximum_log_std": 0.0,
    "wheel_effort_coefficient": 0.0,
    "ball_direction_coefficient": 0.0,
    "useful_touch_impulse_coefficient": 0.0,
    "goal_geometry_coefficient": 0.0,
    "goal_geometry_discount": 0.99,
    "attacker_alignment_coefficient": 0.0,
    "time_penalty_coefficient": 0.0,
    "movement_speed_threshold": 0.03,
    "curriculum_heuristic_iterations": 0,
    "league_self_play_weight": 1.0,
    "league_historical_weight": 0.0,
    "league_heuristic_weight": 0.0,
    "league_history_window": 16,
    "policy_architecture": "mlp",
    "network_activation": "tanh",
    "layer_norm": False,
    "action_parser": "continuous",
    "adaptive_curriculum": False,
    "scenario_suite": "",
    "observation_dropout": 0.0,
    "observation_noise_std": 0.0,
    "goal_coefficient": 10.0,
    "progress_coefficient": 0.0,
    "semantic_curriculum": False,
    "semantic_phased_curriculum": False,
    "semantic_full_match_fraction": 0.25,
    "semantic_terminal_reward": 2.0,
    "semantic_regression_patience": 0,
    "semantic_regression_warmup_evaluations": 0,
    "semantic_phase_patience": 2,
    "semantic_phase_rehearsal_fraction": 0.20,
    "semantic_phase_full_match_floor": 0.0,
    "semantic_promotion_floors": {},
    "semantic_max_idle_spin_ratio": 1.0,
    "semantic_min_match_win_rate": 0.0,
    "semantic_max_match_draw_rate": 1.0,
    "contact_distance": 0.082,
    "contact_grace_seconds": 0.5,
    "ally_deadlock_coefficient": 0.0,
    "opponent_deadlock_coefficient": 0.0,
    "idle_spin_coefficient": 0.0,
    "idle_spin_grace_seconds": 0.5,
    "idle_spin_turn_threshold": 0.13,
    "idle_spin_drive_threshold": 0.07,
    "idle_spin_speed_threshold": 0.08,
    "idle_spin_ball_distance": 0.12,
}
ACTION_EPSILON = 1e-6


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


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


def von_mises_entropy(concentration: Tensor) -> Tensor:
    """Return the differential entropy of a von Mises heading, per element.

    Torch does not implement it, and the naive form overflows for a concentrated
    heading, so this uses the exponentially scaled Bessel functions.
    """
    scaled_zero = torch.special.i0e(concentration)
    ratio = torch.special.i1e(concentration) / scaled_zero
    return cast(
        Tensor,
        -concentration * ratio + torch.log(2.0 * math.pi * scaled_zero) + concentration,
    )


def circular_action_log_prob(
    heading: Tensor,
    concentration: Tensor,
    sampled_heading: Tensor,
) -> Tensor:
    """Evaluate a von Mises heading in the transported angle domain."""
    return cast(
        Tensor,
        VonMises(heading, concentration).log_prob(sampled_heading),  # type: ignore[no-untyped-call]
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
            "bootstrap_value",
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
    bootstrap_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantage = torch.zeros_like(reward)
    estimate = torch.zeros_like(reward[-1])
    for index in range(reward.shape[0] - 1, -1, -1):
        continuation = 1.0 - done[index].to(dtype=reward.dtype)
        next_value = bootstrap_value if index == len(value) - 1 else value[index + 1]
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
            LocalCritic(
                config.hidden_size,
                activation=config.network_activation,
                layer_norm=config.layer_norm,
            )
            if config.algorithm == "ippo"
            else CentralizedCritic(
                config.hidden_size,
                activation=config.network_activation,
                layer_norm=config.layer_norm,
            )
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
                data["bootstrap_value"][-1],
                self.config.gamma,
                self.config.gae_lambda,
            )
            active = observation.self_features[..., -1] > 0.5
            active_advantage = advantage[active]
            advantage = torch.where(
                active,
                (advantage - active_advantage.mean())
                / (active_advantage.std(unbiased=False) + 1e-8),
                torch.zeros_like(advantage),
            )

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        if isinstance(self.actor, CircularPrimitiveRoleActor):
            # Angular precision lives in the concentration, not in log_std, so it has to
            # be reported for exploration to be readable at all.
            totals["heading_concentration"] = 0.0
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
                sample_active = sample_observation.self_features[..., -1] > 0.5
                if isinstance(self.actor, (LatticeSharedActor, PrimitiveRoleActor)):
                    logits, _ = self.actor(sample_observation)
                    distribution_discrete = Categorical(logits=logits)
                    log_probability = distribution_discrete.log_prob(  # type: ignore[no-untyped-call]
                        sample["action_index"]
                    )
                    ratio = (log_probability - sample["sample_log_prob"]).exp()
                    entropy = _masked_mean(
                        distribution_discrete.entropy(),  # type: ignore[no-untyped-call]
                        sample_active,
                    )
                elif isinstance(self.actor, ParametricPrimitiveRoleActor):
                    skill_logits, parameter_mean, parameter_log_std = self.actor(sample_observation)
                    skill_distribution = Categorical(logits=skill_logits)
                    parameter_distribution = Normal(
                        parameter_mean,
                        parameter_log_std.exp(),
                    )
                    log_probability = (
                        skill_distribution.log_prob(  # type: ignore[no-untyped-call]
                            sample["action_index"]
                        )
                        + bounded_action_log_prob(
                            parameter_distribution,
                            sample["action"][..., 1:],
                        )
                    )
                    ratio = (log_probability - sample["sample_log_prob"]).exp()
                    entropy_action = torch.tanh(parameter_distribution.rsample())
                    parameter_entropy = -bounded_action_log_prob(
                        parameter_distribution,
                        entropy_action,
                    )
                    entropy = _masked_mean(
                        skill_distribution.entropy() + parameter_entropy,  # type: ignore[no-untyped-call]
                        sample_active,
                    )
                elif isinstance(self.actor, CircularPrimitiveRoleActor):
                    (
                        skill_logits,
                        heading,
                        concentration,
                        intensity_mean,
                        intensity_log_std,
                    ) = self.actor(sample_observation)
                    skill_distribution = Categorical(logits=skill_logits)
                    intensity_distribution = Normal(intensity_mean, intensity_log_std.exp())
                    # The transported heading is the sampled angle itself, so it is scored
                    # directly with no change of variables to invert.
                    sampled_heading = sample["action"][..., 1] * math.pi
                    log_probability = (
                        skill_distribution.log_prob(  # type: ignore[no-untyped-call]
                            sample["action_index"]
                        )
                        + circular_action_log_prob(heading, concentration, sampled_heading)
                        + bounded_action_log_prob(
                            intensity_distribution,
                            sample["action"][..., 2:],
                        )
                    )
                    ratio = (log_probability - sample["sample_log_prob"]).exp()
                    intensity_entropy = -bounded_action_log_prob(
                        intensity_distribution,
                        torch.tanh(intensity_distribution.rsample()),
                    )
                    entropy = _masked_mean(
                        skill_distribution.entropy()  # type: ignore[no-untyped-call]
                        + von_mises_entropy(concentration)
                        + intensity_entropy,
                        sample_active,
                    )
                    totals["heading_concentration"] += float(
                        _masked_mean(concentration, sample_active).detach()
                    )
                elif isinstance(self.actor, RecurrentSharedActor):
                    mean, log_std, _ = self.actor.forward_with_state(
                        sample_observation,
                        RecurrentState(sample["recurrent_hidden"]),
                    )
                else:
                    mean, log_std = self.actor(sample_observation)
                if not isinstance(
                    self.actor,
                    (
                        LatticeSharedActor,
                        PrimitiveRoleActor,
                        ParametricPrimitiveRoleActor,
                        CircularPrimitiveRoleActor,
                    ),
                ):
                    distribution = Normal(mean, log_std.exp())
                    log_probability = bounded_action_log_prob(distribution, sample["action"])
                    ratio = (log_probability - sample["sample_log_prob"]).exp()
                    entropy_action = torch.tanh(distribution.rsample())
                    entropy = -_masked_mean(
                        bounded_action_log_prob(distribution, entropy_action),
                        sample_active,
                    )
                clipped = ratio.clamp(
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                policy_loss = -_masked_mean(
                    torch.minimum(
                        ratio * sample_advantage,
                        clipped * sample_advantage,
                    ),
                    sample_active,
                )
                value = self.critic(sample_observation)
                value_loss = 0.5 * _masked_mean(
                    (value - value_target[indices]).square(),
                    sample_active,
                )
                log_ratio = log_probability - sample["sample_log_prob"]
                approx_kl = _masked_mean((ratio - 1.0) - log_ratio, sample_active)
                clip_fraction = _masked_mean(
                    ((ratio - 1.0).abs() > self.config.clip_epsilon).float(),
                    sample_active,
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(
                    (*self.actor.parameters(), *self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                with torch.no_grad():
                    if not isinstance(self.actor, (LatticeSharedActor, PrimitiveRoleActor)):
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
        metric_action = (
            data["action"][..., 1:]
            if self.config.action_parser in ("parametric_primitive", "circular_primitive")
            else data["action"]
        )
        # Absent roster slots still emit tokens the parser discards, so reporting them
        # would average untrained noise from robots that are not on the field.
        metric_action = metric_action[observation.self_features[..., -1] > 0.5]
        result["mean_abs_action"] = float(metric_action.abs().mean())
        result["action_saturation"] = float((metric_action.abs() > 0.95).float().mean())
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
        architectural = (
            "hidden_size",
            "policy_architecture",
            "action_parser",
            "network_activation",
            "layer_norm",
        )
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
    if config.action_parser == "primitive":
        return PrimitiveRoleActor(
            config.hidden_size,
            activation=config.network_activation,
            layer_norm=config.layer_norm,
        )
    if config.action_parser == "circular_primitive":
        return CircularPrimitiveRoleActor(
            config.hidden_size,
            activation=config.network_activation,
            layer_norm=config.layer_norm,
        )
    if config.action_parser == "parametric_primitive":
        return ParametricPrimitiveRoleActor(
            config.hidden_size,
            activation=config.network_activation,
            layer_norm=config.layer_norm,
        )
    if config.policy_architecture == "gru":
        return RecurrentSharedActor(config.hidden_size)
    if config.policy_architecture == "attention":
        return EntityAttentionActor(config.hidden_size)
    if config.policy_architecture == "role_mlp":
        return RoleSharedActor(
            config.hidden_size,
            activation=config.network_activation,
            layer_norm=config.layer_norm,
        )
    return SharedActor(
        config.hidden_size,
        activation=config.network_activation,
        layer_norm=config.layer_norm,
    )
