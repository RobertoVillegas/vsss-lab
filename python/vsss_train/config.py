"""Versioned M5 training configuration."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TrainConfig:
    schema_version: int = 1
    seed: int = 7
    device: str = "cpu"
    hidden_size: int = 64
    warmup_samples: int = 32_768
    warmup_epochs: int = 20
    rollout_steps: int = 4_096
    updates: int = 10
    epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 1e-3
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    eval_episodes: int = 20
    eval_every: int = 5
    initial_stage: int = 0
    max_episode_steps: int = 3_000
    success_radius: float = 0.09

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported training config schema")
        if not 0 <= self.initial_stage <= 5:
            raise ValueError("initial_stage must be in [0, 5]")
        positive = (
            self.hidden_size,
            self.warmup_samples,
            self.warmup_epochs,
            self.rollout_steps,
            self.updates,
            self.epochs,
            self.minibatch_size,
            self.eval_episodes,
            self.eval_every,
            self.max_episode_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("training counts must be positive")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: str | Path) -> TrainConfig:
    """Load a strict versioned TOML configuration."""
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    unknown = set(document) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown training config keys: {sorted(unknown)}")
    return TrainConfig(**document)


def config_dict(config: TrainConfig) -> dict[str, Any]:
    """Return a JSON-safe config mapping."""
    return asdict(config)


@dataclass(frozen=True)
class MarlConfig:
    """Versioned M6 shared-policy training configuration."""

    schema_version: int = 1
    algorithm: Literal["ippo", "mappo"] = "mappo"
    policy_architecture: Literal["mlp", "gru", "attention", "role_mlp"] = "mlp"
    action_parser: Literal["continuous", "lattice", "primitive"] = "continuous"
    adaptive_curriculum: bool = False
    semantic_curriculum: bool = False
    semantic_phased_curriculum: bool = False
    scenario_suite: str = ""
    semantic_full_match_fraction: float = 0.25
    semantic_terminal_reward: float = 2.0
    semantic_regression_patience: int = 0
    semantic_regression_warmup_evaluations: int = 0
    semantic_phase_patience: int = 2
    semantic_phase_rehearsal_fraction: float = 0.20
    semantic_phase_full_match_floor: float = 0.0
    semantic_promotion_floors: dict[str, float] = field(default_factory=dict)
    semantic_max_idle_spin_ratio: float = 1.0
    semantic_min_match_win_rate: float = 0.0
    semantic_max_match_draw_rate: float = 1.0
    observation_dropout: float = 0.0
    observation_noise_std: float = 0.0
    seed: int = 7
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_envs: int = 64
    hidden_size: int = 64
    network_activation: Literal["tanh", "relu"] = "tanh"
    layer_norm: bool = False
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 1e-3
    minimum_log_std: float = -5.0
    maximum_log_std: float = 0.0
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256
    policy_id: str = "blue-shared"
    curriculum_stage: Literal[7, 8] = 7
    horizon: int = 1_500
    rollout_steps: int = 256
    action_repeat: int = 4
    action_delta_coefficient: float = 0.01
    goal_coefficient: float = 10.0
    progress_coefficient: float = 0.0
    wheel_effort_coefficient: float = 0.0
    ball_direction_coefficient: float = 0.0
    useful_touch_impulse_coefficient: float = 0.0
    goal_geometry_coefficient: float = 0.0
    goal_geometry_discount: float = 0.99
    idle_spin_coefficient: float = 0.0
    idle_spin_grace_seconds: float = 0.5
    idle_spin_turn_threshold: float = 0.13
    idle_spin_drive_threshold: float = 0.07
    idle_spin_speed_threshold: float = 0.08
    idle_spin_ball_distance: float = 0.12
    attacker_alignment_coefficient: float = 0.0
    time_penalty_coefficient: float = 0.0
    movement_speed_threshold: float = 0.03
    teammate_spacing: float = 0.14
    teammate_congestion_coefficient: float = 0.002
    contact_distance: float = 0.082
    contact_grace_seconds: float = 0.5
    ally_deadlock_coefficient: float = 0.0
    opponent_deadlock_coefficient: float = 0.0
    defensive_coverage_coefficient: float = 1.0
    defensive_activation_x: float = 0.15
    draw_penalty: float = 0.25
    stagnation_penalty: float = 0.10
    stagnation_seconds: float = 5.0
    stagnation_ball_distance: float = 0.02
    curriculum_heuristic_iterations: int = 0
    league_self_play_weight: float = 1.0
    league_historical_weight: float = 0.0
    league_heuristic_weight: float = 0.0
    league_history_window: int = 16

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported MARL config schema")
        if self.algorithm not in ("ippo", "mappo"):
            raise ValueError("algorithm must be ippo or mappo")
        if self.policy_architecture not in ("mlp", "gru", "attention", "role_mlp"):
            raise ValueError("policy_architecture must be mlp, gru, attention, or role_mlp")
        if self.action_parser not in ("continuous", "lattice", "primitive"):
            raise ValueError("action_parser must be continuous, lattice, or primitive")
        if self.network_activation not in ("tanh", "relu"):
            raise ValueError("network_activation must be tanh or relu")
        if self.action_parser == "lattice" and self.policy_architecture != "mlp":
            raise ValueError("lattice ablation currently requires the MLP architecture")
        if self.action_parser == "primitive" and self.policy_architecture != "role_mlp":
            raise ValueError("primitive actions require the role_mlp architecture")
        if self.adaptive_curriculum and not self.scenario_suite:
            raise ValueError("adaptive_curriculum requires scenario_suite")
        if self.adaptive_curriculum and self.semantic_curriculum:
            raise ValueError("static and semantic curricula are mutually exclusive")
        if self.semantic_phased_curriculum and not self.semantic_curriculum:
            raise ValueError("phased curriculum requires semantic_curriculum=true")
        if not 0.0 <= self.semantic_full_match_fraction <= 1.0:
            raise ValueError("semantic_full_match_fraction must be in [0, 1]")
        if self.semantic_terminal_reward < 0.0:
            raise ValueError("semantic_terminal_reward must be non-negative")
        if self.semantic_regression_patience < 0:
            raise ValueError("semantic_regression_patience must be non-negative")
        if self.semantic_regression_warmup_evaluations < 0:
            raise ValueError("semantic_regression_warmup_evaluations must be non-negative")
        if self.semantic_phase_patience <= 0:
            raise ValueError("semantic_phase_patience must be positive")
        if not 0.0 <= self.semantic_phase_rehearsal_fraction <= 1.0:
            raise ValueError("semantic_phase_rehearsal_fraction must be in [0, 1]")
        if not 0.0 <= self.semantic_phase_full_match_floor <= 1.0:
            raise ValueError("semantic_phase_full_match_floor must be in [0, 1]")
        known_families = {
            "approach",
            "clearance",
            "interception",
            "pass_receive",
            "rotation_recovery",
            "save_deflection",
            "shot",
        }
        if set(self.semantic_promotion_floors) - known_families:
            raise ValueError("semantic promotion floor has an unknown skill family")
        if any(not 0.0 <= value <= 1.0 for value in self.semantic_promotion_floors.values()):
            raise ValueError("semantic promotion floors must be in [0, 1]")
        if not 0.0 <= self.semantic_max_idle_spin_ratio <= 1.0:
            raise ValueError("semantic_max_idle_spin_ratio must be in [0, 1]")
        if not 0.0 <= self.semantic_min_match_win_rate <= 1.0:
            raise ValueError("semantic_min_match_win_rate must be in [0, 1]")
        if not 0.0 <= self.semantic_max_match_draw_rate <= 1.0:
            raise ValueError("semantic_max_match_draw_rate must be in [0, 1]")
        if not 0.0 <= self.observation_dropout < 1.0:
            raise ValueError("observation_dropout must be in [0, 1)")
        if self.observation_noise_std < 0.0:
            raise ValueError("observation_noise_std must be non-negative")
        if self.curriculum_stage not in (7, 8):
            raise ValueError("curriculum_stage must be 7 or 8")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.horizon <= 0 or self.rollout_steps <= 0:
            raise ValueError("horizon and rollout_steps must be positive")
        non_negative = (
            self.action_delta_coefficient,
            self.goal_coefficient,
            self.progress_coefficient,
            self.wheel_effort_coefficient,
            self.ball_direction_coefficient,
            self.useful_touch_impulse_coefficient,
            self.goal_geometry_coefficient,
            self.idle_spin_coefficient,
            self.attacker_alignment_coefficient,
            self.time_penalty_coefficient,
            self.teammate_congestion_coefficient,
            self.ally_deadlock_coefficient,
            self.opponent_deadlock_coefficient,
            self.defensive_coverage_coefficient,
            self.draw_penalty,
            self.stagnation_penalty,
            self.league_self_play_weight,
            self.league_historical_weight,
            self.league_heuristic_weight,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("MARL reward coefficients must be non-negative")
        if not 0.0 <= self.goal_geometry_discount <= 1.0:
            raise ValueError("goal_geometry_discount must be in [0, 1]")
        if self.idle_spin_grace_seconds <= 0.0:
            raise ValueError("idle_spin_grace_seconds must be positive")
        if not 0.0 <= self.idle_spin_turn_threshold <= 1.0:
            raise ValueError("idle_spin_turn_threshold must be in [0, 1]")
        if not 0.0 <= self.idle_spin_drive_threshold <= 1.0:
            raise ValueError("idle_spin_drive_threshold must be in [0, 1]")
        if self.idle_spin_speed_threshold <= 0.0 or self.idle_spin_ball_distance <= 0.0:
            raise ValueError("idle-spin motion thresholds must be positive")
        if self.teammate_spacing <= 0.075:
            raise ValueError("teammate_spacing must exceed the robot body width")
        if self.contact_distance <= 0.075:
            raise ValueError("contact_distance must exceed the robot body width")
        if self.contact_grace_seconds <= 0.0:
            raise ValueError("contact_grace_seconds must be positive")
        if (
            self.stagnation_seconds <= 0.0
            or self.stagnation_ball_distance <= 0.0
            or self.movement_speed_threshold <= 0.0
        ):
            raise ValueError("stagnation thresholds must be positive")
        if self.minimum_log_std > 0.0:
            raise ValueError("minimum_log_std must not be positive")
        if self.maximum_log_std < self.minimum_log_std:
            raise ValueError("maximum_log_std must not be below minimum_log_std")
        if self.curriculum_heuristic_iterations < 0:
            raise ValueError("curriculum_heuristic_iterations must be non-negative")
        if (
            self.league_self_play_weight
            + self.league_historical_weight
            + self.league_heuristic_weight
            <= 0.0
        ):
            raise ValueError("at least one league opponent weight must be positive")
        if self.league_history_window <= 0:
            raise ValueError("league_history_window must be positive")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_marl_config(path: str | Path) -> MarlConfig:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    unknown = set(document) - set(MarlConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown MARL config keys: {sorted(unknown)}")
    return MarlConfig(**document)
