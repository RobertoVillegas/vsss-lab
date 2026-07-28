"""Versioned M5 training configuration."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
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
    seed: int = 7
    hidden_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 1e-3
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256
    policy_id: str = "blue-shared"
    curriculum_stage: Literal[7, 8] = 7
    horizon: int = 1_000
    action_repeat: int = 4

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported MARL config schema")
        if self.algorithm not in ("ippo", "mappo"):
            raise ValueError("algorithm must be ippo or mappo")
        if self.curriculum_stage not in (7, 8):
            raise ValueError("curriculum_stage must be 7 or 8")

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
