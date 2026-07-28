"""Small explicit PPO lifecycle for the M5 skill gate."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor, nn
from torch.distributions import Normal

from vsss_train.config import TrainConfig
from vsss_train.task import STAGES, GoToTargetEnv

CHECKPOINT_SCHEMA = 1
METRICS_SCHEMA = 1


class ActorCritic(nn.Module):
    """Shared encoder with Gaussian actor and scalar critic."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.encoder(observation)
        mean = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value

    def act(
        self, observation: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std, value = self(observation)
        distribution = Normal(mean, log_std.exp())
        raw_action = mean if deterministic else distribution.sample()  # type: ignore[no-untyped-call]
        log_probability = distribution.log_prob(raw_action)  # type: ignore[no-untyped-call]
        return raw_action, log_probability.sum(-1), value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def warm_start(model: ActorCritic, config: TrainConfig, device: torch.device) -> float:
    """Fit the M4 geometric skill before on-policy curriculum refinement."""
    generator = torch.Generator(device=device).manual_seed(config.seed)
    target_dx = torch.empty(config.warmup_samples, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )
    target_dy = torch.empty(config.warmup_samples, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )
    theta = torch.empty(config.warmup_samples, device=device).uniform_(
        -torch.pi, torch.pi, generator=generator
    )
    zeros = torch.zeros_like(theta)
    observation = torch.stack(
        (target_dx / 1.5, target_dy / 1.3, theta.cos(), theta.sin(), zeros, zeros, zeros),
        dim=-1,
    )
    desired = torch.atan2(target_dy, target_dx)
    error = torch.remainder(desired - theta + torch.pi, 2 * torch.pi) - torch.pi
    distance = torch.hypot(target_dx, target_dy)
    forward = (2.0 * distance).clamp(max=1.0) * error.cos().clamp(min=0.0)
    turn = (error / (torch.pi / 2.0)).clamp(-1.0, 1.0)
    teacher = torch.stack((forward - turn, forward + turn), dim=-1).clamp(-1.0, 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.zeros((), device=device)
    for _ in range(config.warmup_epochs):
        permutations = torch.randperm(config.warmup_samples, generator=generator, device=device)
        for indices in permutations.split(1024):  # type: ignore[no-untyped-call]
            mean, _, _ = model(observation[indices])
            loss = (torch.tanh(mean) - teacher[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    return float(loss.detach())


def _gae(
    reward: Tensor,
    value: Tensor,
    next_value: Tensor,
    done: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantage = torch.zeros_like(reward)
    estimate = torch.zeros((), dtype=reward.dtype, device=reward.device)
    for index in range(len(reward) - 1, -1, -1):
        continuation = 1.0 - done[index]
        delta = reward[index] + gamma * next_value[index] * continuation - value[index]
        estimate = delta + gamma * gae_lambda * continuation * estimate
        advantage[index] = estimate
    return advantage, advantage + value


def collect_rollout(
    env: GoToTargetEnv,
    model: ActorCritic,
    config: TrainConfig,
    *,
    episode_seed: int,
    device: torch.device,
) -> tuple[TensorDict, int, float, float]:
    """Collect one fixed-size on-policy rollout in a TensorDict."""
    observation = env.reset(episode_seed)
    rows: dict[str, list[Tensor]] = {
        name: [] for name in ("observation", "action", "log_prob", "value", "reward", "done")
    }
    successes = 0
    episodes = 0
    episode_return = 0.0
    episode_returns: list[float] = []
    next_seed = episode_seed
    for _ in range(config.rollout_steps):
        observation_tensor = torch.as_tensor(observation, device=device)
        with torch.no_grad():
            action, log_prob, value = model.act(observation_tensor)
        next_observation, reward, terminated, truncated, info = env.step(
            torch.tanh(action).cpu().numpy()
        )
        done = terminated or truncated
        episode_return += reward
        rows["observation"].append(observation_tensor)
        rows["action"].append(action)
        rows["log_prob"].append(log_prob)
        rows["value"].append(value)
        rows["reward"].append(torch.tensor(reward, dtype=torch.float32, device=device))
        rows["done"].append(torch.tensor(float(done), dtype=torch.float32, device=device))
        observation = next_observation
        if done:
            episodes += 1
            successes += int(bool(info["success"]))
            episode_returns.append(episode_return)
            episode_return = 0.0
            next_seed += 1
            observation = env.reset(next_seed)
    with torch.no_grad():
        _, _, bootstrap = model(torch.as_tensor(observation, device=device))
    values = torch.stack(rows["value"])
    done_tensor = torch.stack(rows["done"])
    next_values = torch.cat((values[1:], bootstrap.reshape(1)))
    advantages, returns = _gae(
        torch.stack(rows["reward"]),
        values,
        next_values,
        done_tensor,
        config.gamma,
        config.gae_lambda,
    )
    batch = TensorDict(
        {
            "observation": torch.stack(rows["observation"]),
            "action": torch.stack(rows["action"]),
            "sample_log_prob": torch.stack(rows["log_prob"]),
            "state_value": values,
            "reward": torch.stack(rows["reward"]),
            "done": done_tensor.bool(),
            "advantage": advantages,
            "value_target": returns,
        },
        batch_size=[config.rollout_steps],
        device=device,
    )
    success_rate = successes / episodes if episodes else 0.0
    mean_return = sum(episode_returns) / len(episode_returns) if episode_returns else episode_return
    return batch, next_seed, success_rate, mean_return


def optimize(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: TensorDict,
    config: TrainConfig,
) -> dict[str, float]:
    advantages = batch["advantage"]
    batch["advantage"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    steps = 0
    generator = torch.Generator(device=batch.device).manual_seed(config.seed)
    for _ in range(config.epochs):
        indices = torch.randperm(len(batch), generator=generator, device=batch.device)
        for start in range(0, len(batch), config.minibatch_size):
            sample = batch[indices[start : start + config.minibatch_size]]
            mean, log_std, value = model(sample["observation"])
            distribution = Normal(mean, log_std.exp())
            log_prob = distribution.log_prob(sample["action"]).sum(-1)  # type: ignore[no-untyped-call]
            ratio = (log_prob - sample["sample_log_prob"]).exp()
            unclipped = ratio * sample["advantage"]
            clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
            policy_loss = -torch.minimum(unclipped, clipped * sample["advantage"]).mean()
            value_loss = 0.5 * (value - sample["value_target"]).square().mean()
            entropy = distribution.entropy().sum(-1).mean()  # type: ignore[no-untyped-call]
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            steps += 1
    return {name: value / steps for name, value in totals.items()}


def evaluate(
    env: GoToTargetEnv,
    model: ActorCritic,
    seeds: Iterable[int],
    device: torch.device,
) -> dict[str, Any]:
    outcomes: list[bool] = []
    distances: list[float] = []
    model.eval()
    for seed in seeds:
        observation = env.reset(seed)
        info: dict[str, object] = {"success": False, "distance": float("inf")}
        for _ in range(env.max_steps):
            with torch.no_grad():
                action, _, _ = model.act(
                    torch.as_tensor(observation, device=device), deterministic=True
                )
            observation, _, terminated, truncated, info = env.step(torch.tanh(action).cpu().numpy())
            if terminated or truncated:
                break
        outcomes.append(bool(info["success"]))
        distances.append(cast(float, info["distance"]))
    model.train()
    return {
        "episodes": len(outcomes),
        "successes": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes),
        "mean_final_distance": sum(distances) / len(distances),
    }


def model_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    *,
    update: int,
    frames: int,
    stage: int,
    episode_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = cast(
        tuple[str, Any, int, int, float],
        np.random.get_state(),
    )
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "config": asdict(config),
            "config_fingerprint": config.fingerprint(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "frames": frames,
            "stage": stage,
            "episode_seed": episode_seed,
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


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("incompatible checkpoint schema")
    if payload.get("config_fingerprint") != config.fingerprint():
        raise ValueError("checkpoint configuration fingerprint mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    numpy_rng = payload["numpy_rng"]
    np.random.set_state(
        (
            numpy_rng[0],
            numpy_rng[1].numpy(),
            numpy_rng[2],
            numpy_rng[3],
            numpy_rng[4],
        )
    )
    torch.set_rng_state(payload["torch_rng"])
    return {key: int(payload[key]) for key in ("update", "frames", "stage", "episode_seed")}


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True, separators=(",", ":")) + "\n")


def train(
    env: GoToTargetEnv,
    config: TrainConfig,
    *,
    checkpoint: Path,
    metrics: Path,
    resume: bool = False,
) -> ActorCritic:
    seed_everything(config.seed)
    device = torch.device(config.device)
    model = ActorCritic(env.observation_size, env.action_size, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    progress = {
        "update": 0,
        "frames": 0,
        "stage": config.initial_stage,
        "episode_seed": config.seed,
    }
    if resume:
        progress.update(load_checkpoint(checkpoint, model, optimizer, config))
    else:
        warm_start(model, config, device)
    env.stage = progress["stage"]
    started = time.monotonic()
    for update in range(progress["update"] + 1, config.updates + 1):
        batch, episode_seed, rollout_success, mean_return = collect_rollout(
            env, model, config, episode_seed=progress["episode_seed"], device=device
        )
        losses = optimize(model, optimizer, batch, config)
        progress.update(
            update=update,
            frames=progress["frames"] + config.rollout_steps,
            episode_seed=episode_seed,
        )
        evaluation: dict[str, Any] = {}
        if update % config.eval_every == 0 or update == config.updates:
            evaluation = evaluate(
                env,
                model,
                range(config.seed + 10_000, config.seed + 10_000 + config.eval_episodes),
                device,
            )
            if env.promote(float(evaluation["success_rate"])):
                progress["stage"] = env.stage
        metric = {
            "schema_version": METRICS_SCHEMA,
            "run_id": config.fingerprint()[:12],
            "seed": config.seed,
            "stage": STAGES[env.stage].name,
            "update": update,
            "frames": progress["frames"],
            "rollout_success_rate": rollout_success,
            "mean_episode_return": mean_return,
            "elapsed_seconds": time.monotonic() - started,
            **losses,
            **evaluation,
        }
        append_metric(metrics, metric)
        save_checkpoint(checkpoint, model, optimizer, config, **progress)
    return model
