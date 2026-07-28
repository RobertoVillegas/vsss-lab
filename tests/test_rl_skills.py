from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from vsss_train.config import TrainConfig, load_config
from vsss_train.ppo import (
    ActorCritic,
    _gae,
    collect_rollout,
    load_checkpoint,
    model_checksum,
    optimize,
    save_checkpoint,
    seed_everything,
    train,
    warm_start,
)
from vsss_train.task import STAGES, GoToTargetEnv

ROOT = Path(__file__).parents[1]
MATCH_CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
MATCH_STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def env(stage: int = 0, max_steps: int = 8) -> GoToTargetEnv:
    return GoToTargetEnv(MATCH_CONFIG, MATCH_STATE, stage=stage, max_steps=max_steps)


def test_curriculum_is_monotonic_and_promotes_only_at_threshold() -> None:
    assert [stage.name for stage in STAGES] == [f"C{index}" for index in range(6)]
    assert [stage.max_distance for stage in STAGES] == sorted(
        stage.max_distance for stage in STAGES
    )
    task = env()
    assert not task.promote(STAGES[0].promotion_threshold - 0.01)
    assert task.stage == 0
    assert task.promote(STAGES[0].promotion_threshold)
    assert task.stage == 1


def test_task_reset_is_seeded_and_has_bounded_contract() -> None:
    first = env(stage=5)
    second = env(stage=5)
    assert np.array_equal(first.reset(123), second.reset(123))
    observation, reward, terminated, truncated, info = first.step(
        np.asarray([4.0, -4.0], dtype=np.float32)
    )
    assert observation.shape == (7,)
    assert np.isfinite(observation).all()
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    assert set(info) == {"success", "distance", "stage"}


def test_task_truncates_at_horizon() -> None:
    task = env(max_steps=1)
    task.reset(4)
    _, _, terminated, truncated, info = task.step(np.zeros(2, dtype=np.float32))
    assert not terminated
    assert truncated
    assert info["success"] is False


def test_gae_matches_hand_calculation() -> None:
    reward = torch.tensor([1.0, 1.0])
    value = torch.tensor([0.5, 0.5])
    next_value = torch.tensor([0.5, 0.0])
    done = torch.tensor([0.0, 1.0])
    advantage, returns = _gae(reward, value, next_value, done, 1.0, 1.0)
    assert torch.allclose(advantage, torch.tensor([1.5, 0.5]))
    assert torch.allclose(returns, torch.tensor([2.0, 1.0]))


def test_rollout_uses_tensordict_and_ppo_updates_weights() -> None:
    config = replace(
        TrainConfig(),
        rollout_steps=8,
        warmup_samples=8,
        warmup_epochs=1,
        epochs=1,
        minibatch_size=4,
        max_episode_steps=4,
    )
    seed_everything(config.seed)
    model = ActorCritic(7, 2, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    before = model.actor.weight.detach().clone()
    batch, _, _, mean_return = collect_rollout(
        env(max_steps=4), model, config, episode_seed=config.seed, device=torch.device("cpu")
    )
    losses = optimize(model, optimizer, batch, config)
    assert batch.batch_size == torch.Size([8])
    assert np.isfinite(mean_return)
    assert set(losses) == {"policy_loss", "value_loss", "entropy"}
    assert not torch.equal(before, model.actor.weight)


def test_checkpoint_round_trip_and_rejects_config_drift(tmp_path: Path) -> None:
    config = TrainConfig(hidden_size=8)
    seed_everything(config.seed)
    model = ActorCritic(7, 2, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        config,
        update=2,
        frames=16,
        stage=1,
        episode_seed=9,
    )
    restored = ActorCritic(7, 2, config.hidden_size)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=config.learning_rate)
    progress = load_checkpoint(checkpoint, restored, restored_optimizer, config)
    assert progress == {"update": 2, "frames": 16, "stage": 1, "episode_seed": 9}
    for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(
            checkpoint,
            restored,
            restored_optimizer,
            replace(config, gamma=0.9),
        )


def test_warm_start_learns_geometric_teacher() -> None:
    config = TrainConfig(hidden_size=16, warmup_samples=128, warmup_epochs=2)
    seed_everything(config.seed)
    model = ActorCritic(7, 2, config.hidden_size)
    loss = warm_start(model, config, torch.device("cpu"))
    assert np.isfinite(loss)


def test_config_and_metrics_schema(tmp_path: Path) -> None:
    config = load_config(ROOT / "experiments/configs/m5-go-to-target.toml")
    assert config.schema_version == 1
    assert len(config.fingerprint()) == 64
    smoke = replace(
        config,
        hidden_size=8,
        warmup_samples=8,
        warmup_epochs=1,
        rollout_steps=8,
        updates=1,
        epochs=1,
        minibatch_size=4,
        eval_episodes=1,
        eval_every=1,
        max_episode_steps=4,
    )
    checksums: list[str] = []
    stable_metrics: list[dict[str, object]] = []
    for run in range(2):
        metrics = tmp_path / f"metrics-{run}.jsonl"
        model = train(
            env(max_steps=smoke.max_episode_steps),
            smoke,
            checkpoint=tmp_path / f"checkpoint-{run}.pt",
            metrics=metrics,
        )
        checksums.append(model_checksum(model))
        document = json.loads(metrics.read_text())
        document.pop("elapsed_seconds")
        stable_metrics.append(document)
    assert checksums[0] == checksums[1]
    assert stable_metrics[0] == stable_metrics[1]
    assert {
        "schema_version",
        "run_id",
        "seed",
        "stage",
        "update",
        "frames",
        "mean_episode_return",
        "policy_loss",
        "value_loss",
        "entropy",
    } <= stable_metrics[0].keys()
