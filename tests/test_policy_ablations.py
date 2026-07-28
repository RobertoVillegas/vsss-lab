from __future__ import annotations

from pathlib import Path

import torch
from vsss_league.training import train_iteration
from vsss_train.ablations import (
    EntityAttentionActor,
    RecurrentSharedActor,
    RecurrentState,
    SymmetricWheelLattice,
)
from vsss_train.config import MarlConfig
from vsss_train.marl import TeamBatch
from vsss_train.marl_ppo import MarlLearner

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def observations(worlds: int = 2) -> TeamBatch:
    shape = (worlds, 3)
    return TeamBatch(
        torch.zeros(*shape, 8),
        torch.zeros(*shape, 7),
        torch.zeros(*shape, 4),
        torch.zeros(*shape, 4),
        torch.zeros(*shape, 2, 6),
        torch.zeros(*shape, 3, 6),
    )


def test_recurrent_state_resets_only_finished_world() -> None:
    actor = RecurrentSharedActor(hidden_size=8)
    state = RecurrentState.zeros(worlds=2, agents=3, hidden_size=8)
    _, _, next_state = actor(observations(), state)
    retained = next_state.hidden[1].clone()
    next_state.reset_worlds(torch.tensor([True, False]))
    assert torch.count_nonzero(next_state.hidden[0]) == 0
    assert torch.equal(next_state.hidden[1], retained)


def test_attention_actor_exposes_per_entity_weights() -> None:
    actor = EntityAttentionActor(hidden_size=8, heads=2)
    mean, log_std = actor(observations())
    assert mean.shape == log_std.shape == (2, 3, 2)
    assert actor.latest_attention is not None
    assert actor.latest_attention.shape == (2, 3, 2, 1, 5)
    assert torch.allclose(actor.latest_attention.sum(dim=-1), torch.ones(2, 3, 2, 1))


def test_wheel_lattice_is_symmetric_and_bounded() -> None:
    lattice = SymmetricWheelLattice()
    actions = lattice.parse(torch.arange(len(lattice.values)))
    assert actions.shape == (len(lattice.values), 2)
    assert torch.all(actions.abs() <= 1.0)
    values = set(lattice.values)
    assert all((-left, -right) in values for left, right in values)


def test_attention_actor_runs_a_real_mappo_update_and_checkpoint(tmp_path: Path) -> None:
    config = MarlConfig(
        device="cpu",
        policy_architecture="attention",
        num_envs=1,
        hidden_size=8,
        rollout_steps=2,
        horizon=4,
        action_repeat=1,
        epochs=1,
        minibatch_size=6,
    )
    learner = MarlLearner(config)
    checkpoint = tmp_path / "attention.pt"
    result = train_iteration(
        learner,
        None,
        CONFIG,
        STATE,
        iteration=1,
        seed=14,
        opponent_id="heuristic",
        checkpoint=checkpoint,
    )
    assert result.policy_version == 1
    assert checkpoint.is_file()
    restored = MarlLearner(config)
    restored.load(checkpoint)
    assert restored.policy_version == 1
