from __future__ import annotations

import torch
from vsss_train.ablations import (
    EntityAttentionActor,
    RecurrentSharedActor,
    RecurrentState,
    SymmetricWheelLattice,
)
from vsss_train.marl import TeamBatch


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
