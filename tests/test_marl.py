from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from vsss_env._native import BatchSimulator
from vsss_train.marl import (
    CentralizedCritic,
    LocalCritic,
    SharedActor,
    TeamBatch,
    build_team_observation,
)

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def initial_state() -> np.ndarray:
    return np.asarray(BatchSimulator(CONFIG, STATE, 1).reset()[0], dtype=np.float32)


def assert_batch_equal(actual: TeamBatch, expected: TeamBatch) -> None:
    for actual_field, expected_field in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_field, expected_field)


def test_observation_excludes_ids_and_canonicalizes_team_direction() -> None:
    state = initial_state()
    expected = build_team_observation(state, team=0)
    renamed = state.copy()
    renamed[[10, 21, 32]] = renamed[[32, 10, 21]]
    assert_batch_equal(build_team_observation(renamed, team=0), expected)
    yellow = build_team_observation(state, team=1)
    assert expected.self_features.shape == yellow.self_features.shape == (3, 8)
    assert expected.teammates.shape == (3, 2, 6)
    assert expected.opponents.shape == (3, 3, 6)


def test_deep_sets_ignore_teammate_and_opponent_storage_order() -> None:
    observation = build_team_observation(initial_state(), team=0)
    reordered = observation.permute_entities(torch.tensor([1, 0]), torch.tensor([2, 0, 1]))
    actor = SharedActor(hidden_size=16)
    local = LocalCritic(hidden_size=16)
    central = CentralizedCritic(hidden_size=16)
    torch.testing.assert_close(
        actor.deterministic_action(observation), actor.deterministic_action(reordered)
    )
    torch.testing.assert_close(local(observation), local(reordered))
    torch.testing.assert_close(central(observation), central(reordered))


def test_shared_actor_and_critics_are_agent_equivariant() -> None:
    observation = build_team_observation(initial_state(), team=0)
    order = torch.tensor([2, 0, 1])
    permuted = observation.permute_agents(order)
    actor = SharedActor(hidden_size=16)
    local = LocalCritic(hidden_size=16)
    central = CentralizedCritic(hidden_size=16)
    torch.testing.assert_close(
        actor.deterministic_action(permuted),
        actor.deterministic_action(observation).index_select(0, order),
    )
    torch.testing.assert_close(local(permuted), local(observation).index_select(0, order))
    torch.testing.assert_close(central(permuted), central(observation).index_select(0, order))


def test_actor_has_no_per_agent_parameters() -> None:
    actor = SharedActor(hidden_size=16)
    parameter_names = tuple(name for name, _ in actor.named_parameters())
    assert not any("agent" in name or "robot" in name for name in parameter_names)
    observation = build_team_observation(initial_state(), team=0)
    assert actor.deterministic_action(observation).shape == (3, 2)
