from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch.distributions import Normal
from vsss_env._native import BatchSimulator
from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.marl import (
    CentralizedCritic,
    LocalCritic,
    SharedActor,
    TeamBatch,
    build_team_observation,
)
from vsss_train.marl_env import (
    MarlMatchEnv,
    _attacker_alignment_reward,
    _ball_direction_reward,
    _defensive_threat,
    _teammate_congestion,
    distill_dynamic_teacher,
    evaluate_against_random,
)
from vsss_train.marl_ppo import (
    TRAJECTORY_SCHEMA,
    MarlLearner,
    TeamTrajectory,
    TrajectoryMetadata,
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


def trajectory(learner: MarlLearner, steps: int = 4) -> TeamTrajectory:
    single = build_team_observation(initial_state(), team=0)
    observation = TeamBatch(
        *(field.unsqueeze(0).repeat(steps, *(1,) * field.ndim) for field in single)
    )
    with torch.no_grad():
        mean, log_std = learner.actor(observation)
        distribution = Normal(mean, log_std.exp())
        action = distribution.sample()  # type: ignore[no-untyped-call]
        log_probability = distribution.log_prob(action).sum(-1)  # type: ignore[no-untyped-call]
        value = learner.critic(observation)
    data = TensorDict(
        {
            "tick": torch.arange(steps).unsqueeze(-1).expand(steps, 3),
            **dict(zip(TeamBatch._fields, observation, strict=True)),
            "action": action,
            "sample_log_prob": log_probability,
            "reward_total": torch.linspace(0.0, 1.0, steps).unsqueeze(-1).expand(steps, 3),
            "terminated": torch.zeros(steps, 3, dtype=torch.bool),
            "truncated": torch.zeros(steps, 3, dtype=torch.bool),
            "state_value": value,
        },
        batch_size=[steps, 3],
    )
    return TeamTrajectory(
        TrajectoryMetadata(
            schema_version=TRAJECTORY_SCHEMA,
            run_id="test",
            episode_id=0,
            world_id=0,
            team=0,
            policy_id=learner.config.policy_id,
            policy_version=learner.policy_version,
            global_state_ref="sha256:test",
        ),
        data,
    )


def test_ippo_and_mappo_update_finite_losses() -> None:
    for algorithm in ("ippo", "mappo"):
        learner = MarlLearner(
            MarlConfig(
                device="cpu",
                num_envs=1,
                algorithm=algorithm,
                hidden_size=8,
                epochs=1,
                minibatch_size=4,
            )
        )
        before = learner.actor.action_head.weight.detach().clone()
        losses = learner.optimize(trajectory(learner))
        assert all(np.isfinite(value) for value in losses.values())
        assert learner.policy_version == 1
        assert not torch.equal(before, learner.actor.action_head.weight)


def test_optimizer_preserves_exploration_floor() -> None:
    learner = MarlLearner(
        MarlConfig(
            device="cpu",
            num_envs=1,
            hidden_size=8,
            epochs=1,
            minibatch_size=4,
            minimum_log_std=-2.0,
        )
    )
    with torch.no_grad():
        learner.actor.log_std.fill_(-10.0)

    learner.optimize(trajectory(learner))

    assert torch.all(learner.actor.log_std >= -2.0)


def test_stale_trajectory_is_rejected_before_update() -> None:
    learner = MarlLearner(MarlConfig(device="cpu", num_envs=1, hidden_size=8, epochs=1))
    stale = trajectory(learner)
    learner.policy_version = 1
    before = tuple(parameter.detach().clone() for parameter in learner.actor.parameters())
    with np.testing.assert_raises_regex(ValueError, "stale"):
        learner.optimize(stale)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(learner.actor.parameters(), before, strict=True)
    )


def test_marl_checkpoint_round_trip_and_algorithm_guard(tmp_path: Path) -> None:
    config = MarlConfig(device="cpu", num_envs=1, algorithm="ippo", hidden_size=8, epochs=1)
    learner = MarlLearner(config)
    learner.optimize(trajectory(learner))
    checkpoint = tmp_path / "ippo.pt"
    learner.save(checkpoint)
    restored = MarlLearner(config)
    restored.load(checkpoint)
    assert restored.policy_version == learner.policy_version
    for actual, expected in zip(
        restored.actor.parameters(), learner.actor.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)
    with np.testing.assert_raises_regex(ValueError, "algorithm"):
        MarlLearner(
            MarlConfig(
                device="cpu",
                num_envs=1,
                algorithm="mappo",
                hidden_size=8,
                epochs=1,
            )
        ).load(checkpoint)


def test_legacy_checkpoint_accepts_only_neutral_new_fields(tmp_path: Path) -> None:
    config = MarlConfig(device="cpu", num_envs=1, hidden_size=8, epochs=1)
    learner = MarlLearner(config)
    checkpoint = tmp_path / "legacy.pt"
    learner.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in (
        "minimum_log_std",
        "wheel_effort_coefficient",
        "ball_direction_coefficient",
        "attacker_alignment_coefficient",
        "time_penalty_coefficient",
        "movement_speed_threshold",
        "curriculum_heuristic_iterations",
    ):
        payload["config"].pop(key)
    payload["config_fingerprint"] = "legacy-fingerprint"
    torch.save(payload, checkpoint)

    MarlLearner(config).load(checkpoint)
    with pytest.raises(ValueError, match="fingerprint"):
        MarlLearner(
            MarlConfig(
                device="cpu",
                num_envs=1,
                hidden_size=8,
                epochs=1,
                ball_direction_coefficient=1.0,
            )
        ).load(checkpoint)


def test_versioned_marl_configs() -> None:
    assert load_marl_config(ROOT / "experiments/configs/m6-ippo.toml").algorithm == "ippo"
    assert load_marl_config(ROOT / "experiments/configs/m6-mappo.toml").algorithm == "mappo"
    coordinated = load_marl_config(ROOT / "experiments/configs/m13-mappo-directional.toml")
    assert coordinated.teammate_congestion_coefficient > 0.0
    assert coordinated.defensive_coverage_coefficient > 0.0
    assert coordinated.ball_direction_coefficient == 1.0
    assert coordinated.time_penalty_coefficient == 1.0
    assert coordinated.minimum_log_std == -2.0
    assert coordinated.curriculum_heuristic_iterations == 250


def test_c7_and_c8_have_explicit_opponent_modes_and_team_rewards() -> None:
    for stage, mode in ((7, "inactive"), (8, "heuristic")):
        environment = MarlMatchEnv(CONFIG, STATE, stage=stage, horizon=1)
        observation = environment.reset(9)
        assert observation.self_features.shape == (3, 8)
        _, reward, done, info = environment.step(np.zeros((3, 2), dtype=np.float32))
        assert done
        assert info["opponent_mode"] == mode
        assert np.isfinite(reward.total)


def test_action_delta_regularization_penalizes_abrupt_commands() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=2,
        action_repeat=1,
        action_delta_coefficient=0.25,
    )
    environment.reset(9)
    _, reward, _, _ = environment.step(np.ones((3, 2), dtype=np.float32))
    assert reward.action_delta == pytest.approx(-0.25)


def test_directional_reward_favors_enemy_goal_and_dynamic_attacker_motion() -> None:
    state = initial_state()
    config = json.loads(CONFIG)
    state[5:9] = (0.0, 0.0, 0.4, 0.0)
    toward_enemy = _ball_direction_reward(state, config, speed_threshold=0.03)
    state[7] = -0.4
    toward_ally = _ball_direction_reward(state, config, speed_threshold=0.03)
    assert toward_enemy > 0.0
    assert toward_ally < 0.0

    state = initial_state()
    state[5:7] = state[12:14] + np.asarray((0.2, 0.0), dtype=np.float32)
    state[15:17] = (0.2, 0.0)
    aligned = _attacker_alignment_reward(state, speed_threshold=0.03)
    state[15:17] = (-0.2, 0.0)
    opposed = _attacker_alignment_reward(state, speed_threshold=0.03)
    assert aligned > opposed
    assert aligned == pytest.approx(0.0)


def test_time_and_wheel_effort_are_bounded_by_episode_scale() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=1_500,
        action_repeat=1,
        time_penalty_coefficient=1.0,
        wheel_effort_coefficient=0.0002,
    )
    environment.reset(9)

    _, reward, _, _ = environment.step(np.ones((3, 2), dtype=np.float32))

    assert reward.time == pytest.approx(-1.0 / 1_500)
    assert reward.wheel_effort == pytest.approx(-0.0002)


def test_coordination_reward_detects_congestion_and_scales_defensive_threat() -> None:
    state = initial_state()
    separated = _teammate_congestion(state, spacing=0.14)
    state[21 + 2] = state[10 + 2] + 0.075
    state[21 + 3] = state[10 + 3]
    contacting = _teammate_congestion(state, spacing=0.14)

    assert contacting > separated
    assert _defensive_threat(-0.6, 0.15) > _defensive_threat(0.5, 0.15)
    assert _defensive_threat(0.5, 0.15) == 0.0


def test_scoreless_horizon_has_small_draw_penalty() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=1,
        action_repeat=4,
        draw_penalty=0.25,
    )
    environment.reset(9)

    _, reward, done, info = environment.step(np.zeros((3, 2), dtype=np.float32))

    assert done
    assert info["terminal_reason"] == "draw"
    assert reward.goal == pytest.approx(-0.25)


def test_stationary_ball_ends_episode_before_horizon() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=100,
        action_repeat=4,
        stagnation_penalty=0.10,
        stagnation_seconds=0.04,
        stagnation_ball_distance=0.02,
    )
    environment.reset(9)

    environment.step(np.zeros((3, 2), dtype=np.float32))
    _, reward, done, info = environment.step(np.zeros((3, 2), dtype=np.float32))

    assert done
    assert info["terminal_reason"] == "stagnation"
    assert reward.goal == pytest.approx(-0.10)


def test_distilled_shared_policy_is_finite_under_physical_action_scaling() -> None:
    actor = SharedActor(hidden_size=32)
    loss = distill_dynamic_teacher(actor, CONFIG, STATE, seed=7, samples=512, epochs=10)
    result = evaluate_against_random(
        actor,
        CONFIG,
        STATE,
        stage=8,
        seeds=range(30_007, 30_010),
        horizon=300,
        required_margin=0.02,
    )
    assert np.isfinite(loss)
    assert np.isfinite(result.policy_progress)
    assert np.isfinite(result.random_progress)


def test_normalized_actions_scale_to_physical_wheel_velocity() -> None:
    environment = MarlMatchEnv(CONFIG, STATE, stage=7, horizon=1, action_repeat=1)
    environment.reset(4)
    _, _, _, info = environment.step(np.ones((3, 2), dtype=np.float32))

    assert np.allclose(np.asarray(info["actions"])[:3], 30.0)
