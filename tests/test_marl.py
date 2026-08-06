from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch.distributions import Normal
from vsss_baselines.controllers import TURN_AUTHORITY
from vsss_env._native import BatchSimulator
from vsss_league.training import _policy_stats, create_rollout_session
from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.heterogeneity import ablate_role_features, measure_heterogeneity_gain
from vsss_train.marl import (
    CentralizedCritic,
    LocalCritic,
    RoleSharedActor,
    SharedActor,
    TeamBatch,
    build_team_observation,
)
from vsss_train.marl_env import (
    ROBOT_BASE,
    MarlMatchEnv,
    _attacker_alignment_reward,
    _ball_direction_reward,
    _coverage_formation_potential,
    _defensive_threat,
    _goal_geometry_metrics,
    _goal_geometry_potential,
    _idle_spin_flags,
    _role_formation_potential,
    _seeded_snapshot,
    _support_formation_potential,
    _team_touches_ball,
    _teammate_congestion,
    _useful_touch_impulse,
    distill_dynamic_teacher,
    evaluate_against_random,
    team_action_width,
)
from vsss_train.marl_ppo import (
    TRAJECTORY_SCHEMA,
    MarlLearner,
    TeamTrajectory,
    TrajectoryMetadata,
    _team_gae,
    bounded_action_log_prob,
    sample_bounded_action,
)
from vsss_train.primitives import parametric_primitive_wheel_actions
from vsss_train.roles import assign_roles

ROOT = Path(__file__).parents[1]


def test_team_gae_bootstraps_a_continuing_rollout() -> None:
    reward = torch.zeros(3, 1)
    value = torch.zeros(3, 1)
    done = torch.zeros(3, 1)

    advantage, _ = _team_gae(
        reward,
        value,
        done,
        bootstrap_value=torch.tensor([2.0]),
        gamma=0.9,
        gae_lambda=1.0,
    )

    assert advantage[:, 0].tolist() == pytest.approx([1.458, 1.62, 1.8])


def test_team_gae_never_crosses_an_episode_reset() -> None:
    reward = torch.zeros(3, 1)
    value = torch.zeros(3, 1)
    done = torch.tensor([[False], [True], [False]])

    advantage, _ = _team_gae(
        reward,
        value,
        done,
        bootstrap_value=torch.tensor([2.0]),
        gamma=0.9,
        gae_lambda=1.0,
    )

    assert advantage[:, 0].tolist() == pytest.approx([0.0, 0.0, 1.8])


def test_vector_environment_reports_latched_goal_when_grace_period_finishes() -> None:
    config = json.loads(CONFIG)
    config["reset"]["goal_pause"] = 0.04
    snapshot = json.loads(STATE)
    snapshot["ball"].update(x=0.72, y=0.0, vx=1.0, vy=0.0)
    training_config = MarlConfig(
        device="cpu",
        num_envs=1,
        horizon=100,
        action_repeat=1,
        defensive_coverage_coefficient=0.0,
    )
    environment = create_rollout_session(
        training_config,
        json.dumps(config),
        STATE,
    ).environment
    environment.reset_state(0, snapshot)
    actions = np.zeros((1, 3, 2), dtype=np.float32)
    observed_goal = False
    goal_reward_steps = 0

    for _ in range(20):
        _, rewards, done, events, _ = environment.step(actions, None)
        goal_reward_steps += int(rewards[0] > training_config.goal_coefficient / 2)
        observed_goal |= bool(events[0] & 1)
        if done[0]:
            assert events[0] & 1
            assert environment.last_terminal_reasons[0] == "goal"
            break
    else:
        pytest.fail("goal grace period did not complete")

    assert observed_goal
    assert goal_reward_steps == 1


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


def test_m18_role_actor_supports_relu_layer_norm_and_larger_width() -> None:
    baseline = RoleSharedActor(hidden_size=128)
    candidate = RoleSharedActor(hidden_size=256, activation="relu", layer_norm=True)
    assert any(isinstance(module, torch.nn.LayerNorm) for module in candidate.modules())
    assert any(isinstance(module, torch.nn.ReLU) for module in candidate.modules())
    assert sum(parameter.numel() for parameter in candidate.parameters()) > sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    observation = build_team_observation(initial_state(), team=0)
    assert candidate.deterministic_action(observation).shape == (3, 2)


def test_team_ball_contact_requires_enabled_robot_inside_envelope() -> None:
    state = initial_state()
    state[5:7] = state[ROBOT_BASE + 2 : ROBOT_BASE + 4]
    assert _team_touches_ball(state, 0, json.loads(CONFIG))
    assert not _team_touches_ball(state, 1, json.loads(CONFIG))
    state[ROBOT_BASE + 10] = 0.0
    assert not _team_touches_ball(state, 0, json.loads(CONFIG))


def test_useful_touch_impulse_is_directional_and_cannot_farm_overlap() -> None:
    assert _useful_touch_impulse(0.8, 0.1, 0, True, False) == pytest.approx(0.7)
    assert _useful_touch_impulse(0.8, 0.1, 0, True, True) == 0.0
    assert _useful_touch_impulse(-0.8, -0.1, 1, True, False) == pytest.approx(0.7)


def test_useful_touch_impulse_costs_the_wrong_way_as_much_as_it_pays() -> None:
    forward = _useful_touch_impulse(0.8, 0.1, 0, True, False)
    backward = _useful_touch_impulse(-0.7, 0.0, 0, True, False)
    assert backward == pytest.approx(-forward)
    # An envelope flicker re-enters contact on velocity noise. Signed, those impulses
    # cancel; clamped to the positive side they accumulated into free return.
    noise = (0.03, -0.02, 0.05, -0.06, 0.01, -0.01)
    flicker = [_useful_touch_impulse(delta, 0.0, 0, True, False) for delta in (*noise, -sum(noise))]
    assert sum(flicker) == pytest.approx(0.0)
    assert all(value != 0.0 for value in flicker)


def test_goal_geometry_favors_a_controllable_line_through_goal_aperture() -> None:
    config = json.loads(CONFIG)
    state = initial_state()
    state[5:7] = (0.10, 0.0)
    state[12:14] = (-0.05, 0.0)
    aligned = _goal_geometry_metrics(state, config)
    state[13] = 0.25
    corner_line = _goal_geometry_metrics(state, config)

    assert aligned["attacker_alignment"] > corner_line["attacker_alignment"]
    assert aligned["goal_aperture"] == pytest.approx(1.0)
    assert corner_line["goal_aperture"] == 0.0
    assert aligned["potential"] > corner_line["potential"]


def test_discounted_goal_geometry_cannot_reward_camping_behind_ball() -> None:
    config = json.loads(CONFIG)
    state = initial_state()
    state[5:7] = (0.10, 0.0)
    state[12:14] = (-0.05, 0.0)
    potential = _goal_geometry_potential(state, config)
    stationary_reward = 0.10 * (0.99 * potential - potential)

    advanced = state.copy()
    advanced[5] += 0.10
    advanced[12] += 0.10
    advancing_reward = 0.10 * (0.99 * _goal_geometry_potential(advanced, config) - potential)

    assert stationary_reward < 0.0
    assert advancing_reward > 0.0


def test_terminal_state_carries_no_shaping_potential() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=1,
        action_repeat=1,
        goal_geometry_coefficient=0.5,
        goal_geometry_discount=0.99,
    )
    environment.reset(3)
    entry = _goal_geometry_potential(environment.state, json.loads(CONFIG), 0)

    _, reward, done, _ = environment.step(np.zeros((3, 2), dtype=np.float32))

    # The horizon ends this episode, so shaping may only remove the entry potential.
    assert done
    assert entry > 0.0
    assert reward.goal_geometry == pytest.approx(-0.5 * entry, abs=2e-3)


def test_role_formation_is_terminal_zeroed_and_cannot_pay_for_holding() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=1,
        action_repeat=1,
        role_formation_coefficient=0.2,
        goal_geometry_discount=0.99,
    )
    environment.reset(7)
    assignment = assign_roles(environment.state, 0)
    entry = _role_formation_potential(environment.state, 0, assignment)

    _, reward, done, _ = environment.step(np.zeros((3, 2), dtype=np.float32))

    assert done
    assert entry > 0.0
    assert reward.role_formation == pytest.approx(-0.2 * entry, abs=2e-3)
    assert 0.2 * (0.99 * entry - entry) < 0.0


def test_inactive_support_and_coverage_do_not_contribute_to_formation() -> None:
    state = initial_state()
    assignment = assign_roles(state, 0)
    attacker = assignment.roles.index("attacker")
    for slot in range(3):
        state[ROBOT_BASE + slot * 11 + 10] = float(slot == attacker)
    reduced = assign_roles(state, 0)

    assert _role_formation_potential(state, 0, reduced) == 0.0


def test_role_formation_is_invariant_to_controlled_robot_order() -> None:
    state = initial_state()
    expected = _role_formation_potential(state, 0, assign_roles(state, 0))
    permuted = state.copy()
    first = state[ROBOT_BASE : ROBOT_BASE + 11].copy()
    second = state[ROBOT_BASE + 11 : ROBOT_BASE + 22].copy()
    permuted[ROBOT_BASE : ROBOT_BASE + 11] = second
    permuted[ROBOT_BASE + 11 : ROBOT_BASE + 22] = first

    actual = _role_formation_potential(permuted, 0, assign_roles(permuted, 0))

    assert actual == pytest.approx(expected)


def test_role_formation_uses_both_active_responsibilities_as_a_bottleneck() -> None:
    state = initial_state()
    assignment = assign_roles(state, 0)
    attack_sign = 1.0
    ball_x, ball_y = float(state[5]), float(state[6])
    targets = {
        "support": (ball_x - attack_sign * 0.22, ball_y * 0.55),
        "coverage": (-0.70, ball_y * 0.65),
    }
    contributions = []
    for slot, role in enumerate(assignment.roles):
        if role == "attacker":
            continue
        base = ROBOT_BASE + slot * 11
        target = targets[role]
        distance = math.dist((float(state[base + 2]), float(state[base + 3])), target)
        contributions.append(math.exp(-distance / 0.25))

    potential = _role_formation_potential(state, 0, assignment)

    assert potential == pytest.approx(math.sqrt(math.prod(contributions)))
    assert potential < sum(contributions) / len(contributions)


def test_per_role_formation_potentials_are_independent() -> None:
    state = initial_state()
    assignment = assign_roles(state, 0)
    attack_sign = 1.0
    ball_x, ball_y = float(state[5]), float(state[6])
    targets = {
        "support": (ball_x - attack_sign * 0.22, ball_y * 0.55),
        "coverage": (-0.70, ball_y * 0.65),
    }
    for role in ("support", "coverage"):
        slot = assignment.roles.index(role)
        base = ROBOT_BASE + slot * 11
        state[base + 2], state[base + 3] = targets[role]

    support_potential = _support_formation_potential(state, 0, assignment)
    coverage_potential = _coverage_formation_potential(state, 0, assignment)

    assert support_potential > 0.9
    assert coverage_potential > 0.9
    assert _role_formation_potential(state, 0, assignment) == pytest.approx(
        math.sqrt(support_potential * coverage_potential), abs=1e-6
    )

    coverage_slot = assignment.roles.index("coverage")
    state[ROBOT_BASE + coverage_slot * 11 + 2] -= 0.8

    assert _support_formation_potential(state, 0, assignment) == pytest.approx(support_potential)
    assert _coverage_formation_potential(state, 0, assignment) < 0.05


def test_per_role_formation_terms_are_terminal_zeroed() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=1,
        action_repeat=1,
        support_formation_coefficient=0.2,
        coverage_formation_coefficient=0.3,
        goal_geometry_discount=0.99,
    )
    environment.reset(7)
    assignment = assign_roles(environment.state, 0)
    support_entry = _support_formation_potential(environment.state, 0, assignment)
    coverage_entry = _coverage_formation_potential(environment.state, 0, assignment)

    _, reward, done, _ = environment.step(np.zeros((3, 2), dtype=np.float32))

    assert done
    assert support_entry > 0.0
    assert coverage_entry > 0.0
    assert reward.support_formation == pytest.approx(-0.2 * support_entry, abs=2e-3)
    assert reward.coverage_formation == pytest.approx(-0.3 * coverage_entry, abs=2e-3)


def test_role_hysteresis_knobs_reject_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MarlMatchEnv(CONFIG, STATE, stage=7, role_switch_penalty=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        MarlMatchEnv(CONFIG, STATE, stage=7, role_emergency_margin=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        MarlMatchEnv(CONFIG, STATE, stage=7, support_formation_coefficient=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        MarlMatchEnv(CONFIG, STATE, stage=7, coverage_formation_coefficient=-0.1)


def test_idle_spin_detection_exempts_orientation_and_ball_control() -> None:
    state = initial_state()
    actions = np.asarray(((-0.8, 0.8), (0.4, 0.8), (-0.8, 0.8)), dtype=np.float32)
    state[12:14] = (-0.50, -0.40)
    state[23:25] = (-0.25, 0.0)
    state[34:36] = state[5:7]
    state[17] = 2.0
    state[28] = 2.0
    state[39] = 2.0

    flags, intensity = _idle_spin_flags(
        state,
        0,
        actions,
        angular_speed_threshold=1.0,
        drive_threshold=0.15,
        speed_threshold=0.08,
        ball_distance=0.12,
    )

    # Slot 1 asks to drive and slot 2 is on the ball, so only slot 0 is idle spin.
    assert flags.tolist() == [True, False, False]
    assert intensity.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_idle_spin_detection_is_reachable_under_every_action_parser() -> None:
    """Judged on measured yaw, so the differential a parser can request is irrelevant.

    A geometric controller spends at most a small fraction of the wheel limit on turning.
    The command-space threshold this replaced was either unreachable through such a parser
    or, once rescaled by that fraction, fired on an ordinary heading error of 12 degrees.
    """
    state = initial_state()
    state[12:14] = (-0.50, -0.40)
    state[23:25] = (-0.25, 0.0)
    state[34:36] = (-0.40, 0.30)
    state[17] = 1.4
    state[28] = 0.2
    state[39] = 1.4
    # Wheels a skill parser can actually produce: the executor never exceeds its own turn
    # authority, so these differentials are tiny next to direct wheel control.
    skill_parser_wheels = np.asarray(
        ((-TURN_AUTHORITY, TURN_AUTHORITY), (-TURN_AUTHORITY, TURN_AUTHORITY), (0.5, 0.6)),
        dtype=np.float32,
    )

    flags, intensity = _idle_spin_flags(
        state,
        0,
        skill_parser_wheels,
        angular_speed_threshold=1.0,
        drive_threshold=0.07,
        speed_threshold=0.08,
        ball_distance=0.12,
    )

    # Slot 0 spins in place, slot 1 turns slowly, slot 2 asks to drive.
    assert flags.tolist() == [True, False, False]
    assert intensity.tolist() == pytest.approx([0.7, 0.1, 0.7])


def test_idle_spin_detection_ignores_an_ordinary_heading_correction() -> None:
    """The false positive a rescaled command-space threshold reintroduced."""
    state = initial_state()
    state[12:14] = (-0.50, -0.40)
    state[23:25] = (-0.25, 0.0)
    state[34:36] = (-0.40, 0.30)
    turn = TURN_AUTHORITY * math.radians(20.0) / (math.pi / 2.0)
    wheels = np.asarray(((-turn, turn), (-turn, turn), (-turn, turn)), dtype=np.float32)
    # A twenty degree correction yields well under a radian per second of yaw.
    state[17] = 0.35
    state[28] = 0.35
    state[39] = 0.35

    flags, _ = _idle_spin_flags(
        state,
        0,
        wheels,
        angular_speed_threshold=1.0,
        drive_threshold=0.07,
        speed_threshold=0.08,
        ball_distance=0.12,
    )

    assert not flags.any()


def trajectory(learner: MarlLearner, steps: int = 4) -> TeamTrajectory:
    single = build_team_observation(initial_state(), team=0)
    observation = TeamBatch(
        *(field.unsqueeze(0).repeat(steps, *(1,) * field.ndim) for field in single)
    )
    with torch.no_grad():
        mean, log_std = learner.actor(observation)
        distribution = Normal(mean, log_std.exp())
        action, log_probability = sample_bounded_action(distribution)
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
            "bootstrap_value": value[-1].unsqueeze(0).expand(steps, -1),
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


def test_policy_stats_separate_primitive_choice_by_dynamic_role() -> None:
    learner = MarlLearner(
        MarlConfig(
            device="cpu",
            num_envs=1,
            hidden_size=8,
        )
    )
    sample = trajectory(learner, steps=2)
    # The observation has one agent in each role. Attacker always strikes, support navigates,
    # and coverage stops, making any aggregate-only report deliberately ambiguous.
    roles = sample.data["context"][..., 4:7].argmax(dim=-1)
    action_index = torch.zeros_like(roles)
    action_index[roles == 0] = 9
    action_index[roles == 1] = 1
    sample.data["action_index"] = action_index

    stats = _policy_stats(sample, "primitive")

    assert stats is not None
    by_role = stats["actions_by_role"]
    assert isinstance(by_role, dict)
    assert by_role["attacker"]["strike_fraction"] == 1.0
    assert by_role["support"]["navigate_fraction"] == 1.0
    assert by_role["coverage"]["stop_fraction"] == 1.0


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
        assert isinstance(learner.actor, SharedActor)
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


def test_optimizer_enforces_exploration_ceiling() -> None:
    learner = MarlLearner(
        MarlConfig(
            device="cpu",
            num_envs=1,
            hidden_size=8,
            epochs=1,
            minibatch_size=4,
            maximum_log_std=-0.2,
        )
    )
    with torch.no_grad():
        learner.actor.log_std.fill_(3.0)

    learner.optimize(trajectory(learner))

    assert torch.all(learner.actor.log_std <= -0.2)


def test_bounded_action_matches_transformed_gaussian_density() -> None:
    distribution = Normal(torch.zeros(64, 2), torch.ones(64, 2))
    action, log_probability = sample_bounded_action(distribution)

    assert torch.all(action > -1.0)
    assert torch.all(action < 1.0)
    assert torch.all(torch.isfinite(log_probability))
    torch.testing.assert_close(
        bounded_action_log_prob(distribution, action),
        log_probability,
    )


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
        "maximum_log_std",
        "wheel_effort_coefficient",
        "ball_direction_coefficient",
        "attacker_alignment_coefficient",
        "time_penalty_coefficient",
        "movement_speed_threshold",
        "curriculum_heuristic_iterations",
        "semantic_curriculum",
        "semantic_full_match_fraction",
        "semantic_terminal_reward",
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


def test_legacy_checkpoint_rejects_non_neutral_clearing_knob(tmp_path: Path) -> None:
    """ADR 0027: a pre-clearing checkpoint loads only at the neutral flag; the flag on is a
    fingerprint mismatch, so no silently different behavior rides on a legacy load."""
    legacy = MarlLearner(
        MarlConfig(
            device="cpu",
            num_envs=1,
            hidden_size=8,
            epochs=1,
            strike_clearing_enabled=False,
        )
    )
    checkpoint = tmp_path / "legacy.pt"
    legacy.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["config"].pop("strike_clearing_enabled")
    payload["config"].pop("strike_clearing_distance")
    payload["config_fingerprint"] = "legacy-fingerprint"
    torch.save(payload, checkpoint)

    MarlLearner(
        MarlConfig(
            device="cpu",
            num_envs=1,
            hidden_size=8,
            epochs=1,
            strike_clearing_enabled=False,
        )
    ).load(checkpoint)
    with pytest.raises(ValueError, match="fingerprint"):
        MarlLearner(
            MarlConfig(
                device="cpu",
                num_envs=1,
                hidden_size=8,
                epochs=1,
                strike_clearing_enabled=True,
            )
        ).load(checkpoint)


def test_policy_warm_start_allows_reward_change_and_resets_version(tmp_path: Path) -> None:
    source = MarlLearner(MarlConfig(device="cpu", num_envs=1, hidden_size=8, epochs=1))
    source.policy_version = 700
    checkpoint = tmp_path / "source.pt"
    source.save(checkpoint)
    target = MarlLearner(
        MarlConfig(
            device="cpu",
            num_envs=1,
            hidden_size=8,
            epochs=1,
            semantic_curriculum=True,
            entropy_coefficient=0.003,
            minimum_log_std=-1.2,
        )
    )

    source_version = target.initialize_policy(checkpoint)

    assert source_version == 700
    assert target.policy_version == 0
    for actual, expected in zip(target.actor.parameters(), source.actor.parameters(), strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(
        target.critic.parameters(), source.critic.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)


def test_policy_warm_start_rejects_architecture_change(tmp_path: Path) -> None:
    source = MarlLearner(MarlConfig(device="cpu", num_envs=1, hidden_size=8, epochs=1))
    checkpoint = tmp_path / "source.pt"
    source.save(checkpoint)
    target = MarlLearner(MarlConfig(device="cpu", num_envs=1, hidden_size=16, epochs=1))
    with pytest.raises(ValueError, match="architecture mismatch"):
        target.initialize_policy(checkpoint)


def test_versioned_marl_configs() -> None:
    assert load_marl_config(ROOT / "experiments/configs/m6-ippo.toml").algorithm == "ippo"
    assert load_marl_config(ROOT / "experiments/configs/m6-mappo.toml").algorithm == "mappo"
    coordinated = load_marl_config(ROOT / "experiments/configs/m13-mappo-directional.toml")
    clean_curriculum = load_marl_config(
        ROOT / "experiments/configs/m23-mappo-clean-curriculum.toml"
    )
    run_0013 = load_marl_config(ROOT / "experiments/configs/m24-3-mappo-circular.toml")
    role_formation = load_marl_config(ROOT / "experiments/configs/m24-3-mappo-role-formation.toml")
    assert clean_curriculum.seed == 23
    assert clean_curriculum.semantic_full_match_fraction == 0.30
    assert coordinated.teammate_congestion_coefficient > 0.0
    assert coordinated.defensive_coverage_coefficient > 0.0
    assert coordinated.ball_direction_coefficient == 1.0
    assert coordinated.time_penalty_coefficient == 1.0
    assert coordinated.minimum_log_std == -2.0
    assert coordinated.maximum_log_std == -0.2
    assert coordinated.curriculum_heuristic_iterations == 1000
    assert coordinated.league_heuristic_weight == 0.35
    assert run_0013.role_formation_coefficient == 0.0
    assert role_formation.role_formation_coefficient == 0.10
    assert role_formation.policy_id != run_0013.policy_id
    per_role = load_marl_config(ROOT / "experiments/configs/m24-4-mappo-role-formation.toml")
    complementarity = load_marl_config(ROOT / "experiments/configs/m24-5-mappo-role-formation.toml")
    assert per_role.role_formation_coefficient == 0.0
    assert per_role.support_formation_coefficient == 0.15
    assert per_role.coverage_formation_coefficient == 0.15
    assert per_role.role_switch_penalty == 0.30
    assert per_role.role_emergency_margin == 0.30
    assert complementarity.role_formation_coefficient == 0.10
    assert complementarity.support_formation_coefficient == 0.15
    assert complementarity.coverage_formation_coefficient == 0.15
    assert complementarity.policy_id != per_role.policy_id


def test_role_ablation_rewrites_only_role_columns() -> None:
    observation = build_team_observation(initial_state(), team=0)
    rng = np.random.default_rng(0)
    uniform = ablate_role_features(observation, "uniform", rng)
    assert torch.equal(uniform.context[..., :4], observation.context[..., :4])
    roles = uniform.context[..., 4:]
    assert torch.all(roles[..., 0] == 1.0)
    assert torch.all(roles[..., 1:] == 0.0)
    none = ablate_role_features(observation, "none", rng)
    assert torch.all(none.context[..., 4:] == 0.0)
    shuffled = ablate_role_features(observation, "shuffle", rng)
    assert torch.allclose(
        shuffled.context[..., 4:].sum(dim=0), observation.context[..., 4:].sum(dim=0)
    )
    with pytest.raises(ValueError, match="ablation mode"):
        ablate_role_features(observation, "mirror", rng)


def test_heterogeneity_gain_reports_paired_progress() -> None:
    config = MarlConfig(
        device="cpu",
        num_envs=1,
        hidden_size=8,
        epochs=1,
        policy_architecture="role_mlp",
    )
    learner = MarlLearner(config)
    result = measure_heterogeneity_gain(
        learner.actor,
        CONFIG,
        STATE,
        stage=7,
        seeds=range(3),
        horizon=20,
        action_repeat=1,
        ablation="uniform",
    )
    assert result.seeds == 3
    assert math.isfinite(result.gain)
    assert math.isfinite(result.conditioned_progress)
    assert math.isfinite(result.ablated_progress)


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


def test_environment_rejects_actions_that_do_not_match_its_action_parser() -> None:
    assert team_action_width("continuous") == 2
    assert team_action_width("primitive") == 2
    assert team_action_width("parametric_primitive") == 4
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=8,
        action_repeat=1,
        action_parser="parametric_primitive",
    )
    environment.reset(5)

    with pytest.raises(ValueError, match=r"controlled team actions must have shape \(3, 4\)"):
        environment.step(np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"opponent team actions must have shape \(3, 4\)"):
        environment.step(
            np.zeros((3, 4), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
        )

    _, _, _, info = environment.step(np.zeros((3, 4), dtype=np.float32))

    assert "actions" in info


def test_learned_opponent_token_is_parsed_once_per_decision() -> None:
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=8,
        horizon=8,
        action_repeat=4,
        action_parser="parametric_primitive",
    )
    environment.reset(11)
    before = environment.state.copy()
    opponent = np.asarray(
        ((0.0, 0.9, 0.4, 0.5), (1.0, -0.3, 0.8, 0.2), (0.0, 0.2, -0.7, 1.0)),
        dtype=np.float32,
    )
    expected = parametric_primitive_wheel_actions(before, team=1, tokens=opponent)

    _, _, _, info = environment.step(np.zeros((3, 4), dtype=np.float32), opponent)

    maximum = float(json.loads(CONFIG)["max_wheel_speed"])
    assert np.allclose(np.asarray(info["actions"])[3:], expected * maximum)


def test_normalized_actions_scale_to_physical_wheel_velocity() -> None:
    environment = MarlMatchEnv(CONFIG, STATE, stage=7, horizon=1, action_repeat=1)
    environment.reset(4)
    _, _, _, info = environment.step(np.ones((3, 2), dtype=np.float32))

    assert np.allclose(np.asarray(info["actions"])[:3], 30.0)


def test_impasse_restarts_at_the_free_ball_mark_instead_of_ending_the_game() -> None:
    """Rule 15: ten seconds of impasse away from the goal areas restarts play.

    Ending the episode and charging a penalty taught that a stalled ball is a loss, which is
    not what the rulebook says happens.
    """
    config = MarlConfig(
        device="cpu",
        num_envs=1,
        horizon=2_000,
        action_repeat=4,
        free_ball_seconds=1.0,
        stagnation_ball_distance=0.02,
        goal_coefficient=0.0,
    )
    environment = create_rollout_session(config, CONFIG, STATE).environment
    snapshot = json.loads(STATE)
    # A ball parked away from both goal areas with nobody near it is an impasse.
    snapshot["ball"].update(x=0.10, y=0.20, vx=0.0, vy=0.0, omega=0.0)
    for robot, (x, y) in zip(
        snapshot["robots"],
        ((-0.60, 0.0), (-0.55, 0.30), (-0.55, -0.30), (0.60, 0.0), (0.55, 0.30), (0.55, -0.30)),
        strict=True,
    ):
        robot["pose"].update(x=x, y=y)
        robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
    environment.reset_state(0, snapshot)
    actions = np.zeros((1, 3, 2), dtype=np.float32)

    ended = False
    for _ in range(environment.free_ball_limit + 3):
        _, _, done, _, _ = environment.step(actions, None)
        ended |= bool(done[0])

    assert not ended
    assert environment.free_balls[0] >= 1
    assert environment.last_terminal_reasons[0] != "stagnation"
    # The ball is on a free-ball mark rather than where it stalled.
    assert abs(float(environment.states[0, 5])) == pytest.approx(0.375, abs=1e-3)


def test_full_match_kickoff_samples_inside_the_center_circle() -> None:
    """Rule 7: every full-match kickoff places the ball inside the 20 cm circle."""
    template = json.loads(STATE)
    for seed in range(48):
        snapshot = _seeded_snapshot(template, seed, full_match_kickoff_radius=0.20)
        assert math.hypot(snapshot["ball"]["x"], snapshot["ball"]["y"]) <= 0.20 + 1e-6


def test_full_match_kickoff_radius_zero_rests_at_center() -> None:
    template = json.loads(STATE)
    snapshot = _seeded_snapshot(template, 9, full_match_kickoff_radius=0.0)
    assert snapshot["ball"]["x"] == pytest.approx(0.0, abs=1e-9)
    assert snapshot["ball"]["y"] == pytest.approx(0.0, abs=1e-9)


def test_full_match_kickoff_rejects_an_illegal_radius() -> None:
    with pytest.raises(ValueError, match="full_match_kickoff_radius"):
        MarlMatchEnv(CONFIG, STATE, stage=7, horizon=1, full_match_kickoff_radius=0.21)
    with pytest.raises(ValueError, match="full_match_kickoff_radius"):
        MarlConfig(full_match_kickoff_radius=-0.01)


def test_full_match_kickoff_defenders_start_outside_the_circle() -> None:
    template = json.loads(STATE)
    for seed in range(48):
        snapshot = _seeded_snapshot(template, seed, full_match_kickoff_radius=0.20)
        for robot in snapshot["robots"][3:]:
            assert math.hypot(robot["pose"]["x"], robot["pose"]["y"]) > 0.20


def test_full_match_kickoff_resets_are_seeded() -> None:
    template = json.loads(STATE)
    first = _seeded_snapshot(template, 13, full_match_kickoff_radius=0.20)
    second = _seeded_snapshot(template, 13, full_match_kickoff_radius=0.20)
    assert first["ball"] == second["ball"]
    for left, right in zip(first["robots"], second["robots"], strict=True):
        assert left == right
