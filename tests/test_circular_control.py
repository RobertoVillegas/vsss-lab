from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, Normal
from vsss_baselines.controllers import go_to_target, robot_pose
from vsss_env._native import BatchSimulator
from vsss_league.training import collect_self_play_trajectory, create_rollout_session
from vsss_train.config import MarlConfig
from vsss_train.marl import CircularPrimitiveRoleActor, build_team_observation
from vsss_train.marl_env import team_action_width
from vsss_train.marl_ppo import (
    MarlLearner,
    bounded_action_log_prob,
    circular_action_log_prob,
    load_policy_actor,
    observation_from_trajectory,
    von_mises_entropy,
)
from vsss_train.primitives import (
    CircularPrimitiveSet,
    ParametricPrimitiveSet,
    _strike_target,
    circular_primitive_wheel_actions,
)

ROBOT_BASE = 10
ROBOT_WIDTH = 11

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def circular_config(**overrides: object) -> MarlConfig:
    base = {
        "device": "cpu",
        "num_envs": 2,
        "rollout_steps": 4,
        "minibatch_size": 6,
        "epochs": 1,
        "hidden_size": 32,
        "action_parser": "circular_primitive",
        "policy_architecture": "role_mlp",
    }
    return MarlConfig(**{**base, **overrides})  # type: ignore[arg-type]


def initial_state() -> np.ndarray:
    return np.asarray(BatchSimulator(CONFIG, STATE, 1).reset()[0], dtype=np.float32)


def circular_deviation_degrees(mean: float, concentration: float, samples: int = 40_000) -> float:
    torch.manual_seed(11)
    heading = torch.full((samples,), mean)
    sampled = torch.distributions.VonMises(
        heading,
        torch.full((samples,), concentration),
    ).sample()  # type: ignore[no-untyped-call]
    resultant = torch.hypot(torch.cos(sampled).mean(), torch.sin(sampled).mean()).clamp_min(1e-12)
    return float(torch.rad2deg(torch.sqrt(-2.0 * torch.log(resultant))))


def test_heading_precision_does_not_depend_on_the_direction_requested() -> None:
    """The bounded pair confines concentration to a square; the circle does not."""
    concentration = 40.0
    axis = circular_deviation_degrees(0.0, concentration)
    diagonal = circular_deviation_degrees(math.pi / 4.0, concentration)
    behind = circular_deviation_degrees(math.pi, concentration)

    assert axis == pytest.approx(diagonal, rel=0.08)
    assert behind == pytest.approx(diagonal, rel=0.08)
    # The parameterization this replaces spans a factor of about thirty between the two.
    assert max(axis, diagonal, behind) / min(axis, diagonal, behind) < 1.2


def test_transport_width_and_round_trip_survive_the_bounded_contract() -> None:
    assert team_action_width("circular_primitive") == 3
    assert CircularPrimitiveSet.token_width == 3
    skills = torch.tensor([0, 1, 2])
    headings = torch.tensor([0.0, 3.0, -3.0])
    intensities = torch.tensor([-1.0, 0.25, 1.0])
    tokens = CircularPrimitiveSet.encode(skills, headings, intensities)

    assert tokens.shape == (3, 3)
    assert bool((tokens.abs() <= 1.0).all())
    clipped = np.clip(tokens.numpy(), -1.0, 1.0)
    for index, token in enumerate(clipped):
        command = CircularPrimitiveSet.decode(token)
        assert command.skill == CircularPrimitiveSet.skill_labels[index]
        if index:
            assert command.direction == pytest.approx(float(headings[index]), abs=1e-5)
            assert command.intensity == pytest.approx(
                float((intensities[index] + 1.0) * 0.5), abs=1e-6
            )


def test_heading_wraps_without_a_discontinuity() -> None:
    """Continuity belongs to the requested heading, not to the controller's tie-break.

    Reversing by exactly half a turn is a genuine tie for a differential drive: spinning
    left and spinning right are equally good, and `go_to_target` breaks that tie by the
    sign of the heading error, which flips across the wrap. What must stay continuous is
    the direction and the target the token asks for.
    """
    epsilon = 1e-4
    below = CircularPrimitiveSet.decode(
        np.clip(
            CircularPrimitiveSet.encode(
                torch.tensor([1]), torch.tensor([math.pi - epsilon]), torch.tensor([0.5])
            ).numpy()[0],
            -1.0,
            1.0,
        )
    )
    above = CircularPrimitiveSet.decode(
        np.clip(
            CircularPrimitiveSet.encode(
                torch.tensor([1]), torch.tensor([-math.pi + epsilon]), torch.tensor([0.5])
            ).numpy()[0],
            -1.0,
            1.0,
        )
    )

    assert below.skill == above.skill == "navigate"
    assert below.intensity == pytest.approx(above.intensity)
    assert math.cos(below.direction) == pytest.approx(math.cos(above.direction), abs=1e-6)
    assert math.sin(below.direction) == pytest.approx(math.sin(above.direction), abs=1e-3)


def test_rollout_and_update_recover_one_log_probability() -> None:
    """The invariant that keeps the PPO ratio meaningful, asserted rather than inspected."""
    config = circular_config()
    learner = MarlLearner(config)
    session = create_rollout_session(config, CONFIG, STATE)
    trajectory, *_ = collect_self_play_trajectory(
        learner,
        None,
        CONFIG,
        STATE,
        seed=1,
        opponent_id="heuristic",
        session=session,
    )
    observation = observation_from_trajectory(trajectory.data)
    with torch.no_grad():
        skill_logits, heading, concentration, intensity_mean, intensity_log_std = learner.actor(
            observation
        )
        recomputed = (
            Categorical(logits=skill_logits).log_prob(  # type: ignore[no-untyped-call]
                trajectory.data["action_index"]
            )
            + circular_action_log_prob(
                heading,
                concentration,
                trajectory.data["action"][..., 1] * math.pi,
            )
            + bounded_action_log_prob(
                Normal(intensity_mean, intensity_log_std.exp()),
                trajectory.data["action"][..., 2:],
            )
        )
    ratio = (recomputed - trajectory.data["sample_log_prob"]).exp()

    assert trajectory.data["action"].shape[-1] == 3
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-4, atol=1e-4)


def test_requested_authority_informs_intercept_selection() -> None:
    """Reachability is judged at the authority that will execute, not at full speed.

    The effect is bounded: it only changes the choice where full authority would have
    committed to an early intercept. When no candidate is reachable at all, both
    authorities still fall through to the furthest prediction, which is recorded as an
    open question rather than silently redesigned here.
    """
    state = initial_state()
    state[5:9] = (0.0, 0.0, 0.3, 0.0)
    pose = (-0.12, 0.0, 0.0)
    direction = (1.0, 0.0)
    full = _strike_target(state, pose, direction, ball_deceleration=0.8, authority=1.0)
    reduced = _strike_target(state, pose, direction, ball_deceleration=0.8, authority=0.2)
    default = _strike_target(state, pose, direction, ball_deceleration=0.8)

    assert reduced != full
    # Full authority reproduces the behavior from before authority was consulted.
    assert default == full
    # Full authority has earned the drive-through; reduced authority still aims to acquire.
    assert full[0] < reduced[0]


def test_a_circular_policy_cannot_load_under_the_previous_heading_contract(
    tmp_path: Path,
) -> None:
    config = circular_config()
    learner = MarlLearner(config)
    checkpoint = tmp_path / "circular.pt"
    learner.save(checkpoint)
    parametric = circular_config(action_parser="parametric_primitive")

    with pytest.raises(ValueError):
        load_policy_actor(checkpoint, parametric, torch.device("cpu"))


def test_von_mises_entropy_matches_its_definition_and_stays_finite() -> None:
    concentration = torch.tensor([0.25, 1.0, 50.0, 5_000.0])
    entropy = von_mises_entropy(concentration)
    uniform = math.log(2.0 * math.pi)

    assert bool(torch.isfinite(entropy).all())
    assert float(entropy[0]) < uniform
    assert float(entropy[0]) > float(entropy[-1])
    reference = (
        -concentration * torch.special.i1(concentration) / torch.special.i0(concentration)
        + torch.log(2.0 * math.pi * torch.special.i0(concentration))
    )[:2]
    torch.testing.assert_close(entropy[:2], reference, rtol=1e-5, atol=1e-5)


def test_deterministic_action_reports_the_skill_mix_the_actor_prefers() -> None:
    actor = CircularPrimitiveRoleActor(32)
    observation = build_team_observation(initial_state(), team=0)
    action = actor.deterministic_action(observation)

    assert action.shape == (3, team_action_width("circular_primitive"))
    assert bool((action[:, 1].abs() <= 1.0).all())
    assert ParametricPrimitiveSet.action_count == CircularPrimitiveSet.action_count
    parsed = circular_primitive_wheel_actions(
        initial_state(),
        team=0,
        tokens=action.detach().numpy(),
    )
    assert parsed.shape == (3, 2)
    assert bool(np.isfinite(parsed).all())
    assert json.loads(CONFIG)["max_wheel_speed"] > 0.0


def _clearing_state(pose: tuple[float, float, float]) -> np.ndarray:
    state = initial_state().copy()
    offset = ROBOT_BASE + 0 * ROBOT_WIDTH
    state[offset + 2 : offset + 5] = pose
    state[5:9] = (0.30, 0.0, 0.0, 0.0)
    return state


def _clear_waypoint(state: np.ndarray) -> tuple[float, float]:
    """The waypoint _strike_target returns from the f32 state on the exit line."""
    return (float(state[5]) - (0.10 + 0.16), float(state[6]))


def _strike_tokens(heading: float) -> np.ndarray:
    tokens = np.zeros((3, 3), dtype=np.float32)
    tokens[:, 0] = -1.0
    tokens[0] = (1.0, heading / math.pi, 1.0)
    return tokens


def test_angled_strike_routes_through_a_clearing_waypoint_outside_contact() -> None:
    """ADR 0027: an approach that is behind the ball but facing away from the exit line
    targets a waypoint on the exit line beyond the acquisition point, so the run-in turns
    clear of the ball. The straight-line target is not returned anymore at this stage."""
    pose = (-0.05, 0.0, math.pi)
    state = _clearing_state(pose)
    waypoint, driving = _strike_target(
        state,
        robot_pose(state, 0),
        (1.0, 0.0),
        ball_deceleration=0.8,
        authority=1.0,
        strike_clearing_enabled=True,
    )
    assert driving
    assert waypoint[0] == pytest.approx(_clear_waypoint(state)[0], abs=1e-6)
    assert waypoint[1] == pytest.approx(0.0, abs=1e-6)
    assert math.hypot(waypoint[0] - 0.30, waypoint[1]) >= 0.082


def test_waypoint_approach_uses_reduced_both_wheel_authority() -> None:
    """When the clearing waypoint is the current target, both wheel requests are scaled by
    the approach authority so the commanded path is unchanged but the yaw builds against
    the acceleration limit and the tracked arc pulls clear of the ball."""
    pose = (-0.05, 0.0, math.pi)
    state = _clearing_state(pose)
    tokens = _strike_tokens(0.0)
    on = circular_primitive_wheel_actions(
        state, team=0, tokens=tokens, strike_clearing_enabled=True
    )[0]
    off = circular_primitive_wheel_actions(
        state, team=0, tokens=tokens, strike_clearing_enabled=False
    )[0]
    expected = go_to_target(robot_pose(state, 0), _clear_waypoint(state), settle=True) * np.float32(
        0.35
    )
    np.testing.assert_allclose(on, expected, rtol=0.0, atol=1e-6)
    assert bool((np.abs(off) > np.abs(on)).any())


def test_acquisition_reaim_keeps_full_both_wheel_authority() -> None:
    """Inside the clearing region the robot re-aims at the acquisition point; that turn is
    a turn in place (the settle taper zeroes the forward), so scaling it would turn the
    release into a crawl. It runs at full turning authority."""
    pose = (0.10, 0.0, math.pi)  # 0.10 short of the acquisition point, behind the ball, facing away
    state = _clearing_state(pose)
    target, driving = _strike_target(
        state,
        robot_pose(state, 0),
        (1.0, 0.0),
        ball_deceleration=0.8,
        authority=1.0,
        strike_clearing_enabled=True,
    )
    assert driving
    assert target[0] == pytest.approx(_clear_waypoint(state)[0] + 0.16, abs=1e-6)
    on = circular_primitive_wheel_actions(
        state, team=0, tokens=_strike_tokens(0.0), strike_clearing_enabled=True
    )[0]
    np.testing.assert_allclose(
        on,
        go_to_target(robot_pose(state, 0), (_clear_waypoint(state)[0] + 0.16, 0.0), settle=True),
        rtol=0.0,
        atol=1e-6,
    )


def test_drive_through_is_unchanged_when_aligned() -> None:
    """A robot already on the acquisition point facing the exit drives through exactly as
    before: the clearing branch never fires and the enabled/disabled paths are identical."""
    pose = (0.20, 0.0, 0.0)  # on the acquisition point, facing the exit
    state = _clearing_state(pose)
    for enabled in (True, False):
        target, driving = _strike_target(
            state,
            robot_pose(state, 0),
            (1.0, 0.0),
            ball_deceleration=0.8,
            authority=1.0,
            strike_clearing_enabled=enabled,
        )
        assert driving
        assert target[0] == pytest.approx(0.58, abs=1e-6)
    on = circular_primitive_wheel_actions(
        state, team=0, tokens=_strike_tokens(0.0), strike_clearing_enabled=True
    )[0]
    off = circular_primitive_wheel_actions(
        state, team=0, tokens=_strike_tokens(0.0), strike_clearing_enabled=False
    )[0]
    np.testing.assert_allclose(on, off, rtol=0.0, atol=0.0)


def test_clearing_leaves_navigate_and_stop_untouched() -> None:
    """The clearing branch lives entirely inside the strike path; navigate and stop produce
    byte-identical wheel requests with the knob either way."""
    state = initial_state()
    tokens = np.zeros((3, 3), dtype=np.float32)
    tokens[:, 0] = -1.0
    tokens[0] = (0.0, 0.0, 1.0)  # navigate
    on = circular_primitive_wheel_actions(
        state, team=0, tokens=tokens, strike_clearing_enabled=True
    )
    off = circular_primitive_wheel_actions(
        state, team=0, tokens=tokens, strike_clearing_enabled=False
    )
    np.testing.assert_allclose(on, off, rtol=0.0, atol=0.0)
    tokens[0] = (-1.0, 0.0, 1.0)  # stop
    stopped = circular_primitive_wheel_actions(
        state, team=0, tokens=tokens, strike_clearing_enabled=True
    )
    np.testing.assert_allclose(stopped, np.zeros_like(stopped), rtol=0.0, atol=0.0)


def _angled_striker(ball: tuple[float, float], reach: float, degrees: float) -> str:
    """One striker `degrees` off the shooting line around the ball; everyone else parked."""
    state = json.loads(STATE)
    state.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    state["ball"].update(x=ball[0], y=ball[1], vx=0.0, vy=0.0, omega=0.0)
    goal_x = json.loads(CONFIG)["field"]["length"] / 2.0
    shooting = math.atan2(-ball[1], goal_x - ball[0])
    bearing = shooting + math.pi + math.radians(degrees)
    for index, robot in enumerate(state["robots"]):
        robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
        robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0)
        if index == 0:
            robot["pose"].update(
                x=ball[0] + reach * math.cos(bearing),
                y=ball[1] + reach * math.sin(bearing),
                theta=bearing + math.pi,
            )
        else:
            robot["pose"].update(x=-0.65, y=-0.45 + 0.2 * index, theta=0.0)
    return json.dumps(state)


def test_clearing_run_in_stays_outside_the_ball_contact_radius() -> None:
    """The regression the probe measured: at 60° a clearing run-in keeps the robot clear of
    the ball's contact radius while the straight-line approach both breaks contact and knocks
    the ball off line."""
    config = json.loads(CONFIG)
    goal_x = config["field"]["length"] / 2.0
    max_wheel = float(config["max_wheel_speed"])
    for enabled, expected in ((True, True), (False, False)):
        simulator = BatchSimulator(CONFIG, _angled_striker((0.30, 0.0), 0.26, 60.0), 1)
        state = np.asarray(simulator.reset())[0]
        min_gap = float("inf")
        for _ in range(120):
            if int(state[-1]) & 1:
                break
            heading = math.atan2(-state[6], goal_x - state[5])
            wheels = circular_primitive_wheel_actions(
                state, team=0, tokens=_strike_tokens(heading), strike_clearing_enabled=enabled
            )
            command = np.zeros((1, 6, 2), dtype=np.float32)
            command[0, :3] = wheels * max_wheel
            state = np.asarray(simulator.step_repeated(command, 4))[0]
            rx, ry, _ = robot_pose(state, 0)
            gap = math.hypot(state[5] - rx, state[6] - ry)
            min_gap = min(min_gap, gap)
        assert (min_gap >= 0.082) == expected
