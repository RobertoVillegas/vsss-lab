from __future__ import annotations

import copy
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from vsss_train.marl_env import MarlMatchEnv
from vsss_train.primitives import (
    ParametricPrimitiveSet,
    SoccerPrimitiveSet,
    canonical_direction,
    nearest_canonical_direction,
    parametric_primitive_wheel_actions,
)
from vsss_train.trajectory_diagnostics import analyze_trajectory_replay

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def test_primitive_tokens_round_trip_every_action() -> None:
    indices = torch.arange(SoccerPrimitiveSet.action_count)
    tokens = SoccerPrimitiveSet.encode(indices).numpy()
    decoded = [SoccerPrimitiveSet.decode(token) for token in tokens]
    assert decoded[0].skill == "stop"
    assert [command.skill for command in decoded[1:9]] == ["navigate"] * 8
    assert [command.skill for command in decoded[9:]] == ["strike"] * 8
    assert [command.direction_index for command in decoded[1:9]] == list(range(8))
    assert [command.direction_index for command in decoded[9:]] == list(range(8))


def test_primitive_action_table_matches_golden_contract() -> None:
    contract = json.loads((ROOT / "tests/golden/m24_primitive_actions.json").read_text())
    assert contract["action_count"] == SoccerPrimitiveSet.action_count
    tokens = SoccerPrimitiveSet.encode(torch.arange(SoccerPrimitiveSet.action_count)).numpy()
    decoded = [SoccerPrimitiveSet.decode(token) for token in tokens]
    assert [
        {"index": index, "skill": command.skill, "direction": command.direction_index}
        for index, command in enumerate(decoded)
    ] == [
        {
            "index": action["index"],
            "skill": action["skill"],
            "direction": (
                None
                if action["direction"] is None
                else (
                    "east",
                    "north_east",
                    "north",
                    "north_west",
                    "west",
                    "south_west",
                    "south",
                    "south_east",
                ).index(action["direction"])
            ),
        }
        for action in contract["actions"]
    ]


def test_parametric_primitives_preserve_continuous_heading_and_intensity() -> None:
    skills = torch.tensor([0, 1, 2])
    parameters = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0],
            [math.cos(-3 * math.pi / 8), math.sin(-3 * math.pi / 8), 1.0],
        ]
    )
    tokens = ParametricPrimitiveSet.encode(skills, parameters).numpy()
    decoded = [ParametricPrimitiveSet.decode(token) for token in tokens]
    assert [command.skill for command in decoded] == ["stop", "navigate", "strike"]
    assert decoded[1].direction == pytest.approx(math.pi / 8)
    assert decoded[2].direction == pytest.approx(-3 * math.pi / 8)
    assert decoded[0].intensity == pytest.approx(0.0)
    assert decoded[1].intensity == pytest.approx(0.5)
    assert decoded[2].intensity == pytest.approx(1.0)


def test_parametric_navigation_has_smooth_noncanonical_wheel_command() -> None:
    snapshot = copy.deepcopy(json.loads(STATE))
    snapshot["robots"][0]["enabled"] = True
    snapshot["robots"][0]["pose"].update(x=0.0, y=0.0, theta=0.0)
    environment = MarlMatchEnv(CONFIG, STATE, stage=7, horizon=2)
    environment.reset_state(snapshot)
    token = ParametricPrimitiveSet.encode(
        torch.tensor([1, 0, 0]),
        torch.tensor(
            [
                [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
            ]
        ),
    ).numpy()
    wheels = parametric_primitive_wheel_actions(
        environment.state,
        team=0,
        tokens=token,
    )
    assert 0.0 < wheels[0, 0] < wheels[0, 1] < 1.0
    assert wheels[0].mean() < 0.5


def test_mappo_ippo_primitive_configs_are_paired() -> None:
    mappo = tomllib.loads((ROOT / "experiments/configs/m24-mappo-primitives.toml").read_text())
    ippo = tomllib.loads((ROOT / "experiments/configs/m24-ippo-primitives.toml").read_text())
    assert mappo.pop("algorithm") == "mappo"
    assert ippo.pop("algorithm") == "ippo"
    mappo.pop("policy_id")
    ippo.pop("policy_id")
    assert mappo == ippo


@pytest.mark.parametrize("index", range(8))
def test_canonical_directions_are_team_reflections(index: int) -> None:
    blue = canonical_direction(index, 0)
    yellow = canonical_direction(index, 1)
    assert yellow == pytest.approx((-blue[0], -blue[1]))
    assert nearest_canonical_direction(blue, 0) == index
    assert nearest_canonical_direction(yellow, 1) == index


def test_stationary_ball_strike_contacts_and_exits_toward_command() -> None:
    snapshot = copy.deepcopy(json.loads(STATE))
    snapshot.update(tick=0, simulation_time=0.0, score_blue=0, score_yellow=0, events=0)
    snapshot["ball"].update(x=0.0, y=0.0, vx=0.0, vy=0.0, omega=0.0)
    for index, robot in enumerate(snapshot["robots"]):
        robot["enabled"] = index == 0
        robot["pose"].update(
            x=-0.28 if index == 0 else robot["pose"]["x"],
            y=0.0 if index == 0 else robot["pose"]["y"],
            theta=0.0 if index == 0 else robot["pose"]["theta"],
        )
        robot["twist"].update(vx=0.0, vy=0.0, omega=0.0)
        robot.update(wheel_speed_left=0.0, wheel_speed_right=0.0)
    environment = MarlMatchEnv(
        CONFIG,
        STATE,
        stage=7,
        horizon=500,
        action_repeat=4,
        action_parser="primitive",
        stagnation_seconds=8.0,
    )
    environment.reset_state(snapshot)
    strike_forward = SoccerPrimitiveSet.encode(torch.tensor([9, 0, 0])).numpy()
    contacted = False
    maximum_forward_velocity = 0.0
    for _ in range(400):
        _, _, done, info = environment.step(strike_forward)
        robot_x = float(environment.state[12])
        robot_y = float(environment.state[13])
        contacted |= (
            math.hypot(float(environment.state[5]) - robot_x, float(environment.state[6]) - robot_y)
            <= 0.0775
        )
        maximum_forward_velocity = max(maximum_forward_velocity, float(environment.state[7]))
        if done:
            break
    assert np.isfinite(environment.state).all()
    assert contacted
    assert maximum_forward_velocity > 0.05
    assert int(info["events"]) & 2 == 0


def test_replay_tick_episode_is_available_to_metrics(tmp_path: Path) -> None:
    # Contract fixture: metrics must reject an interception target in another episode.
    replay = tmp_path / "episodes.jsonl"
    header = {
        "type": "header",
        "config": {"control_period": 0.1},
    }
    ticks = [
        {
            "type": "tick",
            "index": 1,
            "episode": 0,
            "snapshot": {"ball": {"x": 0.0, "y": 0.0}},
            "perception": {
                "ball_estimate": None,
                "goalkeeper_interception": {"elapsed": 0.1, "x": 0.5, "y": 0.0},
            },
        },
        {
            "type": "tick",
            "index": 2,
            "episode": 1,
            "snapshot": {"ball": {"x": 0.5, "y": 0.0}},
            "perception": {
                "ball_estimate": None,
                "goalkeeper_interception": None,
            },
        },
    ]
    replay.write_text("\n".join(json.dumps(item) for item in (header, *ticks)) + "\n")
    from vsss_vision.metrics import analyze_replay

    report = analyze_replay(replay)
    assert report.goalkeeper_interception.samples == 0


def test_trajectory_diagnostics_are_episode_aware_and_retain_action_signal(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "trajectory.jsonl"
    header = {
        "type": "header",
        "config": {
            "control_period": 0.1,
            "robot": {"length": 0.075, "width": 0.075},
            "ball": {"radius": 0.0215},
        },
    }
    robots = [
        {
            "id": "blue-0",
            "team": "blue",
            "enabled": True,
            "pose": {"x": -0.4, "y": 0.0},
            "twist": {"vx": 0.0, "vy": 0.0},
        }
    ]
    ticks = [
        {
            "type": "tick",
            "episode": episode,
            "snapshot": {
                "simulation_time": time,
                "ball": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
                "robots": robots,
            },
            "actions": [[1.0, 1.0]],
        }
        for episode, time in ((0, 0.0), (0, 0.1), (1, 0.0), (1, 0.1))
    ]
    replay.write_text("\n".join(json.dumps(item) for item in (header, *ticks)) + "\n")
    report = analyze_trajectory_replay(replay)
    assert report.sampled_seconds == pytest.approx(0.2)
    assert report.stationary_remote_seconds == pytest.approx(0.2)
    assert report.mean_nearest_ball_distance == pytest.approx(0.4)
    assert report.mean_abs_action == pytest.approx(1.0)
