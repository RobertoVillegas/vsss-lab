"""Learned-policy evaluation replay compatible with the M4/M6 viewer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from vsss_train.marl import SharedActor, build_team_observation
from vsss_train.marl_env import MarlMatchEnv
from vsss_vision import (
    BallEstimate,
    BallKalmanFilter,
    CameraPerturbationProfile,
    EstimatorCalibration,
    Prediction,
    RobotEkf,
    SyntheticCamera,
    collision_aware_ball_prediction,
)


def run_policy_replay(
    blue: SharedActor,
    yellow: SharedActor | None,
    config_json: str,
    state_json: str,
    *,
    seed: int,
    ticks: int,
    replay_path: Path,
    blue_policy: str,
    yellow_policy: str,
) -> dict[str, Any]:
    """Evaluate learned blue versus learned or heuristic yellow and write JSONL."""
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    environment = MarlMatchEnv(config_json, state_json, stage=8, horizon=ticks)
    blue_device = next(blue.parameters()).device
    observation = environment.reset(seed)
    config = json.loads(config_json)
    camera = SyntheticCamera(CameraPerturbationProfile(), seed=seed + 50_000)
    calibration = EstimatorCalibration()
    ball_filter: BallKalmanFilter | None = None
    robot_filters: dict[int, RobotEkf] = {}
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    final_checksum = ""
    with replay_path.open("w", encoding="utf-8", newline="\n") as replay:
        _write(
            replay,
            {
                "type": "header",
                "version": 1,
                "seed": seed,
                "ticks": ticks,
                "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
                "config": config,
                "policies": {"blue": blue_policy, "yellow": yellow_policy},
            },
        )
        index = 0
        goals_blue = 0
        goals_yellow = 0
        episode = 0
        while index < ticks:
            with torch.inference_mode():
                blue_action = blue.deterministic_action(observation.to(blue_device)).cpu().numpy()
                yellow_action = (
                    yellow.deterministic_action(
                        build_team_observation(environment.state, team=1).to(
                            next(yellow.parameters()).device
                        )
                    )
                    .cpu()
                    .numpy()
                    if yellow is not None
                    else None
                )
            observation, reward, done, info = environment.step(blue_action, yellow_action)
            snapshot = environment.snapshot()
            if int(info["events"]) & 1:
                goals_blue += 1
            if int(info["events"]) & 2:
                goals_yellow += 1
            snapshot["score_blue"] = goals_blue
            snapshot["score_yellow"] = goals_yellow
            snapshot["tick"] = (index + 1) * environment.action_repeat
            snapshot["simulation_time"] = (index + 1) * float(config["control_period"])
            camera_frame = camera.observe(snapshot)
            ball_estimate: BallEstimate | None
            ball_prediction: Prediction | None
            if camera_frame.ball is not None:
                if ball_filter is None:
                    ball_filter = BallKalmanFilter.initialize(camera_frame.ball, calibration)
                ball_estimate = ball_filter.update(camera_frame.ball)
                ball_prediction = collision_aware_ball_prediction(
                    ball_estimate,
                    generated_time=camera_frame.arrival_time,
                )
            else:
                ball_estimate = (
                    ball_filter.predict_only(
                        camera_frame.capture_time,
                        camera_frame.arrival_time,
                    )
                    if ball_filter is not None
                    else None
                )
                ball_prediction = (
                    collision_aware_ball_prediction(
                        ball_estimate,
                        generated_time=camera_frame.arrival_time,
                    )
                    if ball_estimate is not None
                    else None
                )
            robot_estimates = []
            measured_markers: set[int] = set()
            for measurement in camera_frame.robots:
                marker_id = measurement.association.marker_id
                if marker_id is None:
                    continue
                measured_markers.add(marker_id)
                estimator = robot_filters.get(marker_id)
                if estimator is None:
                    estimator = RobotEkf.initialize(measurement, calibration)
                    robot_filters[marker_id] = estimator
                robot_estimates.append(estimator.update(measurement))
            for marker_id, estimator in robot_filters.items():
                if marker_id in measured_markers:
                    continue
                estimate = estimator.predict_only(
                    camera_frame.capture_time,
                    camera_frame.arrival_time,
                )
                if estimate is not None:
                    robot_estimates.append(estimate)
            canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            final_checksum = hashlib.sha256(canonical.encode()).hexdigest()
            index += 1
            actions = np.asarray(info["actions"], dtype=np.float32)
            _write(
                replay,
                {
                    "type": "tick",
                    "index": index,
                    "actions": actions.tolist(),
                    "events": int(info["events"]),
                    "checksum": final_checksum,
                    "snapshot": snapshot,
                    "perception": {
                        "policy_visible": False,
                        "camera": asdict(camera_frame),
                        "ball_estimate": (
                            asdict(ball_estimate) if ball_estimate is not None else None
                        ),
                        "robot_estimates": [asdict(estimate) for estimate in robot_estimates],
                        "ball_prediction": (
                            asdict(ball_prediction) if ball_prediction is not None else None
                        ),
                    },
                    "rewards": [reward.total] * 3 + [-reward.total] * 3,
                },
            )
            if done and index < ticks:
                episode += 1
                observation = environment.reset(seed + episode)
    return {
        "ticks": index,
        "score_blue": goals_blue,
        "score_yellow": goals_yellow,
        "outcome": (
            "win" if goals_blue > goals_yellow else "loss" if goals_blue < goals_yellow else "draw"
        ),
        "goals": goals_blue + goals_yellow,
        "simulation_seconds": ticks * float(config["control_period"]),
        "progress": environment.progress_score(),
        "final_checksum": final_checksum,
        "replay": str(replay_path.resolve()),
    }


def _write(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
