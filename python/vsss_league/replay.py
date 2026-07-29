"""Learned-policy evaluation replay compatible with the M4/M6 viewer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from vsss_train.ablations import (
    EntityAttentionActor,
    LatticeSharedActor,
    RecurrentSharedActor,
    RecurrentState,
)
from vsss_train.marl import RoleSharedActor, SharedActor, build_team_observation
from vsss_train.marl_env import MarlMatchEnv
from vsss_train.roles import assign_roles
from vsss_vision import (
    BallEstimate,
    BallKalmanFilter,
    CameraPerturbationProfile,
    EstimatorCalibration,
    Prediction,
    RobotEkf,
    SyntheticCamera,
    collision_aware_ball_prediction,
    goalkeeper_interception,
)


def run_policy_replay(
    blue: SharedActor
    | RoleSharedActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor,
    yellow: SharedActor
    | RoleSharedActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor
    | None,
    config_json: str,
    state_json: str,
    *,
    seed: int,
    ticks: int,
    replay_path: Path,
    blue_policy: str,
    yellow_policy: str,
    semantic_context: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Evaluate learned blue versus learned or heuristic yellow and write JSONL."""
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    environment = MarlMatchEnv(config_json, state_json, stage=8, horizon=ticks)
    blue_device = next(blue.parameters()).device
    observation = environment.reset(seed)
    blue_recurrent = _initial_recurrent(blue, blue_device)
    yellow_recurrent = (
        _initial_recurrent(yellow, next(yellow.parameters()).device) if yellow is not None else None
    )
    config = json.loads(config_json)
    camera = SyntheticCamera(CameraPerturbationProfile(), seed=seed + 50_000)
    calibration = EstimatorCalibration()
    ball_filter: BallKalmanFilter | None = None
    robot_filters: dict[int, RobotEkf] = {}
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path = replay_path.with_suffix(".analysis.jsonl")
    pending_replay_path = replay_path.with_name(f"{replay_path.name}.partial")
    pending_analysis_path = analysis_path.with_name(f"{analysis_path.name}.partial")
    pending_replay_path.unlink(missing_ok=True)
    pending_analysis_path.unlink(missing_ok=True)
    final_checksum = ""
    pending_predictions: list[dict[str, Any]] = []
    with (
        pending_replay_path.open("w", encoding="utf-8", newline="\n") as replay,
        pending_analysis_path.open("w", encoding="utf-8", newline="\n") as analysis,
    ):
        header: dict[str, Any] = {
            "type": "header",
            "version": 1,
            "seed": seed,
            "ticks": ticks,
            "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
            "config": config,
            "policies": {"blue": blue_policy, "yellow": yellow_policy},
        }
        if semantic_context is not None:
            header["semantic_context"] = semantic_context
        _write(replay, header)
        _write(
            analysis,
            {
                "type": "analysis_header",
                "version": 1,
                "source_replay": replay_path.name,
                "policy_visible": False,
            },
        )
        index = 0
        goals_blue = 0
        goals_yellow = 0
        episode = 0
        while index < ticks:
            with torch.inference_mode():
                blue_action, blue_recurrent = _policy_action(
                    blue,
                    observation.to(blue_device),
                    blue_recurrent,
                )
                yellow_action: np.ndarray[Any, np.dtype[np.float32]] | None = None
                if yellow is not None:
                    yellow_device = next(yellow.parameters()).device
                    yellow_action, yellow_recurrent = _policy_action(
                        yellow,
                        build_team_observation(environment.state, team=1).to(yellow_device),
                        yellow_recurrent,
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
            current_index = index + 1
            remaining_predictions = []
            for pending in pending_predictions:
                if pending["episode"] != episode:
                    continue
                if pending["target_index"] > current_index:
                    remaining_predictions.append(pending)
                    continue
                error_x = float(snapshot["ball"]["x"]) - pending["predicted_x"]
                error_y = float(snapshot["ball"]["y"]) - pending["predicted_y"]
                _write(
                    analysis,
                    {
                        "type": "prediction_error",
                        "source_index": pending["source_index"],
                        "target_index": current_index,
                        "elapsed": pending["elapsed"],
                        "error_x": error_x,
                        "error_y": error_y,
                        "error": math.hypot(error_x, error_y),
                        "episode": episode,
                    },
                )
            pending_predictions = remaining_predictions
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
            interception = (
                goalkeeper_interception(ball_prediction) if ball_prediction is not None else None
            )
            if ball_prediction is not None:
                for elapsed, predicted_x, predicted_y in ball_prediction.samples[1:]:
                    pending_predictions.append(
                        {
                            "source_index": current_index,
                            "target_index": current_index
                            + round(elapsed / float(config["control_period"])),
                            "elapsed": elapsed,
                            "predicted_x": predicted_x,
                            "predicted_y": predicted_y,
                            "episode": episode,
                        }
                    )
            canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            final_checksum = hashlib.sha256(canonical.encode()).hexdigest()
            index += 1
            actions = np.asarray(info["actions"], dtype=np.float32)
            yellow_roles = assign_roles(environment.state, 1)
            _write(
                replay,
                {
                    "type": "tick",
                    "index": index,
                    "actions": actions.tolist(),
                    "roles": list(info["roles"]) + list(yellow_roles.roles),
                    "role_changes": list(info["role_changes"]) + list(yellow_roles.changed),
                    "coverage_uncovered": {
                        "blue": bool(info["coverage_uncovered"]),
                        "yellow": yellow_roles.uncovered,
                    },
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
                        "goalkeeper_interception": (
                            asdict(interception) if interception is not None else None
                        ),
                    },
                    "rewards": [reward.total] * 3 + [-reward.total] * 3,
                },
            )
            if done and index < ticks:
                pending_predictions.clear()
                episode += 1
                observation = environment.reset(seed + episode)
                blue_recurrent = _initial_recurrent(blue, blue_device)
                yellow_recurrent = (
                    _initial_recurrent(yellow, next(yellow.parameters()).device)
                    if yellow is not None
                    else None
                )
    pending_replay_path.replace(replay_path)
    pending_analysis_path.replace(analysis_path)
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
        "analysis": str(analysis_path.resolve()),
    }


def _write(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _initial_recurrent(
    actor: SharedActor
    | RoleSharedActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor,
    device: torch.device,
) -> RecurrentState | None:
    if not isinstance(actor, RecurrentSharedActor):
        return None
    return RecurrentState(torch.zeros((3, actor.hidden_size), dtype=torch.float32, device=device))


def _policy_action(
    actor: SharedActor
    | RoleSharedActor
    | RecurrentSharedActor
    | EntityAttentionActor
    | LatticeSharedActor,
    observation: Any,
    recurrent: RecurrentState | None,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], RecurrentState | None]:
    if isinstance(actor, RecurrentSharedActor):
        if recurrent is None:
            raise AssertionError("recurrent actor requires state")
        mean, _, recurrent = actor.forward_with_state(observation, recurrent)
        return torch.tanh(mean).cpu().numpy(), recurrent
    return actor.deterministic_action(observation).cpu().numpy(), None
