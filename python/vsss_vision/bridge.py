"""Transport-neutral detection bridge for ROS/Gazebo and physical cameras."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vsss_vision.camera import CameraFrame
from vsss_vision.contracts import (
    Association,
    BallEstimate,
    BallMeasurement,
    EstimatorCalibration,
    RobotEstimate,
    RobotMeasurement,
)
from vsss_vision.filters import BallKalmanFilter, RobotEkf

DETECTION_SCHEMA = "vsss.camera-detections/v1"


def _finite(record: Mapping[str, Any], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def camera_frame_from_mapping(record: Mapping[str, Any]) -> CameraFrame:
    """Decode one detector message without importing a ROS client library."""
    if record.get("schema") != DETECTION_SCHEMA:
        raise ValueError(f"schema must be {DETECTION_SCHEMA!r}")
    capture_time = _finite(record, "capture_time")
    arrival_time = _finite(record, "arrival_time")
    if arrival_time < capture_time:
        raise ValueError("arrival_time must not precede capture_time")
    sequence = int(record["source_sequence"])
    if sequence < 0:
        raise ValueError("source_sequence must be non-negative")
    calibration_id = str(record["calibration_id"])
    if not calibration_id:
        raise ValueError("calibration_id must not be empty")

    ball_record = record.get("ball")
    ball = (
        BallMeasurement(
            capture_time=capture_time,
            arrival_time=arrival_time,
            source_sequence=sequence,
            calibration_id=calibration_id,
            x=_finite(ball_record, "x"),
            y=_finite(ball_record, "y"),
            confidence=_confidence(ball_record),
        )
        if isinstance(ball_record, Mapping)
        else None
    )
    robots: list[RobotMeasurement] = []
    marker_ids: set[int] = set()
    for robot_record in record.get("robots", ()):
        if not isinstance(robot_record, Mapping):
            raise ValueError("robot detection must be an object")
        marker = robot_record.get("marker_id")
        marker_id = int(marker) if marker is not None else None
        if marker_id is not None:
            if marker_id in marker_ids:
                raise ValueError("marker_id must be unique within a frame")
            marker_ids.add(marker_id)
        robots.append(
            RobotMeasurement(
                capture_time=capture_time,
                arrival_time=arrival_time,
                source_sequence=sequence,
                calibration_id=calibration_id,
                x=_finite(robot_record, "x"),
                y=_finite(robot_record, "y"),
                theta=_finite(robot_record, "theta"),
                association=Association(
                    marker_id=marker_id,
                    confidence=_confidence(robot_record),
                    ambiguous=bool(robot_record.get("ambiguous", False)),
                ),
            )
        )
    return CameraFrame(capture_time, arrival_time, sequence, ball, tuple(robots))


def camera_frame_from_json(line: str) -> CameraFrame:
    record = json.loads(line)
    if not isinstance(record, Mapping):
        raise ValueError("camera detection message must be an object")
    return camera_frame_from_mapping(record)


def _confidence(record: Mapping[str, Any]) -> float:
    confidence = _finite(record, "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return confidence


@dataclass(frozen=True)
class EstimatorFrame:
    camera: CameraFrame
    ball: BallEstimate | None
    robots: tuple[RobotEstimate, ...]


class CameraEstimatorBridge:
    """Stateful bridge from timestamped detector output to causal estimates."""

    def __init__(self, calibration: EstimatorCalibration | None = None) -> None:
        self.calibration = calibration or EstimatorCalibration()
        self._ball: BallKalmanFilter | None = None
        self._robots: dict[int, RobotEkf] = {}
        self._last_sequence = -1

    def update(self, frame: CameraFrame) -> EstimatorFrame:
        if frame.source_sequence <= self._last_sequence:
            raise ValueError("camera frames must have increasing source_sequence")
        self._last_sequence = frame.source_sequence
        ball_estimate = self._update_ball(frame)
        robot_estimates = self._update_robots(frame)
        return EstimatorFrame(frame, ball_estimate, tuple(robot_estimates))

    def _update_ball(self, frame: CameraFrame) -> BallEstimate | None:
        if frame.ball is not None:
            if self._ball is None:
                self._ball = BallKalmanFilter.initialize(frame.ball, self.calibration)
            return self._ball.update(frame.ball)
        if self._ball is None:
            return None
        return self._ball.predict_only(frame.capture_time, frame.arrival_time)

    def _update_robots(self, frame: CameraFrame) -> list[RobotEstimate]:
        estimates: list[RobotEstimate] = []
        observed: set[int] = set()
        for measurement in frame.robots:
            marker_id = measurement.association.marker_id
            if marker_id is None:
                continue
            observed.add(marker_id)
            estimator = self._robots.get(marker_id)
            if estimator is None:
                estimator = RobotEkf.initialize(measurement, self.calibration)
                self._robots[marker_id] = estimator
            estimates.append(estimator.update(measurement))
        for marker_id, estimator in self._robots.items():
            if marker_id in observed:
                continue
            estimate = estimator.predict_only(frame.capture_time, frame.arrival_time)
            if estimate is not None:
                estimates.append(estimate)
        return estimates
