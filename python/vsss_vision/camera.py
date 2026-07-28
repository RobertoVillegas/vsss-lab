"""Seeded overhead-camera measurement generation for simulation evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vsss_vision.contracts import Association, BallMeasurement, RobotMeasurement


@dataclass(frozen=True)
class CameraPerturbationProfile:
    profile_id: str = "m12-camera-reference-v1"
    latency_seconds: float = 0.02
    position_noise_std: float = 0.002
    heading_noise_std: float = 0.01
    occlusion_probability: float = 0.0
    false_detection_probability: float = 0.0
    misassociation_probability: float = 0.0

    def __post_init__(self) -> None:
        probabilities = (
            self.occlusion_probability,
            self.false_detection_probability,
            self.misassociation_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("camera perturbation probabilities must be in [0, 1]")
        if min(self.latency_seconds, self.position_noise_std, self.heading_noise_std) < 0.0:
            raise ValueError("camera latency and noise must be non-negative")


@dataclass(frozen=True)
class CameraFrame:
    capture_time: float
    arrival_time: float
    source_sequence: int
    ball: BallMeasurement | None
    robots: tuple[RobotMeasurement, ...]


class SyntheticCamera:
    def __init__(self, profile: CameraPerturbationProfile, seed: int) -> None:
        self.profile = profile
        self._rng = np.random.default_rng(seed)
        self._sequence = 0

    def observe(self, truth: dict[str, Any]) -> CameraFrame:
        """Sample detections without mutating the authoritative truth mapping."""
        capture_time = float(truth["simulation_time"])
        arrival_time = capture_time + self.profile.latency_seconds
        sequence = self._sequence
        self._sequence += 1
        ball_data = truth["ball"]
        ball = None
        if not self._occluded():
            ball = BallMeasurement(
                capture_time=capture_time,
                arrival_time=arrival_time,
                source_sequence=sequence,
                calibration_id=self.profile.profile_id,
                x=self._position(float(ball_data["x"])),
                y=self._position(float(ball_data["y"])),
                confidence=self._confidence(),
            )
        robots = []
        for marker_id, robot in enumerate(truth["robots"]):
            if self._occluded():
                continue
            pose = robot["pose"]
            misassociated = self._rng.random() < self.profile.misassociation_probability
            association = Association(
                marker_id=(marker_id + 1) % len(truth["robots"]) if misassociated else marker_id,
                confidence=0.35 if misassociated else self._confidence(),
                ambiguous=misassociated,
            )
            robots.append(
                RobotMeasurement(
                    capture_time=capture_time,
                    arrival_time=arrival_time,
                    source_sequence=sequence,
                    calibration_id=self.profile.profile_id,
                    x=self._position(float(pose["x"])),
                    y=self._position(float(pose["y"])),
                    theta=self._angle(float(pose["theta"])),
                    association=association,
                )
            )
        return CameraFrame(capture_time, arrival_time, sequence, ball, tuple(robots))

    def _occluded(self) -> bool:
        return bool(self._rng.random() < self.profile.occlusion_probability)

    def _position(self, value: float) -> float:
        sampled = value + float(self._rng.normal(0.0, self.profile.position_noise_std))
        if self._rng.random() < self.profile.false_detection_probability:
            sampled += float(self._rng.uniform(-0.25, 0.25))
        return sampled

    def _angle(self, value: float) -> float:
        sampled = value + float(self._rng.normal(0.0, self.profile.heading_noise_std))
        return (sampled + math.pi) % (2.0 * math.pi) - math.pi

    def _confidence(self) -> float:
        return float(self._rng.uniform(0.85, 1.0))
