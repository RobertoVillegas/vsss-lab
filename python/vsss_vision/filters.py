"""Deterministic NumPy reference Kalman filters for overhead-camera detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vsss_vision.contracts import (
    BallEstimate,
    BallMeasurement,
    EstimatorCalibration,
    RobotEstimate,
    RobotMeasurement,
)

FloatArray = NDArray[np.float64]


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _covariance_tuple(covariance: FloatArray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in covariance)


@dataclass
class BallKalmanFilter:
    calibration: EstimatorCalibration
    state: FloatArray
    covariance: FloatArray
    effective_time: float

    @classmethod
    def initialize(
        cls, measurement: BallMeasurement, calibration: EstimatorCalibration
    ) -> BallKalmanFilter:
        state = np.array([measurement.x, 0.0, 0.0, measurement.y, 0.0, 0.0])
        return cls(calibration, state, np.eye(6), measurement.capture_time)

    def update(self, measurement: BallMeasurement) -> BallEstimate:
        dt = measurement.capture_time - self.effective_time
        if dt < 0.0:
            raise ValueError("ball measurements must be ordered by capture time")
        transition = np.array(
            [
                [1.0, dt, 0.5 * dt * dt, 0.0, 0.0, 0.0],
                [0.0, 1.0, dt, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, dt, 0.5 * dt * dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        process = np.eye(6) * self.calibration.ball_process_variance * max(dt, 1e-6)
        predicted = transition @ self.state
        predicted_covariance = transition @ self.covariance @ transition.T + process
        observation = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        noise = np.eye(2) * self.calibration.ball_measurement_variance
        innovation = np.array([measurement.x, measurement.y]) - observation @ predicted
        innovation_covariance = observation @ predicted_covariance @ observation.T + noise
        mahalanobis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        accepted = mahalanobis <= self.calibration.innovation_gate
        if accepted:
            gain = np.linalg.solve(innovation_covariance, observation @ predicted_covariance).T
            self.state = predicted + gain @ innovation
            identity = np.eye(6)
            correction = identity - gain @ observation
            self.covariance = (
                correction @ predicted_covariance @ correction.T + gain @ noise @ gain.T
            )
        else:
            self.state = predicted
            self.covariance = predicted_covariance
        self.effective_time = measurement.capture_time
        return BallEstimate(
            effective_time=self.effective_time,
            update_time=measurement.arrival_time,
            source_sequence=measurement.source_sequence,
            state=tuple(float(value) for value in self.state),  # type: ignore[arg-type]
            covariance=_covariance_tuple(self.covariance),
            measurement_accepted=accepted,
            rejection_reason=None if accepted else "innovation_gate",
        )


@dataclass
class RobotEkf:
    calibration: EstimatorCalibration
    state: FloatArray
    covariance: FloatArray
    effective_time: float

    @classmethod
    def initialize(
        cls, measurement: RobotMeasurement, calibration: EstimatorCalibration
    ) -> RobotEkf:
        state = np.array([measurement.x, measurement.y, measurement.theta, 0.0, 0.0])
        return cls(calibration, state, np.eye(5), measurement.capture_time)

    def update(self, measurement: RobotMeasurement) -> RobotEstimate:
        dt = measurement.capture_time - self.effective_time
        if dt < 0.0:
            raise ValueError("robot measurements must be ordered by capture time")
        x, y, theta, velocity, omega = self.state
        predicted = np.array(
            [
                x + velocity * math.cos(theta) * dt,
                y + velocity * math.sin(theta) * dt,
                wrap_angle(theta + omega * dt),
                velocity,
                omega,
            ]
        )
        jacobian = np.eye(5)
        jacobian[0, 2] = -velocity * math.sin(theta) * dt
        jacobian[0, 3] = math.cos(theta) * dt
        jacobian[1, 2] = velocity * math.cos(theta) * dt
        jacobian[1, 3] = math.sin(theta) * dt
        jacobian[2, 4] = dt
        process = np.eye(5) * self.calibration.robot_process_variance * max(dt, 1e-6)
        predicted_covariance = jacobian @ self.covariance @ jacobian.T + process
        observation = np.zeros((3, 5))
        observation[0, 0] = observation[1, 1] = observation[2, 2] = 1.0
        noise = np.diag(
            [
                self.calibration.robot_position_variance,
                self.calibration.robot_position_variance,
                self.calibration.robot_heading_variance,
            ]
        )
        innovation = (
            np.array([measurement.x, measurement.y, measurement.theta]) - observation @ predicted
        )
        innovation[2] = wrap_angle(float(innovation[2]))
        innovation_covariance = observation @ predicted_covariance @ observation.T + noise
        mahalanobis = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        accepted = mahalanobis <= self.calibration.innovation_gate
        if accepted:
            gain = np.linalg.solve(innovation_covariance, observation @ predicted_covariance).T
            self.state = predicted + gain @ innovation
            self.state[2] = wrap_angle(float(self.state[2]))
            identity = np.eye(5)
            correction = identity - gain @ observation
            self.covariance = (
                correction @ predicted_covariance @ correction.T + gain @ noise @ gain.T
            )
        else:
            self.state = predicted
            self.covariance = predicted_covariance
        self.effective_time = measurement.capture_time
        return RobotEstimate(
            effective_time=self.effective_time,
            update_time=measurement.arrival_time,
            source_sequence=measurement.source_sequence,
            state=tuple(float(value) for value in self.state),  # type: ignore[arg-type]
            covariance=_covariance_tuple(self.covariance),
            association=measurement.association,
            measurement_accepted=accepted,
            rejection_reason=None if accepted else "innovation_gate",
        )
