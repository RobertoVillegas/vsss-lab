"""Versioned contracts that keep measurements, estimates, and predictions distinct."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Association:
    marker_id: int | None
    confidence: float
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("association confidence must be in [0, 1]")


@dataclass(frozen=True)
class BallMeasurement:
    capture_time: float
    arrival_time: float
    source_sequence: int
    calibration_id: str
    x: float
    y: float
    confidence: float
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class RobotMeasurement:
    capture_time: float
    arrival_time: float
    source_sequence: int
    calibration_id: str
    x: float
    y: float
    theta: float
    association: Association
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class BallEstimate:
    effective_time: float
    update_time: float
    source_sequence: int
    state: tuple[float, float, float, float, float, float]
    covariance: tuple[tuple[float, ...], ...]
    measurement_accepted: bool
    rejection_reason: str | None
    schema_version: int = SCHEMA_VERSION

    @property
    def age(self) -> float:
        return self.update_time - self.effective_time


@dataclass(frozen=True)
class RobotEstimate:
    effective_time: float
    update_time: float
    source_sequence: int
    state: tuple[float, float, float, float, float]
    covariance: tuple[tuple[float, ...], ...]
    association: Association
    measurement_accepted: bool
    rejection_reason: str | None
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class Prediction:
    source_time: float
    generated_time: float
    model_id: str
    samples: tuple[tuple[float, float, float], ...]
    uncertainty: tuple[tuple[float, float, float], ...]
    stale: bool
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class Interception:
    team: str
    elapsed: float
    x: float
    y: float
    model_id: str
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class PredictiveFeatures:
    schema_version: int
    adapter_id: str
    available: bool
    values: tuple[float, ...]


@dataclass(frozen=True)
class PolicyVisionRecord:
    decision_time: float
    estimate: BallEstimate | None
    prediction: Prediction | None
    interception: Interception | None
    features: PredictiveFeatures
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class EstimatorCalibration:
    calibration_id: str = "m12-reference-v1"
    ball_process_variance: float = 0.08
    ball_measurement_variance: float = 0.0004
    robot_process_variance: float = 0.04
    robot_position_variance: float = 0.0004
    robot_heading_variance: float = 0.01
    innovation_gate: float = 11.34
    maximum_prediction_age: float = 0.25

    def __post_init__(self) -> None:
        positive = (
            self.ball_process_variance,
            self.ball_measurement_variance,
            self.robot_process_variance,
            self.robot_position_variance,
            self.robot_heading_variance,
            self.innovation_gate,
            self.maximum_prediction_age,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("estimator calibration values must be positive")
