"""Causal camera-state estimation and prediction contracts."""

from vsss_vision.camera import CameraFrame, CameraPerturbationProfile, SyntheticCamera
from vsss_vision.contracts import (
    Association,
    BallEstimate,
    BallMeasurement,
    EstimatorCalibration,
    Interception,
    Prediction,
    RobotEstimate,
    RobotMeasurement,
)
from vsss_vision.filters import BallKalmanFilter, RobotEkf
from vsss_vision.prediction import (
    FieldPredictionModel,
    analytic_ball_prediction,
    collision_aware_ball_prediction,
    goalkeeper_interception,
    segment_interception,
)

__all__ = [
    "Association",
    "BallEstimate",
    "BallKalmanFilter",
    "BallMeasurement",
    "CameraFrame",
    "CameraPerturbationProfile",
    "EstimatorCalibration",
    "FieldPredictionModel",
    "Interception",
    "Prediction",
    "RobotEkf",
    "RobotEstimate",
    "RobotMeasurement",
    "SyntheticCamera",
    "analytic_ball_prediction",
    "collision_aware_ball_prediction",
    "goalkeeper_interception",
    "segment_interception",
]
