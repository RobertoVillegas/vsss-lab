"""Causal camera-state estimation and prediction contracts."""

from vsss_vision.adapter import PredictiveObservationAdapter
from vsss_vision.bridge import (
    DETECTION_SCHEMA,
    CameraEstimatorBridge,
    EstimatorFrame,
    camera_frame_from_json,
    camera_frame_from_mapping,
)
from vsss_vision.camera import CameraFrame, CameraPerturbationProfile, SyntheticCamera
from vsss_vision.contracts import (
    Association,
    BallEstimate,
    BallMeasurement,
    EstimatorCalibration,
    Interception,
    PolicyVisionRecord,
    Prediction,
    PredictiveFeatures,
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
    "DETECTION_SCHEMA",
    "Association",
    "BallEstimate",
    "BallKalmanFilter",
    "BallMeasurement",
    "CameraEstimatorBridge",
    "CameraFrame",
    "CameraPerturbationProfile",
    "EstimatorCalibration",
    "EstimatorFrame",
    "FieldPredictionModel",
    "Interception",
    "PolicyVisionRecord",
    "Prediction",
    "PredictiveFeatures",
    "PredictiveObservationAdapter",
    "RobotEkf",
    "RobotEstimate",
    "RobotMeasurement",
    "SyntheticCamera",
    "analytic_ball_prediction",
    "camera_frame_from_json",
    "camera_frame_from_mapping",
    "collision_aware_ball_prediction",
    "goalkeeper_interception",
    "segment_interception",
]
