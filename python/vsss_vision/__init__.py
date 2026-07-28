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
from vsss_vision.image import (
    BallImagePipeline,
    BallImagePipelineFrame,
    ImagePipelineTiming,
    OverheadImageCalibration,
    RawCameraImage,
    decode_gazebo_image_pbtxt,
    decode_ros_image,
    detect_orange_ball,
    profile_ball_pipeline,
)
from vsss_vision.metrics import ErrorSummary, VisionMetrics, analyze_replay
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
    "BallImagePipeline",
    "BallImagePipelineFrame",
    "BallKalmanFilter",
    "BallMeasurement",
    "CameraEstimatorBridge",
    "CameraFrame",
    "CameraPerturbationProfile",
    "ErrorSummary",
    "EstimatorCalibration",
    "EstimatorFrame",
    "FieldPredictionModel",
    "ImagePipelineTiming",
    "Interception",
    "OverheadImageCalibration",
    "PolicyVisionRecord",
    "Prediction",
    "PredictiveFeatures",
    "PredictiveObservationAdapter",
    "RawCameraImage",
    "RobotEkf",
    "RobotEstimate",
    "RobotMeasurement",
    "SyntheticCamera",
    "VisionMetrics",
    "analytic_ball_prediction",
    "analyze_replay",
    "camera_frame_from_json",
    "camera_frame_from_mapping",
    "collision_aware_ball_prediction",
    "decode_gazebo_image_pbtxt",
    "decode_ros_image",
    "detect_orange_ball",
    "goalkeeper_interception",
    "profile_ball_pipeline",
    "segment_interception",
]
