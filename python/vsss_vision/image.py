"""Minimal CPU image ingestion for Gazebo and calibrated overhead cameras."""

from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vsss_vision.contracts import BallEstimate, BallMeasurement, EstimatorCalibration
from vsss_vision.filters import BallKalmanFilter

ByteImage = NDArray[np.uint8]


@dataclass(frozen=True)
class OverheadImageCalibration:
    calibration_id: str = "gazebo-top-camera-v1"
    center_u: float = 340.0
    center_v: float = 260.0
    pixels_per_world_x: float = 322.0
    pixels_per_world_y: float = 324.0

    def pixel_to_world(self, u: float, v: float) -> tuple[float, float]:
        return (
            (v - self.center_v) / self.pixels_per_world_x,
            (self.center_u - u) / self.pixels_per_world_y,
        )


@dataclass(frozen=True)
class RawCameraImage:
    width: int
    height: int
    pixels: ByteImage
    capture_time: float
    source_sequence: int


@dataclass(frozen=True)
class ImagePipelineTiming:
    decode_ms: float
    segmentation_ms: float
    association_ms: float
    filter_transfer_ms: float


@dataclass(frozen=True)
class BallImagePipelineFrame:
    measurement: BallMeasurement | None
    estimate: BallEstimate | None
    timing: ImagePipelineTiming


class RosStamp(Protocol):
    sec: int
    nanosec: int


class RosHeader(Protocol):
    stamp: RosStamp


class RosImage(Protocol):
    header: RosHeader
    width: int
    height: int
    step: int
    data: bytes | bytearray | memoryview


def decode_gazebo_image_pbtxt(message: str) -> RawCameraImage:
    """Decode one `gz.msgs.Image` text-format message without protobuf bindings."""
    width = _integer_field(message, "width")
    height = _integer_field(message, "height")
    step = _integer_field(message, "step")
    if step != width * 3:
        raise ValueError("only packed RGB_INT8 images are supported")
    if "pixel_format_type: RGB_INT8" not in message:
        raise ValueError("only RGB_INT8 images are supported")
    data_line = next((line for line in message.splitlines() if line.startswith("data: ")), None)
    if data_line is None:
        raise ValueError("image data is missing")
    raw = ast.literal_eval(data_line.removeprefix("data: ")).encode("latin1")
    if len(raw) != width * height * 3:
        raise ValueError("image payload length does not match dimensions")
    seconds = _integer_field(message, "sec")
    nanoseconds = _integer_field(message, "nsec")
    sequence_match = re.search(r'key: "seq"\s+value: "(\d+)"', message)
    sequence = int(sequence_match.group(1)) if sequence_match else 0
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    return RawCameraImage(width, height, pixels, seconds + nanoseconds * 1e-9, sequence)


def decode_ros_image(message: RosImage, *, source_sequence: int) -> RawCameraImage:
    """Adapt `sensor_msgs/msg/Image` by structural typing; ROS stays optional."""
    if message.step != message.width * 3:
        raise ValueError("only packed RGB8 ROS images are supported")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if len(raw) != message.width * message.height * 3:
        raise ValueError("ROS image payload length does not match dimensions")
    capture_time = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
    pixels = raw.reshape((message.height, message.width, 3))
    return RawCameraImage(
        message.width,
        message.height,
        pixels,
        capture_time,
        source_sequence,
    )


def detect_orange_ball(
    image: RawCameraImage,
    *,
    arrival_time: float,
    calibration: OverheadImageCalibration | None = None,
) -> BallMeasurement | None:
    profile = calibration or OverheadImageCalibration()
    red = image.pixels[:, :, 0]
    green = image.pixels[:, :, 1]
    blue = image.pixels[:, :, 2]
    mask = (red >= 170) & (green >= 60) & (green <= 190) & (blue <= 80) & (red > green)
    rows, columns = np.nonzero(mask)
    if len(rows) < 4:
        return None
    u = float(columns.mean())
    v = float(rows.mean())
    x, y = profile.pixel_to_world(u, v)
    compactness = min(1.0, len(rows) / 80.0)
    return BallMeasurement(
        capture_time=image.capture_time,
        arrival_time=arrival_time,
        source_sequence=image.source_sequence,
        calibration_id=profile.calibration_id,
        x=x,
        y=y,
        confidence=compactness,
    )


def profile_ball_pipeline(
    message: str,
    *,
    arrival_time: float,
    calibration: OverheadImageCalibration | None = None,
) -> tuple[BallMeasurement | None, ImagePipelineTiming]:
    started = time.perf_counter_ns()
    image = decode_gazebo_image_pbtxt(message)
    decoded = time.perf_counter_ns()
    measurement = detect_orange_ball(
        image,
        arrival_time=arrival_time,
        calibration=calibration,
    )
    segmented = time.perf_counter_ns()
    # Ball identity is unique in VSSS; no data association or device transfer is required.
    associated = time.perf_counter_ns()
    transferred = time.perf_counter_ns()
    return measurement, ImagePipelineTiming(
        decode_ms=(decoded - started) / 1e6,
        segmentation_ms=(segmented - decoded) / 1e6,
        association_ms=(associated - segmented) / 1e6,
        filter_transfer_ms=(transferred - associated) / 1e6,
    )


class BallImagePipeline:
    """Causal CPU path shared by Gazebo recordings and live ROS images."""

    def __init__(
        self,
        calibration: OverheadImageCalibration | None = None,
        estimator_calibration: EstimatorCalibration | None = None,
    ) -> None:
        self.calibration = calibration or OverheadImageCalibration()
        self.estimator_calibration = estimator_calibration or EstimatorCalibration()
        self._filter: BallKalmanFilter | None = None

    def update(self, image: RawCameraImage, *, arrival_time: float) -> BallImagePipelineFrame:
        started = time.perf_counter_ns()
        measurement = detect_orange_ball(
            image,
            arrival_time=arrival_time,
            calibration=self.calibration,
        )
        segmented = time.perf_counter_ns()
        associated = time.perf_counter_ns()
        if measurement is not None:
            if self._filter is None:
                self._filter = BallKalmanFilter.initialize(
                    measurement,
                    self.estimator_calibration,
                )
            estimate = self._filter.update(measurement)
        elif self._filter is not None:
            estimate = self._filter.predict_only(image.capture_time, arrival_time)
        else:
            estimate = None
        filtered = time.perf_counter_ns()
        return BallImagePipelineFrame(
            measurement,
            estimate,
            ImagePipelineTiming(
                decode_ms=0.0,
                segmentation_ms=(segmented - started) / 1e6,
                association_ms=(associated - segmented) / 1e6,
                filter_transfer_ms=(filtered - associated) / 1e6,
            ),
        )


def _integer_field(message: str, field: str) -> int:
    match = re.search(rf"^\s*{re.escape(field)}: (\d+)$", message, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{field} is missing")
    value = int(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value
