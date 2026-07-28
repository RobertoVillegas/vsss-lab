from dataclasses import dataclass

import numpy as np
import pytest
from vsss_vision.image import (
    BallImagePipeline,
    OverheadImageCalibration,
    RawCameraImage,
    decode_ros_image,
    detect_orange_ball,
)


@dataclass
class Stamp:
    sec: int
    nanosec: int


@dataclass
class Header:
    stamp: Stamp


@dataclass
class ImageMessage:
    header: Header
    width: int
    height: int
    step: int
    data: bytes


def test_orange_ball_detection_projects_pixels_to_world() -> None:
    pixels = np.zeros((7, 7, 3), dtype=np.uint8)
    pixels[2:5, 3:6] = [220, 120, 10]
    image = RawCameraImage(7, 7, pixels, capture_time=1.0, source_sequence=4)
    calibration = OverheadImageCalibration(
        center_u=3.0,
        center_v=3.0,
        pixels_per_world_x=10.0,
        pixels_per_world_y=10.0,
    )

    measurement = detect_orange_ball(image, arrival_time=1.02, calibration=calibration)

    assert measurement is not None
    assert measurement.x == pytest.approx(0.0)
    assert measurement.y == pytest.approx(-0.1)
    assert measurement.source_sequence == 4


def test_orange_ball_detection_does_not_fabricate_missing_ball() -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    image = RawCameraImage(4, 4, pixels, capture_time=0.0, source_sequence=0)

    assert detect_orange_ball(image, arrival_time=0.02) is None


def test_ros_image_adapter_requires_no_ros_runtime_dependency() -> None:
    message = ImageMessage(Header(Stamp(4, 500_000_000)), 2, 1, 6, bytes(range(6)))

    image = decode_ros_image(message, source_sequence=9)

    assert image.capture_time == pytest.approx(4.5)
    assert image.source_sequence == 9
    assert image.pixels.tolist() == [[[0, 1, 2], [3, 4, 5]]]


def test_cpu_image_pipeline_filters_detection_and_dropout() -> None:
    pixels = np.zeros((7, 7, 3), dtype=np.uint8)
    pixels[2:5, 2:5] = [220, 120, 10]
    calibration = OverheadImageCalibration(
        center_u=3.0,
        center_v=3.0,
        pixels_per_world_x=10.0,
        pixels_per_world_y=10.0,
    )
    pipeline = BallImagePipeline(calibration)

    detected = pipeline.update(
        RawCameraImage(7, 7, pixels, capture_time=1.0, source_sequence=1),
        arrival_time=1.02,
    )
    missing = pipeline.update(
        RawCameraImage(7, 7, np.zeros_like(pixels), capture_time=1.02, source_sequence=2),
        arrival_time=1.04,
    )

    assert detected.estimate is not None
    assert detected.estimate.measurement_accepted
    assert missing.estimate is not None
    assert missing.estimate.rejection_reason == "measurement_missing"
