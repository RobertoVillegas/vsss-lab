import math
from dataclasses import asdict

import numpy as np
import pytest
from vsss_vision import (
    Association,
    BallKalmanFilter,
    BallMeasurement,
    EstimatorCalibration,
    RobotEkf,
    RobotMeasurement,
)


def ball_measurement(sequence: int, time: float, x: float, y: float) -> BallMeasurement:
    return BallMeasurement(time, time + 0.02, sequence, "fixture", x, y, 1.0)


def robot_measurement(
    sequence: int, time: float, x: float, y: float, theta: float
) -> RobotMeasurement:
    return RobotMeasurement(
        time,
        time + 0.02,
        sequence,
        "fixture",
        x,
        y,
        theta,
        Association(marker_id=7, confidence=0.95),
    )


def test_ball_filter_estimates_velocity_and_rejects_outlier() -> None:
    calibration = EstimatorCalibration(innovation_gate=20.0)
    estimator = BallKalmanFilter.initialize(ball_measurement(0, 0.0, 0.0, 0.0), calibration)

    for sequence in range(1, 6):
        estimate = estimator.update(
            ball_measurement(sequence, sequence * 0.1, sequence * 0.02, 0.0)
        )

    assert estimate.measurement_accepted
    assert estimate.state[1] > 0.1
    rejected = estimator.update(ball_measurement(6, 0.6, 10.0, 10.0))
    assert not rejected.measurement_accepted
    assert rejected.rejection_reason == "innovation_gate"
    assert rejected.state[0] < 1.0


def test_robot_ekf_wraps_heading_innovation() -> None:
    calibration = EstimatorCalibration(innovation_gate=20.0)
    estimator = RobotEkf.initialize(
        robot_measurement(0, 0.0, 0.0, 0.0, math.pi - 0.02), calibration
    )

    estimate = estimator.update(robot_measurement(1, 0.1, 0.0, 0.0, -math.pi + 0.02))

    assert estimate.measurement_accepted
    assert abs(abs(estimate.state[2]) - math.pi) < 0.1
    assert np.asarray(estimate.covariance).shape == (5, 5)


def test_measurements_must_be_capture_ordered() -> None:
    estimator = BallKalmanFilter.initialize(
        ball_measurement(1, 1.0, 0.0, 0.0), EstimatorCalibration()
    )

    try:
        estimator.update(ball_measurement(0, 0.9, 0.0, 0.0))
    except ValueError as error:
        assert str(error) == "ball measurements must be ordered by capture time"
    else:
        raise AssertionError("out-of-order measurement was accepted")


def test_dropout_prediction_is_bounded_by_calibration_age() -> None:
    calibration = EstimatorCalibration(maximum_prediction_age=0.25)
    ball = BallKalmanFilter.initialize(ball_measurement(0, 0.0, 0.0, 0.0), calibration)
    robot = RobotEkf.initialize(robot_measurement(0, 0.0, 0.0, 0.0, 0.0), calibration)

    ball_prediction = ball.predict_only(0.2, 0.22)
    robot_prediction = robot.predict_only(0.2, 0.22)

    assert ball_prediction is not None
    assert robot_prediction is not None
    assert ball_prediction.rejection_reason == "measurement_missing"
    assert robot_prediction.rejection_reason == "measurement_missing"
    assert ball.predict_only(0.5, 0.52) is None
    assert robot.predict_only(0.5, 0.52) is None


def test_ball_filter_golden_transition_and_covariance() -> None:
    estimator = BallKalmanFilter.initialize(
        ball_measurement(0, 0.0, 0.0, 0.0), EstimatorCalibration()
    )

    estimate = estimator.update(ball_measurement(1, 0.1, 0.02, -0.01))

    assert estimate.state == pytest.approx(
        (
            0.019992144733289148,
            0.0019736357611017013,
            0.00009819083388565681,
            -0.009996072366644574,
            -0.0009868178805508506,
            -0.000049095416942828404,
        )
    )
    covariance = np.asarray(estimate.covariance)
    assert covariance[0, 0] == pytest.approx(0.000399842895)
    assert covariance[1, 1] == pytest.approx(1.0080824803)
    assert covariance[0, 3] == 0.0


def test_measurement_and_estimate_contracts_are_not_interchangeable() -> None:
    measurement = ball_measurement(0, 0.0, 0.0, 0.0)
    estimator = BallKalmanFilter.initialize(measurement, EstimatorCalibration())
    estimate = estimator.update(measurement)

    assert set(asdict(measurement)) != set(asdict(estimate))
    with pytest.raises(TypeError):
        BallMeasurement(**asdict(estimate))
