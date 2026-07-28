"""Python controller SDK and cross-language codec tests."""

from __future__ import annotations

import math

from vsss_controller.codec import encode_action, encode_hello
from vsss_controller.models import (
    ControllerSlot,
    ControlMode,
    EnvelopeMeta,
    RobotCommand,
    StopController,
)


def _meta() -> EnvelopeMeta:
    return EnvelopeMeta(b"python-sdk-test1", ControllerSlot.BLUE, 4, 42, 100, 200)


def test_python_sdk_builds_v1_messages() -> None:
    assert encode_hello(_meta(), "python-test")[4:8] == b"VSS1"
    commands = StopController().act({})
    assert encode_action(_meta(), commands)[4:8] == b"VSS1"
    assert len(commands) == 3


def test_python_sdk_rejects_non_finite_action() -> None:
    invalid = RobotCommand(ControlMode.WHEEL_VELOCITY, math.nan, 0.0)
    try:
        encode_action(_meta(), (invalid, invalid, invalid))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite action was accepted")
