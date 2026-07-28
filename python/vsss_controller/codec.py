"""Safe Python builders for protocol-v1 messages."""

from __future__ import annotations

import math

import flatbuffers  # type: ignore[import-untyped]
from vsss.protocol.v1 import Action, Envelope, Hello, Payload, RobotAction

from vsss_controller.models import EnvelopeMeta, RobotCommand

_IDENTIFIER = b"VSS1"


def _finish(
    builder: flatbuffers.Builder, meta: EnvelopeMeta, payload_type: int, payload: int
) -> bytes:
    match_id = builder.CreateByteVector(meta.match_id)
    Envelope.Start(builder)
    Envelope.AddProtocolVersion(builder, 1)
    Envelope.AddMatchId(builder, match_id)
    Envelope.AddControllerSlot(builder, int(meta.slot))
    Envelope.AddSequence(builder, meta.sequence)
    Envelope.AddServerTick(builder, meta.server_tick)
    Envelope.AddSentMonotonicNs(builder, meta.sent_monotonic_ns)
    Envelope.AddDeadlineMonotonicNs(builder, meta.deadline_monotonic_ns)
    Envelope.AddPayloadType(builder, payload_type)
    Envelope.AddPayload(builder, payload)
    root = Envelope.End(builder)
    builder.Finish(root, file_identifier=_IDENTIFIER)
    return bytes(builder.Output())


def encode_hello(meta: EnvelopeMeta, controller_name: str) -> bytes:
    """Encode a protocol-v1 handshake."""
    builder = flatbuffers.Builder(256)
    name = builder.CreateString(controller_name)
    sdk_name = builder.CreateString("vsss-python")
    sdk_version = builder.CreateString("0.0.0")
    Hello.Start(builder)
    Hello.AddControllerName(builder, name)
    Hello.AddSdkName(builder, sdk_name)
    Hello.AddSdkVersion(builder, sdk_version)
    Hello.AddMinProtocolVersion(builder, 1)
    Hello.AddMaxProtocolVersion(builder, 1)
    payload = Hello.End(builder)
    return _finish(builder, meta, Payload.Payload.Hello, payload)


def encode_action(
    meta: EnvelopeMeta, commands: tuple[RobotCommand, RobotCommand, RobotCommand]
) -> bytes:
    """Encode exactly three finite robot commands."""
    if any(
        not math.isfinite(value)
        for command in commands
        for value in (command.first, command.second)
    ):
        raise ValueError("robot commands must be finite")
    builder = flatbuffers.Builder(256)
    Action.StartRobotsVector(builder, 3)
    for command in reversed(commands):
        RobotAction.CreateRobotAction(builder, int(command.mode), command.first, command.second)
    robots = builder.EndVector()
    Action.Start(builder)
    Action.AddRobots(builder, robots)
    payload = Action.End(builder)
    return _finish(builder, meta, Payload.Payload.Action, payload)


def verify(payload: bytes) -> None:
    """Reject a message without the v1 file identifier."""
    if not Envelope.Envelope.EnvelopeBufferHasIdentifier(payload, 0):
        raise ValueError("invalid VSS1 protocol identifier")
