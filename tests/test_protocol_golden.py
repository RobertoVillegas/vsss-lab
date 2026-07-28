"""Cross-language checks for the committed FlatBuffers protocol fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

GENERATED_ROOT = Path(__file__).parents[1] / "python" / "vsss_controller" / "generated"
sys.path.insert(0, str(GENERATED_ROOT))

from vsss.protocol.v1 import Action, Envelope, Hello, Payload  # noqa: E402

GOLDEN_ROOT = Path(__file__).parent / "golden"


def _envelope(name: str) -> Envelope.Envelope:
    payload = (GOLDEN_ROOT / name).read_bytes()
    assert Envelope.Envelope.EnvelopeBufferHasIdentifier(payload, 0)
    return Envelope.Envelope.GetRootAs(payload, 0)


def test_python_decodes_rust_compatible_hello() -> None:
    envelope = _envelope("m8_hello_v1.vsss")
    hello = Hello.Hello()
    payload = envelope.Payload()
    assert payload is not None
    hello.Init(payload.Bytes, payload.Pos)

    assert envelope.ProtocolVersion() == 1
    assert envelope.PayloadType() == Payload.Payload.Hello
    assert bytes(envelope.MatchIdAsNumpy()) == b"golden-match-001"
    assert hello.ControllerName() == b"golden-python"
    assert hello.MinProtocolVersion() == 1
    assert hello.MaxProtocolVersion() == 1


def test_python_decodes_three_robot_action() -> None:
    envelope = _envelope("m8_action_v1.vsss")
    action = Action.Action()
    payload = envelope.Payload()
    assert payload is not None
    action.Init(payload.Bytes, payload.Pos)

    assert envelope.PayloadType() == Payload.Payload.Action
    assert envelope.Sequence() == 7
    assert envelope.ServerTick() == 42
    assert action.RobotsLength() == 3
    assert action.Robots(1).First() == -0.5


def test_invalid_fixture_is_rejected_by_identifier_gate() -> None:
    payload = bytearray((GOLDEN_ROOT / "m8_hello_v1.vsss").read_bytes())
    payload[4:8] = b"NOPE"
    assert not Envelope.Envelope.EnvelopeBufferHasIdentifier(payload, 0)
