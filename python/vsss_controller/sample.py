"""Executable Python sample controller transport smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from vsss.protocol.v1 import Capabilities, Envelope, MatchResult, Observation, Payload, Reset

from vsss_controller.client import DealerClient
from vsss_controller.codec import encode_action, encode_hello
from vsss_controller.models import ControllerSlot, EnvelopeMeta, StopController


async def exchange(endpoint: str, *, exchange_only: bool = False) -> None:
    """Run the safe sample controller until the match result arrives."""
    client = DealerClient(endpoint, b"vsss-python-sample")
    match_id = b"external-match01"
    slot = ControllerSlot.UNASSIGNED
    sequence = 1
    controller = StopController()
    try:
        payload = encode_hello(
            EnvelopeMeta(
                match_id=match_id,
                slot=slot,
                sequence=sequence,
                server_tick=0,
                sent_monotonic_ns=time.monotonic_ns(),
                deadline_monotonic_ns=0,
            ),
            "python-stop-controller",
        )
        await client.send(payload)
        while True:
            envelope = Envelope.Envelope.GetRootAs(await client.receive(5_000), 0)
            if exchange_only:
                return
            payload_table = envelope.Payload()
            if payload_table is None:
                raise ValueError("server envelope has no payload")
            if envelope.PayloadType() == Payload.Payload.Capabilities:
                capabilities = Capabilities.Capabilities()
                capabilities.Init(payload_table.Bytes, payload_table.Pos)
                slot = ControllerSlot(capabilities.AssignedSlot())
                continue
            if envelope.PayloadType() == Payload.Payload.Reset:
                reset = Reset.Reset()
                reset.Init(payload_table.Bytes, payload_table.Pos)
                controller.on_reset(
                    json.loads(reset.Config().CanonicalJson()),
                    json.loads(reset.InitialStateJson()),
                )
                continue
            if envelope.PayloadType() == Payload.Payload.Observation:
                observation = Observation.Observation()
                observation.Init(payload_table.Bytes, payload_table.Pos)
                sequence += 1
                await client.send(
                    encode_action(
                        EnvelopeMeta(
                            match_id=match_id,
                            slot=slot,
                            sequence=sequence,
                            server_tick=envelope.ServerTick(),
                            sent_monotonic_ns=time.monotonic_ns(),
                            deadline_monotonic_ns=envelope.DeadlineMonotonicNs(),
                        ),
                        controller.act(json.loads(observation.CanonicalStateJson())),
                    )
                )
                continue
            if envelope.PayloadType() == Payload.Payload.MatchResult:
                result = MatchResult.MatchResult()
                result.Init(payload_table.Bytes, payload_table.Pos)
                controller.on_result(
                    {
                        "score_blue": result.ScoreBlue(),
                        "score_yellow": result.ScoreYellow(),
                        "reason": result.Reason().decode() if result.Reason() else None,
                    }
                )
                return
            raise ValueError(f"unsupported server payload {envelope.PayloadType()}")
    finally:
        client.close()


def main() -> None:
    """Run the sample exchange."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--exchange-only", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(exchange(arguments.endpoint, exchange_only=arguments.exchange_only))


if __name__ == "__main__":
    main()
