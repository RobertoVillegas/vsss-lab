"""Executable Python sample controller transport smoke."""

from __future__ import annotations

import argparse
import asyncio
import time

from vsss_controller.client import DealerClient
from vsss_controller.codec import encode_hello
from vsss_controller.models import ControllerSlot, EnvelopeMeta


async def exchange(endpoint: str) -> None:
    """Complete one verified Hello request/reply exchange."""
    client = DealerClient(endpoint, b"vsss-python-sample")
    try:
        payload = encode_hello(
            EnvelopeMeta(
                match_id=b"python-e2e-test1",
                slot=ControllerSlot.UNASSIGNED,
                sequence=1,
                server_tick=0,
                sent_monotonic_ns=time.monotonic_ns(),
                deadline_monotonic_ns=0,
            ),
            "python-stop-controller",
        )
        await client.send(payload)
        await client.receive(5_000)
    finally:
        client.close()


def main() -> None:
    """Run the sample exchange."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    arguments = parser.parse_args()
    asyncio.run(exchange(arguments.endpoint))


if __name__ == "__main__":
    main()
