"""Async DEALER client with bounded receive deadlines."""

from __future__ import annotations

import zmq
import zmq.asyncio

from vsss_controller.codec import verify


class DealerClient:
    """Own one isolated DEALER connection to the authoritative server."""

    def __init__(self, endpoint: str, identity: bytes) -> None:
        if not endpoint.startswith("tcp://127.0.0.1:"):
            raise ValueError("controller endpoint must be loopback")
        if not identity:
            raise ValueError("identity must not be empty")
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.set(zmq.IDENTITY, identity)
        self._socket.set(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    async def send(self, payload: bytes) -> None:
        """Verify then send one protocol frame."""
        verify(payload)
        await self._socket.send(payload)

    async def receive(self, timeout_ms: int) -> bytes:
        """Receive one verified frame or raise ``TimeoutError``."""
        if not await self._socket.poll(timeout_ms):
            raise TimeoutError("controller receive deadline exceeded")
        payload = await self._socket.recv()
        verify(payload)
        return bytes(payload)

    def close(self) -> None:
        """Close without blocking on pending messages."""
        self._socket.close(linger=0)
        self._context.term()
