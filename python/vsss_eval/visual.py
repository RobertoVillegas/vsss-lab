"""Backend-neutral visual frames and observer delivery semantics."""

from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol


@dataclass(frozen=True)
class VisualFrame:
    """Exact simulation tick adapted for visualization and observability."""

    version: int
    tick: int
    simulation_time: float
    snapshot: dict[str, Any]
    actions: list[list[float]]
    events: int
    checksum: str
    rewards: list[float] | None = None

    @classmethod
    def from_replay_record(
        cls,
        record: dict[str, Any],
        *,
        timestep: float,
    ) -> "VisualFrame":
        """Adapt an M4 replay tick without executing physics."""
        snapshot = record["snapshot"]
        tick = int(snapshot["tick"])
        return cls(
            version=1,
            tick=tick,
            simulation_time=float(snapshot.get("simulation_time", tick * timestep)),
            snapshot=snapshot,
            actions=record["actions"],
            events=int(record["events"]),
            checksum=str(record["checksum"]),
        )


class FrameSink(Protocol):
    """Consumer of exact visual frames."""

    sample_every: int

    def publish(self, frame: VisualFrame) -> None:
        """Accept a completed frame."""


class NullSink:
    """No-op observer for explicit headless execution."""

    sample_every = 1

    def publish(self, frame: VisualFrame) -> None:
        """Discard a frame."""


class LatestFrameSink:
    """Bounded live sink that keeps only the newest unconsumed frame."""

    def __init__(self, *, sample_every: int = 1) -> None:
        if sample_every <= 0:
            raise ValueError("sample_every must be positive")
        self._latest: VisualFrame | None = None
        self._lock = Lock()
        self.sample_every = sample_every
        self.seen = 0
        self.published = 0
        self.dropped = 0

    def publish(self, frame: VisualFrame) -> None:
        """Replace a stale frame instead of blocking its producer."""
        with self._lock:
            self.seen += 1
            if self._latest is not None:
                self.dropped += 1
            self._latest = frame
            self.published += 1

    def consume_latest(self) -> VisualFrame | None:
        """Return and clear the newest available frame."""
        with self._lock:
            frame = self._latest
            self._latest = None
            return frame


class MetricsSink:
    """Constant-memory aggregate metrics for an observed match."""

    def __init__(self) -> None:
        self.sample_every = 1
        self.frames = 0
        self.goals = 0
        self.last_tick = 0

    def publish(self, frame: VisualFrame) -> None:
        """Aggregate frame and goal counts."""
        self.frames += 1
        self.goals += int(bool(frame.events & 0b11))
        self.last_tick = frame.tick
