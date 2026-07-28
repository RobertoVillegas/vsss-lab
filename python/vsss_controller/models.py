"""Transport-independent controller types and callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol


class ControllerSlot(IntEnum):
    """Ephemeral team assignment for one match."""

    UNASSIGNED = 0
    BLUE = 1
    YELLOW = 2


class ControlMode(IntEnum):
    """Interpretation of a robot command."""

    WHEEL_VELOCITY = 0
    BODY_VELOCITY = 1


@dataclass(frozen=True, slots=True)
class EnvelopeMeta:
    """Metadata shared by every wire message."""

    match_id: bytes
    slot: ControllerSlot
    sequence: int
    server_tick: int
    sent_monotonic_ns: int
    deadline_monotonic_ns: int

    def __post_init__(self) -> None:
        if len(self.match_id) != 16:
            raise ValueError("match_id must contain exactly 16 bytes")


@dataclass(frozen=True, slots=True)
class RobotCommand:
    """One robot's two-channel command."""

    mode: ControlMode
    first: float
    second: float


class Controller(Protocol):
    """Policy callbacks invoked by the transport runner."""

    def on_reset(self, config: dict[str, Any], initial_state: dict[str, Any]) -> None: ...

    def act(
        self, observation: dict[str, Any]
    ) -> tuple[RobotCommand, RobotCommand, RobotCommand]: ...

    def on_event(self, event: dict[str, Any]) -> None: ...

    def on_result(self, result: dict[str, Any]) -> None: ...


class StopController:
    """Safe deterministic sample controller."""

    def on_reset(self, config: dict[str, Any], initial_state: dict[str, Any]) -> None:
        del config, initial_state

    def act(self, observation: dict[str, Any]) -> tuple[RobotCommand, RobotCommand, RobotCommand]:
        del observation
        stopped = RobotCommand(ControlMode.WHEEL_VELOCITY, 0.0, 0.0)
        return stopped, stopped, stopped

    def on_event(self, event: dict[str, Any]) -> None:
        del event

    def on_result(self, result: dict[str, Any]) -> None:
        del result
