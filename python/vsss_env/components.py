"""RLGym-like replaceable environment components."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


class ObservationBuilder(Protocol):
    """Build policy observations from a canonical flattened state."""

    def build(self, state: FloatArray, agent_index: int) -> FloatArray: ...


class ActionAdapter(Protocol):
    """Convert one policy action to two wheel commands."""

    def parse(self, action: FloatArray, state: FloatArray, agent_index: int) -> FloatArray: ...


class RewardTerm(Protocol):
    """Compute one agent reward."""

    def compute(self, previous: FloatArray, current: FloatArray, agent_index: int) -> float: ...


class TerminationCondition(Protocol):
    """Determine episode termination from the canonical state."""

    def evaluate(self, state: FloatArray) -> bool: ...


class GlobalObservation:
    """Expose a copy of the global canonical state."""

    def build(self, state: FloatArray, agent_index: int) -> FloatArray:
        del agent_index
        return state.copy()


class WheelVelocityAction:
    """Clip normalized wheel commands to the native action range."""

    def parse(self, action: FloatArray, state: FloatArray, agent_index: int) -> FloatArray:
        del state, agent_index
        return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


class ZeroReward:
    """M3 placeholder reward; M5 owns reward semantics."""

    def compute(self, previous: FloatArray, current: FloatArray, agent_index: int) -> float:
        del previous, current, agent_index
        return 0.0


class GoalTermination:
    """Terminate when either versioned goal-event bit is set."""

    def evaluate(self, state: FloatArray) -> bool:
        return bool(int(state[-1]) & 0b11)
