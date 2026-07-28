"""Backend-neutral native and subprocess adapters."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from vsss_env._native import BatchSimulator

FloatArray = NDArray[np.float32]
STATE_WIDTH = BatchSimulator.state_width()


class CanonicalBackend(Protocol):
    """Minimal backend contract consumed by policies and replay capture."""

    def reset(self) -> FloatArray: ...

    def step(self, actions: FloatArray) -> FloatArray: ...

    def close(self) -> None: ...


def _actions(value: FloatArray) -> FloatArray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    if result.shape != (6, 2) or not np.isfinite(result).all():
        raise ValueError("actions must be finite with shape (6, 2)")
    return result


def _state(value: Sequence[float]) -> FloatArray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (STATE_WIDTH,) or not np.isfinite(result).all():
        raise RuntimeError(f"sidecar state must be {STATE_WIDTH} finite values")
    return result


class NativeBackend:
    """Canonical adapter for the in-process Rapier simulator."""

    def __init__(self, config_json: str, state_json: str) -> None:
        self._simulator = BatchSimulator(config_json, state_json, 1)

    def reset(self) -> FloatArray:
        return np.asarray(self._simulator.reset()[0], dtype=np.float32)

    def step(self, actions: FloatArray) -> FloatArray:
        return np.asarray(self._simulator.step(_actions(actions)[None])[0], dtype=np.float32)

    def close(self) -> None:
        pass


class JsonLineBackend:
    """Canonical process bridge used by ROS/Gazebo sidecars."""

    def __init__(self, command: Sequence[str], config_json: str, state_json: str) -> None:
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._sequence = 0
        self._request(
            {
                "op": "configure",
                "config": json.loads(config_json),
                "state": json.loads(state_json),
            }
        )

    def _request(self, body: dict[str, object]) -> FloatArray:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("sidecar pipes unavailable")
        self._sequence += 1
        body["sequence"] = self._sequence
        self._process.stdin.write(json.dumps(body, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = "" if self._process.stderr is None else self._process.stderr.read()
            raise RuntimeError(f"sidecar exited before response: {stderr.strip()}")
        response = json.loads(line)
        if response.get("sequence") != self._sequence or response.get("ok") is not True:
            raise RuntimeError(f"invalid sidecar response: {response}")
        return _state(response["state"])

    def reset(self) -> FloatArray:
        return self._request({"op": "reset"})

    def step(self, actions: FloatArray) -> FloatArray:
        return self._request({"op": "step", "actions": _actions(actions).tolist()})

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._request({"op": "close"})
            except (BrokenPipeError, RuntimeError):
                self._process.terminate()
        self._process.wait(timeout=5)

    def __enter__(self) -> JsonLineBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def run_policy(backend: CanonicalBackend, ticks: int) -> list[FloatArray]:
    """Run one backend-agnostic constant-wheel policy and return canonical states."""
    states = [backend.reset()]
    actions = np.zeros((6, 2), dtype=np.float32)
    actions[0] = 10.0
    for _ in range(ticks):
        state = backend.step(actions)
        if not math.isfinite(float(state[2])):
            raise RuntimeError("backend returned invalid simulation time")
        states.append(state)
    return states
