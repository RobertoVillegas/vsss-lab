"""Conformance sidecar for the canonical backend bridge."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from vsss_env._native import BatchSimulator


def main() -> None:
    simulator: BatchSimulator | None = None
    state: Any = None
    for line in sys.stdin:
        request: dict[str, Any] = json.loads(line)
        sequence = request["sequence"]
        operation = request["op"]
        if operation == "configure":
            simulator = BatchSimulator(
                json.dumps(request["config"]),
                json.dumps(request["state"]),
                1,
            )
            state = simulator.reset()[0]
        elif operation == "reset" and simulator is not None:
            state = simulator.reset()[0]
        elif operation == "step" and simulator is not None:
            actions = np.asarray(request["actions"], dtype=np.float32)
            state = simulator.step(actions[None])[0]
        elif operation == "close" and simulator is not None and state is not None:
            print(
                json.dumps({"ok": True, "sequence": sequence, "state": state.tolist()}),
                flush=True,
            )
            return
        else:
            raise RuntimeError(f"invalid operation: {operation}")
        print(json.dumps({"ok": True, "sequence": sequence, "state": state.tolist()}), flush=True)


if __name__ == "__main__":
    main()
