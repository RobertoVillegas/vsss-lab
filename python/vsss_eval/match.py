"""Deterministic 3v3 scripted match runner."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from vsss_baselines import DynamicTeamController
from vsss_env._native import BatchSimulator

from vsss_eval.visual import FrameSink, VisualFrame

REPLAY_VERSION = 1


@dataclass(frozen=True)
class MatchSummary:
    """Stable result of a scripted match."""

    ticks: int
    score_blue: int
    score_yellow: int
    goals: int
    final_checksum: str


def run_scripted_match(
    config_json: str,
    state_json: str,
    ticks: int,
    replay_path: Path,
    seed: int = 0,
    observers: Iterable[FrameSink] = (),
) -> MatchSummary:
    """Run and record a deterministic 3v3 match."""
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    observer_sinks = tuple(observers)
    config = json.loads(config_json)
    timestep = float(config["timestep"])
    simulator = BatchSimulator(config_json, state_json, 1)
    state = simulator.reset()[0]
    blue = DynamicTeamController(0, 1)
    yellow = DynamicTeamController(3, -1)
    goals = 0
    final_checksum = ""
    with replay_path.open("w", encoding="utf-8", newline="\n") as replay:
        _write(
            replay,
            {
                "type": "header",
                "version": REPLAY_VERSION,
                "seed": seed,
                "ticks": ticks,
                "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
                "config": config,
            },
        )
        for index in range(ticks):
            actions = np.zeros((1, 6, 2), dtype=np.float32)
            actions[0, :3] = blue.actions(state)
            actions[0, 3:] = yellow.actions(state)
            state = simulator.step(actions)[0]
            snapshot_json = simulator.snapshots()[0]
            snapshot = json.loads(snapshot_json)
            canonical_snapshot = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            final_checksum = hashlib.sha256(canonical_snapshot.encode()).hexdigest()
            event_flags = int(state[-1])
            goals += int(bool(event_flags & 0b11))
            record = {
                "type": "tick",
                "index": index + 1,
                "actions": actions[0].tolist(),
                "events": event_flags,
                "checksum": final_checksum,
                "snapshot": snapshot,
            }
            _write(replay, record)
            frame = VisualFrame.from_replay_record(
                record,
                timestep=timestep,
            )
            for observer in observer_sinks:
                observer.publish(frame)
    return MatchSummary(
        ticks=ticks,
        score_blue=int(state[3]),
        score_yellow=int(state[4]),
        goals=goals,
        final_checksum=final_checksum,
    )


def _write(stream: TextIO, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def summary_json(summary: MatchSummary) -> str:
    """Serialize a match summary deterministically."""
    return json.dumps(asdict(summary), sort_keys=True, separators=(",", ":"))
