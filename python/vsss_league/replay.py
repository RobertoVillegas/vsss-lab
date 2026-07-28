"""Learned-policy evaluation replay compatible with the M4/M6 viewer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from vsss_train.marl import SharedActor, build_team_observation
from vsss_train.marl_env import MarlMatchEnv


def run_policy_replay(
    blue: SharedActor,
    yellow: SharedActor | None,
    config_json: str,
    state_json: str,
    *,
    seed: int,
    ticks: int,
    replay_path: Path,
    blue_policy: str,
    yellow_policy: str,
) -> dict[str, Any]:
    """Evaluate learned blue versus learned or heuristic yellow and write JSONL."""
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    environment = MarlMatchEnv(config_json, state_json, stage=8, horizon=ticks)
    observation = environment.reset(seed)
    config = json.loads(config_json)
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    final_checksum = ""
    with replay_path.open("w", encoding="utf-8", newline="\n") as replay:
        _write(
            replay,
            {
                "type": "header",
                "version": 1,
                "seed": seed,
                "ticks": ticks,
                "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
                "config": config,
                "policies": {"blue": blue_policy, "yellow": yellow_policy},
            },
        )
        done = False
        index = 0
        while not done:
            with torch.inference_mode():
                blue_action = blue.deterministic_action(observation).numpy()
                yellow_action = (
                    yellow.deterministic_action(
                        build_team_observation(environment.state, team=1)
                    ).numpy()
                    if yellow is not None
                    else None
                )
            observation, reward, done, info = environment.step(blue_action, yellow_action)
            snapshot = environment.snapshot()
            canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
            final_checksum = hashlib.sha256(canonical.encode()).hexdigest()
            index += 1
            actions = np.asarray(info["actions"], dtype=np.float32)
            _write(
                replay,
                {
                    "type": "tick",
                    "index": index,
                    "actions": actions.tolist(),
                    "events": int(info["events"]),
                    "checksum": final_checksum,
                    "snapshot": snapshot,
                    "rewards": [reward.total] * 3 + [-reward.total] * 3,
                },
            )
    return {
        "ticks": index,
        "score_blue": int(environment.state[3]),
        "score_yellow": int(environment.state[4]),
        "progress": environment.progress_score(),
        "final_checksum": final_checksum,
        "replay": str(replay_path.resolve()),
    }


def _write(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
