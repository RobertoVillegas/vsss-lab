"""Validation and summary for M4 JSONL replays."""

import hashlib
import json
from pathlib import Path
from typing import Any


def inspect_replay(path: Path) -> dict[str, object]:
    """Validate tick checksums and return a deterministic summary."""
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError("empty replay")
    header = json.loads(lines[0])
    if header.get("type") != "header" or header.get("version") != 1:
        raise ValueError("unsupported replay header")
    goals = 0
    last: dict[str, Any] | None = None
    for line in lines[1:]:
        record: dict[str, Any] = json.loads(line)
        snapshot_json = json.dumps(record["snapshot"], sort_keys=True, separators=(",", ":"))
        checksum = hashlib.sha256(snapshot_json.encode()).hexdigest()
        if checksum != record["checksum"]:
            raise ValueError(f"checksum mismatch at tick {record['index']}")
        goals += int(bool(int(record["events"]) & 0b11))
        last = record
    if last is None:
        raise ValueError("replay has no ticks")
    snapshot: dict[str, Any] = last["snapshot"]
    return {
        "ticks": len(lines) - 1,
        "score_blue": snapshot["score_blue"],
        "score_yellow": snapshot["score_yellow"],
        "goals": goals,
        "final_checksum": last["checksum"],
    }
