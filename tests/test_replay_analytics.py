from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from vsss_eval import analyze_replay

ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "tests/golden/m1_match_config.json").read_text())
STATE = json.loads((ROOT / "tests/golden/m1_match_state.json").read_text())


def _tick(
    index: int, time: float, ball_x: float, contacts: dict[str, float], events: int = 0
) -> dict[str, Any]:
    snapshot = json.loads(json.dumps(STATE))
    snapshot.update(tick=index * 4, simulation_time=time, events=events)
    snapshot["ball"].update(x=ball_x, y=0.0, vx=0.0, vy=0.0)
    for robot in snapshot["robots"]:
        robot["pose"]["x"] = contacts.get(robot["id"], -0.55 if robot["team"] == "blue" else 0.55)
        robot["pose"]["y"] = (
            0.0 if robot["id"] in contacts else (int(robot["id"][1:]) % 3 - 1) * 0.25
        )
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return {
        "type": "tick",
        "index": index,
        "events": events,
        "checksum": hashlib.sha256(canonical.encode()).hexdigest(),
        "snapshot": snapshot,
    }


def _write(path: Path, ticks: list[dict[str, Any]]) -> None:
    records = [{"type": "header", "version": 1, "config": CONFIG}, *ticks]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_derives_possession_pass_pressure_and_goal_without_reward_input(tmp_path: Path) -> None:
    replay = tmp_path / "match.jsonl"
    _write(
        replay,
        [
            _tick(1, 0.1, -0.20, {"R0": -0.20}),
            _tick(2, 0.2, 0.00, {}),
            _tick(3, 0.3, 0.15, {"R1": 0.15}),
            _tick(4, 0.4, 0.40, {}, events=1),
        ],
    )
    report = analyze_replay(replay)
    assert report.schema_version == 1
    assert report.definition_version == "m14.1"
    assert report.teams["blue"]["passes"] == 1
    assert report.teams["blue"]["goals"] == 1
    assert report.teams["yellow"]["goals_conceded"] == 1
    assert report.teams["blue"]["possession_seconds"] == pytest.approx(0.3)
    assert report.possessions[0].progress == pytest.approx(0.6)
    assert report.possessions[0].ended_by == "goal"
    assert report.robots["R0"]["touches"] == 1
    assert report.robots["R1"]["touches"] == 1


def test_opponent_touch_ends_possession_and_team_view_mirrors_pressure(tmp_path: Path) -> None:
    replay = tmp_path / "switch.jsonl"
    _write(
        replay,
        [
            _tick(1, 0.1, -0.4, {"R0": -0.4}),
            _tick(2, 0.2, 0.0, {}),
            _tick(3, 0.3, 0.4, {"R3": 0.4}),
            _tick(4, 0.4, 0.4, {}),
        ],
    )
    report = analyze_replay(replay)
    assert [interval.team for interval in report.possessions] == ["blue", "yellow"]
    assert report.possessions[0].ended_by == "opponent_touch"
    assert report.teams["blue"]["pressure"]["defensive"] == pytest.approx(
        report.teams["yellow"]["pressure"]["attacking"]
    )
    assert report.teams["blue"]["pressure"]["attacking"] == pytest.approx(
        report.teams["yellow"]["pressure"]["defensive"]
    )


def test_analytics_rejects_tampered_replay(tmp_path: Path) -> None:
    replay = tmp_path / "tampered.jsonl"
    tick = _tick(1, 0.1, 0.0, {})
    tick["snapshot"]["ball"]["x"] = 9.0
    _write(replay, [tick])
    with pytest.raises(ValueError, match="checksum mismatch"):
        analyze_replay(replay)
