"""Contract tests for the local replay web server."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.replay_web.server import (  # noqa: E402
    discover_checkpoints,
    discover_replays,
    latest_metric,
    resolve_replay,
)


def test_discovers_iterations_in_numeric_order(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    (replay_dir / "iteration-0010.jsonl").write_text("{}\n")
    (replay_dir / "iteration-0002.jsonl").write_text("{}\n")
    (replay_dir / "notes.jsonl").write_text("{}\n")

    found = discover_replays(tmp_path)

    assert [item.iteration for item in found] == [2, 10]
    assert [item.filename for item in found] == [
        "iteration-0002.jsonl",
        "iteration-0010.jsonl",
    ]


def test_summarizes_replay_from_final_tick(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay = replay_dir / "iteration-0007.jsonl"
    replay.write_text(
        '{"type":"header"}\n'
        '{"type":"tick","snapshot":{"score_blue":0,"score_yellow":0,'
        '"simulation_time":0.02}}\n'
        '{"type":"tick","snapshot":{"score_blue":2,"score_yellow":1,'
        '"simulation_time":60.0}}\n'
    )

    [summary] = discover_replays(tmp_path)

    assert summary.iteration == 7
    assert summary.outcome == "win"
    assert summary.goals == 3
    assert summary.simulation_seconds == 60.0


def test_ignores_incomplete_trailing_record(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay = replay_dir / "iteration-0008.jsonl"
    replay.write_text(
        '{"type":"tick","snapshot":{"score_blue":1,"score_yellow":0,'
        '"simulation_time":30.0}}\n{"type":"tick"'
    )

    [summary] = discover_replays(tmp_path)

    assert summary.outcome == "win"
    assert summary.simulation_seconds == 30.0


def test_resolves_only_discovered_replay(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    expected = replay_dir / "iteration-0001.jsonl"
    expected.write_text("{}\n")

    assert resolve_replay(tmp_path, expected.name) == expected.resolve()
    assert resolve_replay(tmp_path, "../private.jsonl") is None
    assert resolve_replay(tmp_path, "iteration-9999.jsonl") is None


def test_discovers_checkpoints_and_latest_complete_metric(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "iteration-0012.pt").write_bytes(b"model")
    (checkpoint_dir / "iteration-0000.pt").write_bytes(b"bootstrap")
    (checkpoint_dir / "notes.pt").write_bytes(b"ignore")
    (tmp_path / "metrics.jsonl").write_text(
        '{"iteration":11,"return_total":0.5}\n{"iteration":12,"return_total":0.75}\n{"iteration":'
    )

    assert [item.iteration for item in discover_checkpoints(tmp_path)] == [0, 12]
    assert latest_metric(tmp_path) == {"iteration": 12, "return_total": 0.75}
