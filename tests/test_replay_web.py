"""Contract tests for the local replay web server."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.replay_web.server import discover_replays, resolve_replay  # noqa: E402


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


def test_resolves_only_discovered_replay(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    expected = replay_dir / "iteration-0001.jsonl"
    expected.write_text("{}\n")

    assert resolve_replay(tmp_path, expected.name) == expected.resolve()
    assert resolve_replay(tmp_path, "../private.jsonl") is None
    assert resolve_replay(tmp_path, "iteration-9999.jsonl") is None
