from __future__ import annotations

import json
from pathlib import Path

import pytest
from vsss_league.comparison import _evenly_sample, compare_runs, summarize_run


def test_compare_run_metrics_and_clustering(tmp_path: Path) -> None:
    run = tmp_path / "run"
    replay_dir = run / "replays"
    replay_dir.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "iteration": iteration,
                    "frames": 10,
                    "matches": 2,
                    "return_total": float(iteration),
                    "progress": iteration / 10,
                    "terminations": {"goal": 1, "draw": 1, "stagnation": 0},
                    "performance": {"frames_per_second": 100.0},
                }
            )
            for iteration in (1, 2)
        )
        + "\n"
    )
    robots = [
        {"team": "blue", "pose": {"x": 0.0, "y": 0.0}},
        {"team": "blue", "pose": {"x": 0.1, "y": 0.0}},
        {"team": "blue", "pose": {"x": 0.5, "y": 0.0}},
    ]
    (replay_dir / "iteration-000002.jsonl").write_text(
        '{"type":"header"}\n' + json.dumps({"type": "tick", "snapshot": {"robots": robots}}) + "\n"
    )

    summary = summarize_run(run, replay_samples=1, frame_stride=1)
    report = compare_runs(summary, summary)

    assert summary.environment_steps == 20
    assert summary.matches == 4
    assert summary.goal_rate == 0.5
    assert summary.teammate_clustering_rate == 1.0
    assert report["deltas"]["goal_rate"] == pytest.approx(0.0)


def test_even_replay_sampling_keeps_boundaries() -> None:
    paths = [Path(f"{index}.jsonl") for index in range(10)]

    selected = _evenly_sample(paths, 3)

    assert selected == [paths[0], paths[4], paths[9]]
