"""Compare canonical metrics, exploration, and replay clustering across two runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

CHECKPOINT = re.compile(r"iteration-(\d+)\.pt")


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    iterations: int
    environment_steps: int
    matches: int
    goals: int
    draws: int
    stagnations: int
    goal_rate: float
    rolling_return: float
    rolling_progress: float
    frames_per_second: float | None
    actor_log_std: tuple[float, ...] | None
    teammate_clustering_rate: float | None
    sampled_replays: int
    sampled_frames: int


def summarize_run(
    run_dir: Path,
    *,
    replay_samples: int = 8,
    frame_stride: int = 50,
    teammate_spacing: float = 0.14,
    frames_per_second_override: float | None = None,
) -> RunSummary:
    metrics = _read_metrics(run_dir / "metrics.jsonl")
    if not metrics:
        raise ValueError(f"run has no complete metrics: {run_dir}")
    latest = metrics[-1]
    terminations = {
        reason: sum(int(metric.get("terminations", {}).get(reason, 0)) for metric in metrics)
        for reason in ("goal", "draw", "stagnation")
    }
    matches = sum(int(metric.get("matches", 0)) for metric in metrics)
    trailing = metrics[-20:]
    clustering, sampled_replays, sampled_frames = _clustering_rate(
        run_dir / "replays",
        replay_samples=replay_samples,
        frame_stride=frame_stride,
        teammate_spacing=teammate_spacing,
    )
    measured_rate = latest.get("performance", {}).get("frames_per_second")
    return RunSummary(
        run_dir=str(run_dir.resolve()),
        iterations=int(latest["iteration"]),
        environment_steps=int(
            latest.get(
                "environment_steps",
                sum(int(metric.get("frames", 0)) for metric in metrics),
            )
        ),
        matches=matches,
        goals=terminations["goal"],
        draws=terminations["draw"],
        stagnations=terminations["stagnation"],
        goal_rate=terminations["goal"] / matches if matches else 0.0,
        rolling_return=fmean(float(metric["return_total"]) for metric in trailing),
        rolling_progress=fmean(float(metric["progress"]) for metric in trailing),
        frames_per_second=(
            frames_per_second_override
            if frames_per_second_override is not None
            else float(measured_rate)
            if measured_rate is not None
            else None
        ),
        actor_log_std=_latest_actor_log_std(run_dir / "checkpoints"),
        teammate_clustering_rate=clustering,
        sampled_replays=sampled_replays,
        sampled_frames=sampled_frames,
    )


def compare_runs(baseline: RunSummary, candidate: RunSummary) -> dict[str, Any]:
    fields = (
        "goal_rate",
        "rolling_return",
        "rolling_progress",
        "frames_per_second",
        "teammate_clustering_rate",
    )
    deltas = {}
    for field in fields:
        before = getattr(baseline, field)
        after = getattr(candidate, field)
        deltas[field] = None if before is None or after is None else float(after) - float(before)
    return {
        "schema_version": 1,
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "deltas": deltas,
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _latest_actor_log_std(checkpoint_dir: Path) -> tuple[float, ...] | None:
    checkpoints = sorted(
        (
            (int(match.group(1)), path)
            for path in checkpoint_dir.glob("iteration-*.pt")
            if (match := CHECKPOINT.fullmatch(path.name)) is not None
        ),
        key=lambda item: item[0],
    )
    if not checkpoints:
        return None
    payload = torch.load(checkpoints[-1][1], map_location="cpu", weights_only=True)
    actor = payload.get("actor")
    if not isinstance(actor, dict):
        return None
    log_std = actor.get("log_std")
    if not isinstance(log_std, torch.Tensor):
        return None
    return tuple(float(value) for value in log_std.tolist())


def _clustering_rate(
    replay_dir: Path,
    *,
    replay_samples: int,
    frame_stride: int,
    teammate_spacing: float,
) -> tuple[float | None, int, int]:
    replays = sorted(replay_dir.glob("iteration-*.jsonl"))
    selected = _evenly_sample(replays, replay_samples)
    clustered = 0
    frames = 0
    for replay in selected:
        for index, line in enumerate(replay.read_text(encoding="utf-8").splitlines()):
            if index == 0 or index % frame_stride:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            snapshot = record.get("snapshot")
            if not isinstance(snapshot, dict):
                continue
            robots = snapshot.get("robots")
            if not isinstance(robots, list):
                continue
            blue = [
                (float(robot["pose"]["x"]), float(robot["pose"]["y"]))
                for robot in robots
                if isinstance(robot, dict) and robot.get("team") == "blue"
            ]
            if len(blue) != 3:
                continue
            frames += 1
            if any(
                math.dist(blue[first], blue[second]) < teammate_spacing
                for first in range(3)
                for second in range(first + 1, 3)
            ):
                clustered += 1
    return (clustered / frames if frames else None), len(selected), frames


def _evenly_sample(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or not paths:
        return []
    if len(paths) <= count:
        return paths
    if count == 1:
        return [paths[-1]]
    indices = {round(index * (len(paths) - 1) / (count - 1)) for index in range(count)}
    return [paths[index] for index in sorted(indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-samples", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument("--baseline-frames-per-second", type=float)
    arguments = parser.parse_args()
    baseline = summarize_run(
        arguments.baseline,
        replay_samples=arguments.replay_samples,
        frame_stride=arguments.frame_stride,
        frames_per_second_override=arguments.baseline_frames_per_second,
    )
    candidate = summarize_run(
        arguments.candidate,
        replay_samples=arguments.replay_samples,
        frame_stride=arguments.frame_stride,
    )
    report = compare_runs(baseline, candidate)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    arguments.output.write_text(payload)
    print(payload, end="")
