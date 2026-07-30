"""Episode-aware, reward-independent trajectory diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    schema_version: int
    source_replay: str
    episodes: int
    sampled_seconds: float
    stationary_remote_seconds: float
    mean_nearest_ball_distance: float
    mean_robot_speed: float
    mean_abs_action: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_trajectory_replay(path: Path) -> TrajectoryDiagnostics:
    """Measure low-motion valleys without crossing episode reset boundaries."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or records[0].get("type") != "header":
        raise ValueError("replay must start with a header")
    config = records[0].get("config", {})
    robot = config.get("robot", {})
    robot_radius = (
        math.hypot(
            float(robot.get("length", 0.075)),
            float(robot.get("width", 0.075)),
        )
        / 2.0
    )
    contact_distance = robot_radius + float(config.get("ball", {}).get("radius", 0.0215))
    previous_time: dict[int, float] = {}
    episodes: set[int] = set()
    sampled = 0.0
    stationary_remote = 0.0
    distance_time = 0.0
    speed_time = 0.0
    action_time = 0.0
    for record in records[1:]:
        episode = int(record.get("episode", 0))
        episodes.add(episode)
        snapshot = record["snapshot"]
        time = float(snapshot["simulation_time"])
        dt = max(0.0, time - previous_time[episode]) if episode in previous_time else 0.0
        previous_time[episode] = time
        if dt == 0.0:
            continue
        ball = snapshot["ball"]
        ball_speed = math.hypot(float(ball["vx"]), float(ball["vy"]))
        robots = [item for item in snapshot["robots"] if item.get("enabled", True)]
        distances = [
            math.hypot(
                float(item["pose"]["x"]) - float(ball["x"]),
                float(item["pose"]["y"]) - float(ball["y"]),
            )
            for item in robots
        ]
        nearest = min(distances, default=0.0)
        mean_speed = (
            sum(
                math.hypot(float(item["twist"]["vx"]), float(item["twist"]["vy"]))
                for item in robots
            )
            / len(robots)
            if robots
            else 0.0
        )
        raw_actions = record.get("actions", [])
        team_action_groups = (
            raw_actions.values() if isinstance(raw_actions, dict) else (raw_actions,)
        )
        values = [
            abs(float(value))
            for team_actions in team_action_groups
            for action in team_actions
            for value in action
        ]
        mean_action = sum(values) / len(values) if values else 0.0
        sampled += dt
        distance_time += nearest * dt
        speed_time += mean_speed * dt
        action_time += mean_action * dt
        if ball_speed <= 0.02 and nearest > contact_distance + 0.01:
            stationary_remote += dt
    denominator = max(sampled, 1e-12)
    return TrajectoryDiagnostics(
        schema_version=1,
        source_replay=str(path),
        episodes=len(episodes),
        sampled_seconds=sampled,
        stationary_remote_seconds=stationary_remote,
        mean_nearest_ball_distance=distance_time / denominator,
        mean_robot_speed=speed_time / denominator,
        mean_abs_action=action_time / denominator,
    )
