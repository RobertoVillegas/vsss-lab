"""Versioned, reward-independent analytics derived from canonical replays."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from vsss_eval.replay import inspect_replay

Team = Literal["blue", "yellow"]
TEAMS: tuple[Team, Team] = ("blue", "yellow")
ANALYTICS_SCHEMA_VERSION = 1
DEFINITION_VERSION = "m14.1"


@dataclass(frozen=True)
class Possession:
    team: Team
    robot_id: str
    start: float
    end: float
    duration: float
    start_x: float
    end_x: float
    progress: float
    ended_by: str


@dataclass(frozen=True)
class ReplayAnalytics:
    schema_version: int
    definition_version: str
    source_replay: str
    sample_count: int
    sampled_seconds: float
    sampling_period_seconds: float
    teams: dict[str, dict[str, Any]]
    robots: dict[str, dict[str, Any]]
    possessions: tuple[Possession, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["possessions"] = [asdict(interval) for interval in self.possessions]
        return value


def analyze_replay(path: Path) -> ReplayAnalytics:
    """Derive immutable diagnostics; this function never writes into the replay."""
    inspect_replay(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header = records[0]
    ticks = records[1:]
    config = header.get("config", {})
    field = config.get("field", {})
    field_length = float(field.get("length", 1.5))
    robot = config.get("robot", {})
    robot_radius = (
        math.hypot(
            float(robot.get("length", 0.075)),
            float(robot.get("width", 0.075)),
        )
        / 2
    )
    ball_radius = float(config.get("ball", {}).get("radius", 0.0215))
    contact_distance = robot_radius + ball_radius + 0.008
    double_commit_distance = max(0.12, contact_distance * 1.75)

    team_stats: dict[str, dict[str, Any]] = {
        team: {
            "possession_seconds": 0.0,
            "pressure": {"defensive": 0.0, "neutral": 0.0, "attacking": 0.0},
            "passes": 0,
            "double_commit_seconds": 0.0,
            "congestion_seconds": 0.0,
            "goals": 0,
            "goals_conceded": 0,
            "last_defender_failures": 0,
        }
        for team in TEAMS
    }
    robot_stats: dict[str, dict[str, Any]] = {}
    possessions: list[Possession] = []
    active: dict[str, Any] | None = None
    previous_contacts: set[str] = set()
    previous_time: float | None = None
    previous_positions: dict[str, tuple[float, float]] = {}
    last_touch: tuple[Team, str, float] | None = None
    sampled_seconds = 0.0

    for record in ticks:
        snapshot = record["snapshot"]
        time = float(snapshot["simulation_time"])
        dt = 0.0 if previous_time is None else max(0.0, time - previous_time)
        previous_time = time
        sampled_seconds += dt
        ball = snapshot["ball"]
        ball_x = float(ball["x"])
        ball_y = float(ball["y"])
        robots = [item for item in snapshot["robots"] if bool(item.get("enabled", True))]

        contacts: list[tuple[float, Team, str]] = []
        by_team: dict[Team, list[dict[str, Any]]] = {"blue": [], "yellow": []}
        for item in robots:
            team = _team(item["team"])
            by_team[team].append(item)
            robot_id = str(item["id"])
            x = float(item["pose"]["x"])
            y = float(item["pose"]["y"])
            distance_ball = math.hypot(x - ball_x, y - ball_y)
            if distance_ball <= contact_distance:
                contacts.append((distance_ball, team, robot_id))
            stats = robot_stats.setdefault(
                robot_id,
                {
                    "team": team,
                    "sampled_seconds": 0.0,
                    "distance_travelled": 0.0,
                    "stationary_seconds": 0.0,
                    "distance_to_ball_integral": 0.0,
                    "distance_to_teammates_integral": 0.0,
                    "closest_to_ball_seconds": 0.0,
                    "farthest_from_ball_seconds": 0.0,
                    "behind_ball_seconds": 0.0,
                    "last_defender_seconds": 0.0,
                    "touches": 0,
                },
            )
            stats["sampled_seconds"] += dt
            stats["distance_to_ball_integral"] += distance_ball * dt
            previous = previous_positions.get(robot_id)
            travelled = math.dist(previous, (x, y)) if previous is not None else 0.0
            stats["distance_travelled"] += travelled
            if dt and travelled / dt < 0.02:
                stats["stationary_seconds"] += dt
            previous_positions[robot_id] = (x, y)
            behind = x <= ball_x if team == "blue" else x >= ball_x
            stats["behind_ball_seconds"] += dt if behind else 0.0

        for team in TEAMS:
            members = by_team[team]
            distances = [
                (math.hypot(float(r["pose"]["x"]) - ball_x, float(r["pose"]["y"]) - ball_y), r)
                for r in members
            ]
            if distances:
                closest = min(distances, key=lambda pair: pair[0])[1]
                farthest = max(distances, key=lambda pair: pair[0])[1]
                robot_stats[str(closest["id"])]["closest_to_ball_seconds"] += dt
                robot_stats[str(farthest["id"])]["farthest_from_ball_seconds"] += dt
                defender = (
                    min(members, key=lambda r: float(r["pose"]["x"]))
                    if team == "blue"
                    else max(members, key=lambda r: float(r["pose"]["x"]))
                )
                robot_stats[str(defender["id"])]["last_defender_seconds"] += dt
                if sum(distance <= double_commit_distance for distance, _ in distances) >= 2:
                    team_stats[team]["double_commit_seconds"] += dt
            for member in members:
                mates = [mate for mate in members if mate["id"] != member["id"]]
                if mates:
                    mean_distance = sum(
                        math.dist(
                            (float(member["pose"]["x"]), float(member["pose"]["y"])),
                            (float(mate["pose"]["x"]), float(mate["pose"]["y"])),
                        )
                        for mate in mates
                    ) / len(mates)
                    robot_stats[str(member["id"])]["distance_to_teammates_integral"] += (
                        mean_distance * dt
                    )
            if any(
                math.dist(
                    (float(members[i]["pose"]["x"]), float(members[i]["pose"]["y"])),
                    (float(members[j]["pose"]["x"]), float(members[j]["pose"]["y"])),
                )
                < 0.14
                for i in range(len(members))
                for j in range(i + 1, len(members))
            ):
                team_stats[team]["congestion_seconds"] += dt

            zone = _zone(ball_x if team == "blue" else -ball_x, field_length)
            team_stats[team]["pressure"][zone] += dt

        entering = [contact for contact in contacts if contact[2] not in previous_contacts]
        previous_contacts = {contact[2] for contact in contacts}
        if entering:
            _, touch_team, touch_robot = min(entering, key=lambda value: value[0])
            robot_stats[touch_robot]["touches"] += 1
            if active is not None and active["team"] != touch_team:
                possessions.append(_finish_possession(active, time, ball_x, "opponent_touch"))
                active = None
            if (
                last_touch is not None
                and last_touch[0] == touch_team
                and last_touch[1] != touch_robot
            ):
                direction = 1.0 if touch_team == "blue" else -1.0
                if direction * (ball_x - last_touch[2]) >= 0.03:
                    team_stats[touch_team]["passes"] += 1
            last_touch = (touch_team, touch_robot, ball_x)
            if active is None:
                active = {
                    "team": touch_team,
                    "robot_id": touch_robot,
                    "start": time,
                    "start_x": ball_x,
                }
            else:
                active["robot_id"] = touch_robot

        events = int(record.get("events", snapshot.get("events", 0)))
        goal_team: Team | None = "blue" if events & 1 else "yellow" if events & 2 else None
        if goal_team is not None:
            opponent: Team = "yellow" if goal_team == "blue" else "blue"
            team_stats[goal_team]["goals"] += 1
            team_stats[opponent]["goals_conceded"] += 1
            team_stats[opponent]["last_defender_failures"] += 1
            if active is not None:
                possessions.append(_finish_possession(active, time, ball_x, "goal"))
                active = None
            last_touch = None

    if active is not None and previous_time is not None:
        possessions.append(
            _finish_possession(
                active, previous_time, float(ticks[-1]["snapshot"]["ball"]["x"]), "replay_end"
            )
        )
    for possession in possessions:
        team_stats[possession.team]["possession_seconds"] += possession.duration
    for stats in robot_stats.values():
        seconds = float(stats["sampled_seconds"])
        stats["average_distance_to_ball"] = (
            stats.pop("distance_to_ball_integral") / seconds if seconds else 0.0
        )
        stats["average_distance_to_teammates"] = (
            stats.pop("distance_to_teammates_integral") / seconds if seconds else 0.0
        )
    sampling_period = sampled_seconds / (len(ticks) - 1) if len(ticks) > 1 else 0.0
    return ReplayAnalytics(
        schema_version=ANALYTICS_SCHEMA_VERSION,
        definition_version=DEFINITION_VERSION,
        source_replay=str(path.resolve()),
        sample_count=len(ticks),
        sampled_seconds=sampled_seconds,
        sampling_period_seconds=sampling_period,
        teams=team_stats,
        robots=robot_stats,
        possessions=tuple(possessions),
    )


def _finish_possession(
    active: dict[str, Any], end: float, end_x: float, ended_by: str
) -> Possession:
    team = _team(active["team"])
    direction = 1.0 if team == "blue" else -1.0
    return Possession(
        team=team,
        robot_id=str(active["robot_id"]),
        start=float(active["start"]),
        end=end,
        duration=max(0.0, end - float(active["start"])),
        start_x=float(active["start_x"]),
        end_x=end_x,
        progress=direction * (end_x - float(active["start_x"])),
        ended_by=ended_by,
    )


def _team(value: object) -> Team:
    if value not in TEAMS:
        raise ValueError(f"unsupported team: {value}")
    return value


def _zone(team_relative_ball_x: float, field_length: float) -> str:
    boundary = field_length / 6
    if team_relative_ball_x < -boundary:
        return "defensive"
    if team_relative_ball_x > boundary:
        return "attacking"
    return "neutral"
