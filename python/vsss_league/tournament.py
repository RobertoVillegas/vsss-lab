"""Deterministic local tournament evaluation and reports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from vsss_train.marl import SharedActor

from vsss_league.ratings import elo_update
from vsss_league.replay import run_policy_replay


@dataclass(frozen=True)
class MatchRecord:
    candidate: str
    opponent: str
    side: str
    seed: int
    score_blue: int
    score_yellow: int
    progress: float
    outcome: str
    duration_ticks: int
    config_sha256: str
    infrastructure_status: str
    replay: str


@dataclass(frozen=True)
class TournamentReport:
    schema_version: int
    candidate: str
    opponent: str
    candidate_rating: float
    opponent_rating: float
    wins: int
    draws: int
    losses: int
    matches: tuple[MatchRecord, ...]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def evaluate_candidate_vs_heuristic(
    actor: SharedActor,
    config_json: str,
    state_json: str,
    *,
    candidate: str,
    seeds: tuple[int, ...],
    ticks: int,
    replay_dir: Path,
    outcome_margin: float = 0.05,
) -> TournamentReport:
    import hashlib

    records: list[MatchRecord] = []
    candidate_rating = 1_000.0
    opponent_rating = 1_000.0
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    replay_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for side in ("blue", "yellow"):
            selected_state = state_json if side == "blue" else _reflected_state(state_json)
            replay = replay_dir / f"{candidate.replace('@', '-')}-{side}-seed-{seed}.jsonl"
            result = run_policy_replay(
                actor,
                None,
                config_json,
                selected_state,
                seed=seed,
                ticks=ticks,
                replay_path=replay,
                blue_policy=candidate,
                yellow_policy="heuristic",
            )
            progress = float(result["progress"])
            first_score = (
                1.0 if progress > outcome_margin else 0.0 if progress < -outcome_margin else 0.5
            )
            outcome = "win" if first_score == 1.0 else "loss" if first_score == 0.0 else "draw"
            rating = elo_update(
                candidate_rating,
                opponent_rating,
                first_score=first_score,
            )
            candidate_rating, opponent_rating = rating.first, rating.second
            records.append(
                MatchRecord(
                    candidate=candidate,
                    opponent="heuristic",
                    side=side,
                    seed=seed,
                    score_blue=int(result["score_blue"]),
                    score_yellow=int(result["score_yellow"]),
                    progress=progress,
                    outcome=outcome,
                    duration_ticks=int(result["ticks"]),
                    config_sha256=config_hash,
                    infrastructure_status="ok",
                    replay=str(result["replay"]),
                )
            )
    return TournamentReport(
        schema_version=1,
        candidate=candidate,
        opponent="heuristic",
        candidate_rating=candidate_rating,
        opponent_rating=opponent_rating,
        wins=sum(record.outcome == "win" for record in records),
        draws=sum(record.outcome == "draw" for record in records),
        losses=sum(record.outcome == "loss" for record in records),
        matches=tuple(records),
    )


def _reflected_state(state_json: str) -> str:
    state = json.loads(state_json)
    state["score_blue"], state["score_yellow"] = state["score_yellow"], state["score_blue"]
    ball = state["ball"]
    for key in ("x", "y", "vx", "vy"):
        ball[key] = -ball[key]
    for robot in state["robots"]:
        robot["team"] = "yellow" if robot["team"] == "blue" else "blue"
        pose = robot["pose"]
        pose["x"] = -pose["x"]
        pose["y"] = -pose["y"]
        pose["theta"] = (pose["theta"] + math.pi + math.pi) % (2 * math.pi) - math.pi
        robot["twist"]["vx"] = -robot["twist"]["vx"]
        robot["twist"]["vy"] = -robot["twist"]["vy"]
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
