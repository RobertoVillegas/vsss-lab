"""Deterministic local tournament evaluation and reports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from vsss_train.ablations import EntityAttentionActor
from vsss_train.marl import SharedActor
from vsss_train.marl_env import MarlMatchEnv

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


@dataclass(frozen=True)
class CheckpointScorecard:
    checkpoint: str
    policy_version: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    mean_progress: float
    matches: int


def evaluate_checkpoint_scorecard(
    actor: SharedActor | EntityAttentionActor,
    config_json: str,
    state_json: str,
    *,
    checkpoint: Path,
    policy_version: int,
    seeds: tuple[int, ...],
    ticks: int,
) -> CheckpointScorecard:
    """Evaluate terminal matches without producing heavyweight replay files."""
    device = next(actor.parameters()).device
    outcomes: list[str] = []
    progresses: list[float] = []
    goals_for = 0
    goals_against = 0
    for seed in seeds:
        for side in ("blue", "yellow"):
            selected_state = state_json if side == "blue" else _reflected_state(state_json)
            environment = MarlMatchEnv(
                config_json,
                selected_state,
                stage=8,
                horizon=ticks,
            )
            observation = environment.reset(seed)
            environment.mark_progress_origin()
            blue_score = 0
            yellow_score = 0
            done = False
            while not done:
                with torch.inference_mode():
                    action = actor.deterministic_action(observation.to(device)).cpu().numpy()
                observation, _, done, info = environment.step(np.asarray(action, dtype=np.float32))
                events = int(info["events"])
                blue_score += int(bool(events & 1))
                yellow_score += int(bool(events & 2))
            goals_for += blue_score
            goals_against += yellow_score
            if blue_score > yellow_score:
                outcomes.append("win")
            elif blue_score < yellow_score:
                outcomes.append("loss")
            else:
                outcomes.append("draw")
            progresses.append(environment.progress_score())
    return CheckpointScorecard(
        checkpoint=str(checkpoint.resolve()),
        policy_version=policy_version,
        wins=outcomes.count("win"),
        draws=outcomes.count("draw"),
        losses=outcomes.count("loss"),
        goals_for=goals_for,
        goals_against=goals_against,
        mean_progress=float(np.mean(progresses)),
        matches=len(outcomes),
    )


def evaluate_candidate_vs_heuristic(
    actor: SharedActor | EntityAttentionActor,
    config_json: str,
    state_json: str,
    *,
    candidate: str,
    seeds: tuple[int, ...],
    ticks: int,
    replay_dir: Path,
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
            score_blue = int(result["score_blue"])
            score_yellow = int(result["score_yellow"])
            first_score = (
                1.0 if score_blue > score_yellow else 0.0 if score_blue < score_yellow else 0.5
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
                    score_blue=score_blue,
                    score_yellow=score_yellow,
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
