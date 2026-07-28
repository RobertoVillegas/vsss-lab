from __future__ import annotations

from pathlib import Path

import pytest
import torch
from vsss_eval import inspect_replay
from vsss_league.promotion import FixtureResult, decide_promotion
from vsss_league.ratings import elo_update
from vsss_league.registry import LeagueRegistry, PolicyCategory, PolicyEntry
from vsss_league.replay import run_policy_replay
from vsss_league.tournament import TournamentReport, evaluate_candidate_vs_heuristic
from vsss_league.training import train_iteration
from vsss_train.config import MarlConfig
from vsss_train.marl import SharedActor
from vsss_train.marl_ppo import MarlLearner

ROOT = Path(__file__).parents[1]
CONFIG = (ROOT / "tests/golden/m1_match_config.json").read_text()
STATE = (ROOT / "tests/golden/m1_match_state.json").read_text()


def checkpoint_entry(
    tmp_path: Path,
    *,
    policy_id: str,
    version: int,
    category: str = "historical",
    status: str = "historical",
) -> PolicyEntry:
    checkpoint = tmp_path / f"{policy_id}-{version}.pt"
    checkpoint.write_bytes(f"{policy_id}:{version}".encode())
    return PolicyEntry.from_checkpoint(
        policy_id=policy_id,
        version=version,
        category=category,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        checkpoint=checkpoint,
        algorithm="mappo",
        rating=1_000.0,
        parent=None,
        created_at="2026-07-28T00:00:00Z",
        training_iteration=version,
    )


def test_registry_is_canonical_atomic_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = LeagueRegistry.load(path)
    entry = checkpoint_entry(tmp_path, policy_id="main", version=0, category="main", status="main")
    registry.register(entry)
    expected = path.read_bytes()
    assert LeagueRegistry.load(path).entries == (entry,)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(entry)
    assert path.read_bytes() == expected
    assert not tuple(tmp_path.glob(".registry.json.*"))


def test_matchmaking_is_seeded_and_independent_of_registration_order(tmp_path: Path) -> None:
    entries = (
        checkpoint_entry(tmp_path, policy_id="a", version=0),
        checkpoint_entry(tmp_path, policy_id="b", version=0),
        checkpoint_entry(
            tmp_path,
            policy_id="main",
            version=0,
            category="main",
            status="main",
        ),
    )
    first = LeagueRegistry(tmp_path / "first.json", entries)
    second = LeagueRegistry(tmp_path / "second.json", tuple(reversed(entries)))
    weights: dict[PolicyCategory, float] = {"main": 0.35, "historical": 0.25}
    assert (
        first.select_opponent(seed=12, weights=weights).key
        == second.select_opponent(seed=12, weights=weights).key
    )


def test_promotion_retains_previous_main_as_history(tmp_path: Path) -> None:
    main = checkpoint_entry(tmp_path, policy_id="policy", version=0, category="main", status="main")
    candidate = checkpoint_entry(
        tmp_path, policy_id="policy", version=1, category="main", status="candidate"
    )
    registry = LeagueRegistry(tmp_path / "registry.json", (main, candidate))
    registry.save()
    registry.promote(candidate.key)
    assert registry.current_main().key == candidate.key
    assert registry.get(main.key).status == "historical"
    assert registry.get(main.key).category == "historical"


def test_elo_is_zero_sum_and_draw_is_stable_for_equal_ratings() -> None:
    win = elo_update(1_000.0, 1_000.0, first_score=1.0)
    assert win.first == 1_016.0
    assert win.second == 984.0
    assert (win.first - 1_000.0) == -(win.second - 1_000.0)
    draw = elo_update(1_000.0, 1_000.0, first_score=0.5)
    assert draw.first == draw.second == 1_000.0


def fixtures(margin: float) -> tuple[FixtureResult, ...]:
    return tuple(
        FixtureResult(category, category, margin, 0.0, (1, 2))
        for category in ("main", "historical", "heuristic")
    )


def test_promotion_is_reproducible_and_blocks_regression() -> None:
    passed = decide_promotion(
        candidate="candidate@1",
        current_main="main@0",
        identity_gate=True,
        fixtures=fixtures(0.2),
        required_margin=0.1,
    )
    repeated = decide_promotion(
        candidate="candidate@1",
        current_main="main@0",
        identity_gate=True,
        fixtures=fixtures(0.2),
        required_margin=0.1,
    )
    assert passed.promoted
    assert passed.canonical_json() == repeated.canonical_json()
    regressive = (*fixtures(0.2)[:2], FixtureResult("heuristic", "heuristic", -0.1, 0.0, (1, 2)))
    assert not decide_promotion(
        candidate="candidate@1",
        current_main="main@0",
        identity_gate=True,
        fixtures=regressive,
        required_margin=0.0,
    ).promoted


def test_real_self_play_iteration_updates_version_and_checkpoint(tmp_path: Path) -> None:
    config = MarlConfig(
        device="cpu",
        num_envs=1,
        algorithm="mappo",
        hidden_size=8,
        epochs=1,
        minibatch_size=6,
        horizon=4,
        action_repeat=1,
    )
    learner = MarlLearner(config)
    opponent = SharedActor(hidden_size=8)
    checkpoint = tmp_path / "iteration-1.pt"
    result = train_iteration(
        learner,
        opponent,
        CONFIG,
        STATE,
        iteration=1,
        seed=101,
        opponent_id="history@0",
        checkpoint=checkpoint,
    )
    assert result.policy_version == learner.policy_version == 1
    assert result.frames == 4
    assert checkpoint.is_file()
    assert all(torch.isfinite(torch.tensor(value)) for value in result.losses.values())


def test_learned_policy_replay_is_viewer_compatible(tmp_path: Path) -> None:
    replay = tmp_path / "learned.jsonl"
    result = run_policy_replay(
        SharedActor(hidden_size=8),
        None,
        CONFIG,
        STATE,
        seed=17,
        ticks=4,
        replay_path=replay,
        blue_policy="candidate@1",
        yellow_policy="heuristic",
    )
    inspected = inspect_replay(replay)
    assert inspected["ticks"] == result["ticks"] == 4
    assert result["simulation_seconds"] == pytest.approx(0.08)
    assert result["outcome"] in {"win", "loss", "draw"}
    assert inspected["final_checksum"] == result["final_checksum"]
    header = replay.read_text().splitlines()[0]
    assert '"blue":"candidate@1"' in header


def test_tournament_report_is_byte_reproducible_with_side_switch(tmp_path: Path) -> None:
    actor = SharedActor(hidden_size=8)

    def run_tournament() -> TournamentReport:
        return evaluate_candidate_vs_heuristic(
            actor,
            CONFIG,
            STATE,
            candidate="candidate@1",
            seeds=(3,),
            ticks=3,
            replay_dir=tmp_path / "replays",
        )

    first = run_tournament()
    second = run_tournament()
    assert first.canonical_json() == second.canonical_json()
    assert {match.side for match in first.matches} == {"blue", "yellow"}
    assert first.wins + first.draws + first.losses == 2
