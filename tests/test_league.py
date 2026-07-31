from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from vsss_eval import inspect_replay
from vsss_league.cli import _select_training_opponent, _semantic_candidate_score
from vsss_league.promotion import FixtureResult, decide_promotion
from vsss_league.ratings import elo_update
from vsss_league.registry import LeagueRegistry, PolicyCategory, PolicyEntry
from vsss_league.replay import run_policy_replay
from vsss_league.telemetry import TrainingTelemetry
from vsss_league.tournament import (
    TournamentReport,
    evaluate_candidate_vs_heuristic,
    evaluate_checkpoint_scorecard,
    evaluate_policy_pair_scorecard,
)
from vsss_league.training import IterationResult, create_rollout_session, train_iteration
from vsss_train.config import MarlConfig
from vsss_train.marl import ParametricPrimitiveRoleActor, PrimitiveRoleActor, SharedActor
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


def test_training_population_is_seeded_bounded_and_excludes_latest(tmp_path: Path) -> None:
    entries = tuple(
        checkpoint_entry(
            tmp_path,
            policy_id="policy",
            version=version,
            category="main",
            status="main" if version == 0 else "candidate",
        )
        for version in range(5)
    )
    registry = LeagueRegistry(tmp_path / "registry.json", entries)
    selections = [
        registry.select_training_opponent(
            seed=seed,
            self_play_weight=1.0,
            historical_weight=1.0,
            heuristic_weight=1.0,
            history_window=2,
            exclude=frozenset({"policy@4"}),
        )
        for seed in range(100)
    ]
    assert {selection.kind for selection in selections} == {"self", "historical", "heuristic"}
    historical = {
        selection.entry.key
        for selection in selections
        if selection.kind == "historical" and selection.entry is not None
    }
    assert historical <= {"policy@2", "policy@3"}
    assert "policy@4" not in historical


def test_training_population_loads_historical_actor_without_changing_rng(tmp_path: Path) -> None:
    config = MarlConfig(
        device="cpu",
        num_envs=1,
        hidden_size=8,
        epochs=1,
        minibatch_size=6,
        rollout_steps=2,
        curriculum_heuristic_iterations=0,
        league_self_play_weight=0.0,
        league_historical_weight=1.0,
        league_heuristic_weight=0.0,
    )
    learner = MarlLearner(config)
    old_checkpoint = tmp_path / "iteration-000000.pt"
    learner.save(old_checkpoint)
    registry = LeagueRegistry(tmp_path / "registry.json")
    registry.register(
        PolicyEntry.from_checkpoint(
            policy_id=config.policy_id,
            version=0,
            category="main",
            status="main",
            checkpoint=old_checkpoint,
            algorithm=config.algorithm,
            rating=1_000.0,
            parent=None,
            created_at="2026-07-28T00:00:00Z",
            training_iteration=0,
        )
    )
    before = torch.get_rng_state().clone()

    opponent, opponent_id, opponent_kind = _select_training_opponent(
        registry,
        learner,
        config,
        iteration=1,
        latest_checkpoint="not-registered@1",
        cache={},
    )

    assert opponent is not None
    assert opponent_id == f"{config.policy_id}@0"
    # A historical entry shares the run's policy id, so only the selector can name the kind.
    assert opponent_kind == "historical"
    assert torch.equal(before, torch.get_rng_state())


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


def test_checkpoint_scorecard_uses_terminal_outcomes_without_replays(tmp_path: Path) -> None:
    actor = SharedActor(8)
    checkpoint = tmp_path / "iteration-000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    scorecard = evaluate_checkpoint_scorecard(
        actor,
        CONFIG,
        STATE,
        checkpoint=checkpoint,
        policy_version=1,
        seeds=(17,),
        ticks=2,
    )
    assert scorecard.matches == 2
    assert scorecard.wins + scorecard.draws + scorecard.losses == scorecard.matches
    assert scorecard.checkpoint == str(checkpoint.resolve())
    assert tuple(tmp_path.iterdir()) == (checkpoint,)


def test_policy_pair_scorecard_scores_one_lineage_and_rejects_mixed_action_spaces() -> None:
    candidate = ParametricPrimitiveRoleActor(8)
    incumbent = ParametricPrimitiveRoleActor(8)
    scorecard = evaluate_policy_pair_scorecard(
        candidate,
        incumbent,
        CONFIG,
        STATE,
        candidate="parametric@2",
        opponent="parametric@1",
        seeds=(17,),
        ticks=2,
        action_parser="parametric_primitive",
    )
    assert scorecard.matches == 2
    assert scorecard.wins + scorecard.draws + scorecard.losses == scorecard.matches

    with pytest.raises(ValueError, match=r"actions must have shape \(3, 4\)"):
        evaluate_policy_pair_scorecard(
            candidate,
            SharedActor(8),
            CONFIG,
            STATE,
            candidate="parametric@2",
            opponent="continuous@1",
            seeds=(17,),
            ticks=2,
            action_parser="parametric_primitive",
        )


def test_tensorboard_telemetry_records_training_and_exploration(tmp_path: Path) -> None:
    telemetry = TrainingTelemetry.create(
        tmp_path,
        MarlConfig(device="cpu", num_envs=1, rollout_steps=2, minibatch_size=6),
        start_iteration=1,
    )
    telemetry.log_iteration(
        IterationResult(
            iteration=1,
            policy_version=1,
            opponent="heuristic",
            seed=8,
            frames=2,
            matches=1,
            return_total=0.5,
            progress=0.25,
            checkpoint=str(tmp_path / "checkpoint.pt"),
            losses={"policy_loss": 0.1, "entropy": 1.2},
            terminations={"goal": 1, "draw": 0, "stagnation": 0},
        ),
        environment_steps=2,
        matches=1,
        frames_per_second=100.0,
        matches_per_second=2.0,
        iterations_per_second=1.0,
        actor_log_std=(-0.5, -0.6),
    )
    telemetry.close()

    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    scalar_tags = set(events.Tags()["scalars"])
    assert "training/return" in scalar_tags
    assert "termination/goal" in scalar_tags
    assert "exploration/log_std_0" in scalar_tags
    assert events.Scalars("training/return")[0].value == pytest.approx(0.5)


def test_real_self_play_iteration_updates_version_and_checkpoint(tmp_path: Path) -> None:
    config = MarlConfig(
        device="cpu",
        num_envs=1,
        algorithm="mappo",
        hidden_size=8,
        epochs=1,
        minibatch_size=6,
        horizon=4,
        rollout_steps=4,
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
    assert result.matches == 1
    assert result.episode_kinds == {"full_match": 1, "skill": 0}
    assert sum(result.match_outcomes.values()) == 1
    assert checkpoint.is_file()
    assert all(torch.isfinite(torch.tensor(value)) for value in result.losses.values())


def test_match_persists_across_short_ppo_rollouts() -> None:
    config = MarlConfig(
        device="cpu",
        num_envs=1,
        hidden_size=8,
        epochs=1,
        minibatch_size=6,
        horizon=6,
        rollout_steps=4,
        action_repeat=1,
    )
    learner = MarlLearner(config)
    opponent = SharedActor(hidden_size=8)
    session = create_rollout_session(config, CONFIG, STATE)
    first = train_iteration(
        learner,
        opponent,
        CONFIG,
        STATE,
        iteration=1,
        seed=101,
        opponent_id="history@0",
        checkpoint=None,
        session=session,
    )
    second = train_iteration(
        learner,
        opponent,
        CONFIG,
        STATE,
        iteration=2,
        seed=102,
        opponent_id="history@0",
        checkpoint=None,
        session=session,
    )
    assert first.matches == 0
    assert second.matches == 1


def test_semantic_curriculum_runs_mirrored_worlds_and_reports_outcomes() -> None:
    config = MarlConfig(
        device="cpu",
        num_envs=12,
        hidden_size=8,
        epochs=1,
        minibatch_size=72,
        horizon=2,
        rollout_steps=2,
        action_repeat=1,
        semantic_curriculum=True,
        semantic_full_match_fraction=0.0,
    )
    learner = MarlLearner(config)
    session = create_rollout_session(config, CONFIG, STATE)
    result = train_iteration(
        learner,
        None,
        CONFIG,
        STATE,
        iteration=1,
        seed=101,
        opponent_id="heuristic",
        checkpoint=None,
        session=session,
    )
    assert set(session.environment.controlled_teams) == {0, 1}
    assert result.matches == config.num_envs
    assert result.episode_kinds == {"full_match": 0, "skill": config.num_envs}
    assert sum(result.match_outcomes.values()) == 0
    assert result.curriculum is not None
    levels = result.curriculum["levels"]
    assert isinstance(levels, dict)
    assert set(levels) == {
        "approach",
        "interception",
        "save_deflection",
        "clearance",
        "shot",
        "pass_receive",
        "rotation_recovery",
    }


def test_semantic_timeout_resets_once_instead_of_repeating_cached_outcome() -> None:
    config = MarlConfig(
        device="cpu",
        num_envs=2,
        hidden_size=8,
        epochs=1,
        minibatch_size=128,
        horizon=1_500,
        rollout_steps=256,
        action_repeat=1,
        semantic_curriculum=True,
        semantic_full_match_fraction=0.0,
    )
    learner = MarlLearner(config)
    session = create_rollout_session(config, CONFIG, STATE)

    result = train_iteration(
        learner,
        None,
        CONFIG,
        STATE,
        iteration=1,
        seed=301,
        opponent_id="heuristic",
        checkpoint=None,
        session=session,
    )

    assert result.curriculum is not None
    outcomes = result.curriculum["outcomes"]
    trials = result.curriculum["trials"]
    assert isinstance(outcomes, dict)
    assert isinstance(trials, tuple)
    assert outcomes.get("unresolved", 0) <= config.num_envs
    assert len(trials) <= config.num_envs


def test_learned_policy_replay_is_viewer_compatible(tmp_path: Path) -> None:
    replay = tmp_path / "learned.jsonl"
    result = run_policy_replay(
        SharedActor(hidden_size=8),
        None,
        CONFIG,
        STATE,
        seed=17,
        ticks=12,
        replay_path=replay,
        blue_policy="candidate@1",
        yellow_policy="heuristic",
    )
    inspected = inspect_replay(replay)
    assert inspected["ticks"] == result["ticks"] == 12
    assert result["simulation_seconds"] == pytest.approx(0.24)
    assert result["outcome"] in {"win", "loss", "draw"}
    assert inspected["final_checksum"] == result["final_checksum"]
    header = replay.read_text().splitlines()[0]
    assert '"blue":"candidate@1"' in header
    analysis = Path(result["analysis"])
    records = [json.loads(line) for line in analysis.read_text().splitlines()]
    assert records[0]["policy_visible"] is False
    assert any(record["type"] == "prediction_error" for record in records)


def test_primitive_replay_preserves_policy_intent_separately_from_wheels(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "primitive.jsonl"
    run_policy_replay(
        PrimitiveRoleActor(hidden_size=8),
        None,
        CONFIG,
        STATE,
        seed=18,
        ticks=3,
        replay_path=replay,
        blue_policy="primitive@1",
        yellow_policy="heuristic",
        action_parser="primitive",
    )
    records = [json.loads(line) for line in replay.read_text().splitlines()]
    assert records[0]["version"] == 2
    assert records[0]["action_parser"] == "primitive"
    frame = records[1]
    assert len(frame["actions"]) == 6
    assert len(frame["policy_intents"]) == 6
    intent = frame["policy_intents"][0]
    assert intent["skill"] in {"stop", "navigate", "strike"}
    assert 0.0 <= intent["confidence"] <= 1.0
    assert len(intent["top_actions"]) == 3
    assert {"phase", "target", "exit_direction", "ball_distance"} <= intent.keys()
    assert frame["policy_intents"][3:] == [None, None, None]


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


def test_semantic_checkpoint_ranking_prefers_consolidation_over_tiny_minimum_gain() -> None:
    consolidated = {
        "curriculum_phase_index": 2,
        "promotion_eligible": False,
        "promotion_gates_passed": 3,
        "success_rate": 0.52,
        "unresolved": 18,
        "minimum_family_success_rate": 0.0,
    }
    traded_off = {
        **consolidated,
        "success_rate": 0.48,
        "unresolved": 32,
        "minimum_family_success_rate": 0.04,
    }
    assert _semantic_candidate_score(consolidated) > _semantic_candidate_score(traded_off)


def test_semantic_checkpoint_ranking_rejects_behavior_collapse() -> None:
    healthy = {
        "curriculum_phase_index": 2,
        "behavior_gate_passed": True,
        "promotion_eligible": False,
        "promotion_gates_passed": 2,
        "success_rate": 0.40,
        "unresolved": 60,
        "minimum_family_success_rate": 0.0,
    }
    spinning = {
        **healthy,
        "behavior_gate_passed": False,
        "promotion_gates_passed": 4,
        "success_rate": 0.55,
        "unresolved": 20,
    }

    assert _semantic_candidate_score(healthy) > _semantic_candidate_score(spinning)


def test_reward_terms_sum_to_the_reported_return_per_decision() -> None:
    """Only the total was ever recorded, so no one could say which term a policy chased."""
    config = MarlConfig(
        device="cpu",
        num_envs=2,
        rollout_steps=4,
        minibatch_size=6,
        epochs=1,
        hidden_size=16,
        stagnation_penalty=1.0,
        draw_penalty=0.5,
    )
    learner = MarlLearner(config)
    session = create_rollout_session(config, CONFIG, STATE)

    result = train_iteration(
        learner,
        None,
        CONFIG,
        STATE,
        iteration=1,
        seed=7,
        opponent_id="heuristic-dynamic",
        checkpoint=None,
        session=session,
    )

    assert result.reward_terms
    assert {"goal_scored", "goal_conceded", "stagnation", "idle_spin"} <= set(result.reward_terms)
    decisions = config.rollout_steps * config.num_envs
    total = sum(result.reward_terms.values()) * decisions
    # The trajectory reward is per agent; the accounting is per world decision.
    assert total == pytest.approx(result.return_total * decisions / 3.0, rel=0.05, abs=0.05)
    # Accounting is scoped to one rollout, so a second iteration cannot double count.
    assert session.environment.reward_decisions == 0
