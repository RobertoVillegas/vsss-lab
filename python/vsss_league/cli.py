"""Local M7 run and tournament orchestration."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from vsss_train.config import load_marl_config
from vsss_train.marl_env import distill_dynamic_teacher
from vsss_train.marl_ppo import MarlLearner

from vsss_league.promotion import FixtureResult, decide_promotion
from vsss_league.registry import LeagueRegistry, PolicyEntry
from vsss_league.replay import run_policy_replay
from vsss_league.tournament import evaluate_candidate_vs_heuristic
from vsss_league.training import train_iteration


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subcommands = result.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--match-config", type=Path, required=True)
    run.add_argument("--match-state", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--iterations", type=int, default=3)
    run.add_argument("--capture-every", type=int, default=1)
    run.add_argument("--bootstrap-samples", type=int, default=2_048)
    run.add_argument("--bootstrap-epochs", type=int, default=20)
    tournament = subcommands.add_parser("tournament")
    tournament.add_argument("--config", type=Path, required=True)
    tournament.add_argument("--match-config", type=Path, required=True)
    tournament.add_argument("--match-state", type=Path, required=True)
    tournament.add_argument("--checkpoint", type=Path, required=True)
    tournament.add_argument("--output-dir", type=Path, required=True)
    tournament.add_argument("--seeds", type=int, default=5)
    inspect = subcommands.add_parser("inspect")
    inspect.add_argument("--run-dir", type=Path, required=True)
    promote = subcommands.add_parser("promote")
    promote.add_argument("--run-dir", type=Path, required=True)
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--manifest", type=Path, required=True)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "run":
        _run(arguments)
    elif arguments.command == "tournament":
        _tournament(arguments)
    elif arguments.command == "promote":
        _promote(arguments)
    else:
        registry = LeagueRegistry.load(arguments.run_dir / "registry.json")
        print(
            json.dumps(
                {
                    "policies": [asdict(entry) for entry in registry.entries],
                    "current_main": registry.current_main().key,
                },
                sort_keys=True,
            )
        )


def _run(arguments: argparse.Namespace) -> None:
    if arguments.iterations <= 0 or arguments.capture_every <= 0:
        raise ValueError("iterations and capture-every must be positive")
    run_dir: Path = arguments.run_dir
    checkpoint_dir = run_dir / "checkpoints"
    replay_dir = run_dir / "replays"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_marl_config(arguments.config)
    config_json = arguments.match_config.read_text()
    state_json = arguments.match_state.read_text()
    learner = MarlLearner(config)
    distill_dynamic_teacher(
        learner.actor,
        config_json,
        state_json,
        seed=config.seed,
        samples=arguments.bootstrap_samples,
        epochs=arguments.bootstrap_epochs,
    )
    initial_checkpoint = checkpoint_dir / "iteration-0000.pt"
    learner.save(initial_checkpoint)
    registry = LeagueRegistry.load(run_dir / "registry.json")
    if registry.entries:
        raise ValueError("run directory already contains a league registry")
    registry.register(
        PolicyEntry.from_checkpoint(
            policy_id=config.policy_id,
            version=0,
            category="main",
            status="main",
            checkpoint=initial_checkpoint,
            algorithm=config.algorithm,
            rating=1_000.0,
            parent=None,
            created_at=_timestamp(),
            training_iteration=0,
        )
    )
    metrics_path = run_dir / "metrics.jsonl"
    for iteration in range(1, arguments.iterations + 1):
        opponent = copy.deepcopy(learner.actor).eval()
        checkpoint = checkpoint_dir / f"iteration-{iteration:04d}.pt"
        result = train_iteration(
            learner,
            opponent,
            config_json,
            state_json,
            iteration=iteration,
            seed=config.seed + iteration,
            opponent_id=registry.current_main().key,
            checkpoint=checkpoint,
        )
        registry.register(
            PolicyEntry.from_checkpoint(
                policy_id=config.policy_id,
                version=result.policy_version,
                category="main",
                status="candidate",
                checkpoint=checkpoint,
                algorithm=config.algorithm,
                rating=1_000.0,
                parent=registry.current_main().key,
                created_at=_timestamp(),
                training_iteration=iteration,
            )
        )
        with metrics_path.open("a", encoding="utf-8") as metrics:
            metrics.write(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")) + "\n")
        if iteration % arguments.capture_every == 0:
            run_policy_replay(
                learner.actor,
                opponent,
                config_json,
                state_json,
                seed=config.seed + 10_000 + iteration,
                ticks=config.horizon,
                replay_path=replay_dir / f"iteration-{iteration:04d}.jsonl",
                blue_policy=f"{config.policy_id}@{result.policy_version}",
                yellow_policy=f"{config.policy_id}@{result.policy_version - 1}",
            )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "iterations": arguments.iterations,
                "latest_version": learner.policy_version,
                "viewer_replays": str(replay_dir.resolve()),
            },
            sort_keys=True,
        )
    )


def _tournament(arguments: argparse.Namespace) -> None:
    config = load_marl_config(arguments.config)
    learner = MarlLearner(config)
    learner.load(arguments.checkpoint)
    report = evaluate_candidate_vs_heuristic(
        learner.actor,
        arguments.match_config.read_text(),
        arguments.match_state.read_text(),
        candidate=f"{config.policy_id}@{learner.policy_version}",
        seeds=tuple(range(config.seed + 40_000, config.seed + 40_000 + arguments.seeds)),
        ticks=config.horizon,
        replay_dir=arguments.output_dir / "replays",
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "tournament.json"
    report_path.write_text(report.canonical_json() + "\n")
    print(report.canonical_json())


def _promote(arguments: argparse.Namespace) -> None:
    registry = LeagueRegistry.load(arguments.run_dir / "registry.json")
    manifest = json.loads(arguments.manifest.read_text())
    decision = decide_promotion(
        candidate=arguments.candidate,
        current_main=registry.current_main().key,
        identity_gate=bool(manifest["identity_gate"]),
        fixtures=tuple(FixtureResult(**fixture) for fixture in manifest["fixtures"]),
        required_margin=float(manifest["required_margin"]),
    )
    decision_path = arguments.run_dir / "promotion.json"
    decision_path.write_text(decision.canonical_json() + "\n")
    if decision.promoted:
        registry.promote(arguments.candidate)
    print(decision.canonical_json())
    raise SystemExit(0 if decision.promoted else 1)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
