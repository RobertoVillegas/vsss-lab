"""Local M7 run and tournament orchestration."""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.marl_env import distill_dynamic_teacher
from vsss_train.marl_ppo import MarlLearner

from vsss_league.progress import TrainingDashboard
from vsss_league.promotion import FixtureResult, decide_promotion
from vsss_league.registry import LeagueRegistry, PolicyEntry
from vsss_league.replay import run_policy_replay
from vsss_league.tournament import evaluate_candidate_vs_heuristic
from vsss_league.training import (
    IterationResult,
    create_rollout_session,
    train_iteration,
)

FORCE_STOP_WINDOW_SECONDS = 2.0


class TrainingInterrupt:
    """Translate signals into a graceful stop or a deliberate forced stop."""

    def __init__(self) -> None:
        self.stop_requested = False
        self._first_sigint_at: float | None = None

    def handle(self, signum: int, _frame: object) -> bool:
        now = time.monotonic()
        if signum == signal.SIGINT:
            if (
                self._first_sigint_at is not None
                and now - self._first_sigint_at <= FORCE_STOP_WINDOW_SECONDS
            ):
                raise KeyboardInterrupt
            self._first_sigint_at = now
        first_request = not self.stop_requested
        self.stop_requested = True
        return first_request


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subcommands = result.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--match-config", type=Path, required=True)
    run.add_argument("--match-state", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--iterations", type=int, default=3)
    target = run.add_mutually_exclusive_group()
    target.add_argument("--matches", type=int)
    target.add_argument("--steps", type=int)
    run.add_argument("--capture-every", type=int, default=1)
    run.add_argument("--capture-seconds", type=float, default=60.0)
    run.add_argument("--checkpoint-every", type=int, default=100)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"))
    run.add_argument("--num-envs", type=int)
    run.add_argument("--resume", action="store_true")
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
    if (
        arguments.iterations <= 0
        or (arguments.matches is not None and arguments.matches <= 0)
        or (arguments.steps is not None and arguments.steps <= 0)
        or arguments.capture_every <= 0
        or arguments.capture_seconds <= 0
        or arguments.checkpoint_every <= 0
    ):
        raise ValueError(
            "iterations, matches, steps, capture-every, capture-seconds, and checkpoint-every "
            "must be positive"
        )
    run_dir: Path = arguments.run_dir
    checkpoint_dir = run_dir / "checkpoints"
    replay_dir = run_dir / "replays"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_marl_config(arguments.config)
    config = replace(
        config,
        device=arguments.device or config.device,
        num_envs=arguments.num_envs or config.num_envs,
    )
    config_json = arguments.match_config.read_text()
    state_json = arguments.match_state.read_text()
    match_config = json.loads(config_json)
    capture_frames = round(arguments.capture_seconds / float(match_config["control_period"]))
    learner = MarlLearner(config)
    registry = LeagueRegistry.load(run_dir / "registry.json")
    if registry.entries:
        if not arguments.resume:
            raise ValueError("run directory already exists; pass --resume to continue it")
        latest = max(registry.entries, key=lambda entry: entry.training_iteration)
        if latest.checkpoint is None:
            raise ValueError("latest registry entry has no checkpoint")
        learner.load(Path(latest.checkpoint))
        start_iteration = latest.training_iteration + 1
        parent_key = latest.key
    else:
        if arguments.resume:
            raise ValueError("cannot resume a run without a league registry")
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
        start_iteration = 1
        parent_key = registry.current_main().key
    metrics_path = run_dir / "metrics.jsonl"
    requested_iterations = arguments.iterations
    if arguments.matches is not None:
        requested_iterations = math.ceil(arguments.matches / config.num_envs) * math.ceil(
            config.horizon / config.rollout_steps
        )
    elif arguments.steps is not None:
        requested_iterations = math.ceil(arguments.steps / (config.num_envs * config.rollout_steps))
    final_iteration = start_iteration + requested_iterations - 1
    interrupt = TrainingInterrupt()
    forced_stop = False
    completed = 0
    total_frames = 0
    total_matches = 0
    started_at = time.monotonic()
    rollout_session = create_rollout_session(config, config_json, state_json)
    dashboard = TrainingDashboard(
        start_iteration=start_iteration,
        total_iterations=requested_iterations,
        device=learner.device.type,
        num_envs=config.num_envs,
        target_matches=arguments.matches,
        target_steps=arguments.steps,
    )
    dashboard.start()
    if config.device == "auto" and learner.device.type == "cpu":
        dashboard.log(
            "warning: CUDA is unavailable; falling back to CPU (functional but not ideal)"
        )

    def request_stop(_signum: int, _frame: object) -> None:
        if interrupt.handle(_signum, _frame):
            dashboard.request_stop()
            if _signum == signal.SIGINT:
                dashboard.log(
                    "Press Ctrl+C again within 2 seconds to stop immediately; "
                    "the last completed checkpoint will remain available."
                )

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for iteration in range(start_iteration, final_iteration + 1):
            opponent = copy.deepcopy(learner.actor).eval()
            save_checkpoint = (
                iteration % arguments.checkpoint_every == 0 or iteration == final_iteration
            )
            checkpoint = (
                checkpoint_dir / f"iteration-{iteration:06d}.pt" if save_checkpoint else None
            )
            result = train_iteration(
                learner,
                opponent,
                config_json,
                state_json,
                iteration=iteration,
                seed=config.seed + iteration,
                opponent_id=parent_key,
                checkpoint=checkpoint,
                session=rollout_session,
            )
            reached_match_target = (
                arguments.matches is not None
                and total_matches + result.matches >= arguments.matches
            )
            reached_step_target = (
                arguments.steps is not None and total_frames + result.frames >= arguments.steps
            )
            if (reached_match_target or reached_step_target) and checkpoint is None:
                checkpoint = checkpoint_dir / f"iteration-{iteration:06d}.pt"
                learner.save(checkpoint)
                result = replace(result, checkpoint=str(checkpoint.resolve()))
            if checkpoint is not None:
                parent_key = _register_candidate(registry, config, result, checkpoint, parent_key)
            with metrics_path.open("a", encoding="utf-8") as metrics:
                metrics.write(
                    json.dumps(asdict(result), sort_keys=True, separators=(",", ":")) + "\n"
                )
            if iteration % arguments.capture_every == 0:
                run_policy_replay(
                    learner.actor,
                    opponent,
                    config_json,
                    state_json,
                    seed=config.seed + 10_000 + iteration,
                    ticks=capture_frames,
                    replay_path=replay_dir / f"iteration-{iteration:06d}.jsonl",
                    blue_policy=f"{config.policy_id}@{result.policy_version}",
                    yellow_policy=f"{config.policy_id}@{result.policy_version - 1}",
                )
            if interrupt.stop_requested and checkpoint is None:
                checkpoint = checkpoint_dir / f"iteration-{iteration:06d}.pt"
                learner.save(checkpoint)
                result = replace(result, checkpoint=str(checkpoint.resolve()))
                parent_key = _register_candidate(registry, config, result, checkpoint, parent_key)
            completed += 1
            total_frames += result.frames
            total_matches += result.matches
            elapsed = time.monotonic() - started_at
            rate = completed / elapsed
            frame_rate = total_frames / elapsed
            dashboard.update(
                result,
                completed=completed,
                iteration_rate=rate,
                frame_rate=frame_rate,
                environment_steps=total_frames,
                matches=total_matches,
                match_rate=total_matches / elapsed,
                checkpoint=checkpoint is not None,
            )
            if interrupt.stop_requested:
                break
            if reached_match_target:
                break
            if reached_step_target:
                break
    except KeyboardInterrupt:
        forced_stop = True
        dashboard.log("Forced stop; preserving the last completed checkpoint.")
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        dashboard.stop()
    actual_final = start_iteration + completed - 1
    dashboard.log(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "iterations": completed,
                "matches": total_matches,
                "environment_steps": total_frames,
                "final_iteration": actual_final,
                "latest_version": learner.policy_version,
                "viewer_replays": str(replay_dir.resolve()),
                "stopped": interrupt.stop_requested or forced_stop,
                "forced_stop": forced_stop,
            },
            sort_keys=True,
        )
    )


def _register_candidate(
    registry: LeagueRegistry,
    config: MarlConfig,
    result: IterationResult,
    checkpoint: Path,
    parent_key: str,
) -> str:
    entry = PolicyEntry.from_checkpoint(
        policy_id=config.policy_id,
        version=result.policy_version,
        category="main",
        status="candidate",
        checkpoint=checkpoint,
        algorithm=config.algorithm,
        rating=1_000.0,
        parent=parent_key,
        created_at=_timestamp(),
        training_iteration=result.iteration,
    )
    registry.register(entry)
    return entry.key


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
