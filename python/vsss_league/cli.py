"""Local M7 run and tournament orchestration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from vsss_eval import analyze_replay
from vsss_train.config import MarlConfig, load_marl_config
from vsss_train.marl_env import distill_dynamic_teacher
from vsss_train.marl_ppo import MarlLearner, PolicyActor, load_policy_actor
from vsss_train.semantic_evaluation import evaluate_semantic_skills

from vsss_league.progress import TrainingDashboard
from vsss_league.promotion import FixtureResult, decide_promotion
from vsss_league.registry import LeagueRegistry, PolicyEntry
from vsss_league.replay import run_policy_replay
from vsss_league.telemetry import TrainingTelemetry
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
    run.add_argument("--initialize-from", type=Path)
    run.add_argument("--semantic-eval-every", type=int, default=0)
    run.add_argument("--semantic-eval-seeds", type=int, default=3)
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
        or arguments.semantic_eval_every < 0
        or arguments.semantic_eval_seeds <= 0
    ):
        raise ValueError(
            "iterations, matches, steps, capture-every, capture-seconds, and checkpoint-every "
            "must be positive (semantic-eval-every may be zero)"
        )
    if arguments.resume and arguments.initialize_from is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
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
    if arguments.semantic_eval_every and not config.semantic_curriculum:
        raise ValueError("semantic evaluation cadence requires semantic_curriculum=true")
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
        warm_start_parent: str | None = None
        if arguments.initialize_from is not None:
            source_version = learner.initialize_policy(arguments.initialize_from)
            source_hash = _sha256(arguments.initialize_from)
            warm_start_parent = f"warm-start:{source_hash[:12]}@{source_version}"
            _write_json_atomic(
                run_dir / "initialization.json",
                {
                    "schema_version": 1,
                    "mode": "policy_without_optimizer",
                    "source_checkpoint": str(arguments.initialize_from.resolve()),
                    "source_sha256": source_hash,
                    "source_policy_version": source_version,
                    "reset": ["optimizer", "policy_version", "rng", "curriculum"],
                },
            )
        else:
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
                parent=warm_start_parent,
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
    curriculum_state_path = run_dir / "semantic-curriculum.json"
    semantic_evaluations_path = run_dir / "semantic-evaluations.jsonl"
    best_semantic_path = run_dir / "best-semantic.json"
    if arguments.resume and rollout_session.semantic_curriculum is not None:
        if not curriculum_state_path.is_file():
            raise ValueError("semantic resume requires semantic-curriculum.json")
        curriculum_state = json.loads(curriculum_state_path.read_text())
        if not isinstance(curriculum_state, dict):
            raise ValueError("semantic curriculum state must be an object")
        rollout_session.semantic_curriculum.load_state_dict(curriculum_state)
    opponent_cache: dict[str, PolicyActor] = {}
    telemetry = TrainingTelemetry.create(
        run_dir,
        config,
        start_iteration=start_iteration,
    )
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
    semantic_regressions = 0
    semantic_evaluation_count = 0
    semantic_early_stop = False
    try:
        for iteration in range(start_iteration, final_iteration + 1):
            opponent, opponent_id = _select_training_opponent(
                registry,
                learner,
                config,
                iteration=iteration,
                latest_checkpoint=parent_key,
                cache=opponent_cache,
            )
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
                opponent_id=opponent_id,
                checkpoint=checkpoint,
                session=rollout_session,
            )
            if rollout_session.semantic_curriculum is not None:
                _write_json_atomic(
                    curriculum_state_path,
                    rollout_session.semantic_curriculum.state_dict(),
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
            semantic_evaluation: dict[str, object] | None = None
            if (
                checkpoint is not None
                and arguments.semantic_eval_every
                and iteration % arguments.semantic_eval_every == 0
                and rollout_session.semantic_curriculum is not None
            ):
                holdout_seeds = tuple(
                    10_007 + index * 30 for index in range(arguments.semantic_eval_seeds)
                )
                report = evaluate_semantic_skills(
                    learner.actor,
                    rollout_session.semantic_curriculum.holdouts(seeds=holdout_seeds),
                    config_json,
                    state_json,
                    device=learner.device,
                )
                semantic_evaluation_count += 1
                successes = sum(family.successes for family in report.families)
                unresolved = sum(family.unresolved for family in report.families)
                minimum_family_rate = min(family.success_rate for family in report.families)
                semantic_evaluation = {
                    "iteration": iteration,
                    "checkpoint": str(checkpoint.resolve()),
                    "successes": successes,
                    "attempts": report.attempts,
                    "success_rate": successes / report.attempts,
                    "minimum_family_success_rate": minimum_family_rate,
                    "unresolved": unresolved,
                    "families": {
                        family.family: {
                            "success_rate": family.success_rate,
                            "successes": family.successes,
                            "attempts": family.attempts,
                        }
                        for family in report.families
                    },
                }
                with semantic_evaluations_path.open("a", encoding="utf-8") as evaluations:
                    evaluations.write(
                        json.dumps(semantic_evaluation, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                previous_best = (
                    json.loads(best_semantic_path.read_text())
                    if best_semantic_path.is_file()
                    else None
                )
                candidate_score = (
                    minimum_family_rate,
                    successes / report.attempts,
                    -unresolved,
                )
                previous_score = (
                    (
                        float(previous_best["minimum_family_success_rate"]),
                        float(previous_best["success_rate"]),
                        -int(previous_best["unresolved"]),
                    )
                    if isinstance(previous_best, dict)
                    else None
                )
                if previous_score is None or candidate_score > previous_score:
                    _write_json_atomic(best_semantic_path, semantic_evaluation)
                    semantic_regressions = 0
                elif (
                    candidate_score < previous_score
                    and semantic_evaluation_count > config.semantic_regression_warmup_evaluations
                ):
                    semantic_regressions += 1
                    if (
                        config.semantic_regression_patience
                        and semantic_regressions >= config.semantic_regression_patience
                    ):
                        semantic_early_stop = True
                        dashboard.log(
                            "Semantic holdouts regressed for "
                            f"{semantic_regressions} evaluations; stopping at the "
                            "last completed checkpoint. best-semantic.json remains selected."
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
            matches_per_second = total_matches / elapsed
            actor_log_std = tuple(
                float(value) for value in learner.actor.log_std.detach().cpu().tolist()
            )
            metric_record = {
                **asdict(result),
                "environment_steps": total_frames,
                "total_matches": total_matches,
                "performance": {
                    "frames_per_second": frame_rate,
                    "matches_per_second": matches_per_second,
                    "iterations_per_second": rate,
                },
                "exploration": {"actor_log_std": actor_log_std},
                "semantic_evaluation": semantic_evaluation,
            }
            with metrics_path.open("a", encoding="utf-8") as metrics:
                metrics.write(
                    json.dumps(metric_record, sort_keys=True, separators=(",", ":")) + "\n"
                )
            telemetry.log_iteration(
                result,
                environment_steps=total_frames,
                matches=total_matches,
                frames_per_second=frame_rate,
                matches_per_second=matches_per_second,
                iterations_per_second=rate,
                actor_log_std=actor_log_std,
            )
            dashboard.update(
                result,
                completed=completed,
                iteration_rate=rate,
                frame_rate=frame_rate,
                environment_steps=total_frames,
                matches=total_matches,
                match_rate=matches_per_second,
                checkpoint=checkpoint is not None,
            )
            if iteration % arguments.capture_every == 0:
                replay_path = replay_dir / f"iteration-{iteration:06d}.jsonl"
                run_policy_replay(
                    learner.actor,
                    opponent,
                    config_json,
                    state_json,
                    seed=config.seed + 10_000 + iteration,
                    ticks=capture_frames,
                    replay_path=replay_path,
                    blue_policy=f"{config.policy_id}@{result.policy_version}",
                    yellow_policy=opponent_id,
                    semantic_context=result.curriculum,
                )
                if rollout_session.curriculum is not None:
                    for descriptor in analyze_replay(replay_path).failure_descriptors():
                        rollout_session.curriculum.ingest_failure_descriptor(
                            kind=descriptor.kind,
                            digest=descriptor.digest,
                        )
            if interrupt.stop_requested:
                break
            if semantic_early_stop:
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
        telemetry.close()
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


def _select_training_opponent(
    registry: LeagueRegistry,
    learner: MarlLearner,
    config: MarlConfig,
    *,
    iteration: int,
    latest_checkpoint: str,
    cache: dict[str, PolicyActor],
) -> tuple[PolicyActor | None, str]:
    if iteration <= config.curriculum_heuristic_iterations:
        return None, "heuristic-dynamic"
    selection = registry.select_training_opponent(
        seed=config.seed + 100_000 + iteration,
        self_play_weight=config.league_self_play_weight,
        historical_weight=config.league_historical_weight,
        heuristic_weight=config.league_heuristic_weight,
        history_window=config.league_history_window,
        exclude=frozenset({latest_checkpoint}),
    )
    if selection.kind == "heuristic":
        return None, selection.key
    if selection.kind == "self":
        return (
            copy.deepcopy(learner.actor).eval(),
            f"{config.policy_id}@{learner.policy_version}",
        )
    if selection.entry is None or selection.entry.checkpoint is None:
        raise ValueError("selected historical opponent has no checkpoint")
    opponent = cache.get(selection.entry.key)
    if opponent is None:
        opponent, loaded_version = load_policy_actor(
            Path(selection.entry.checkpoint),
            config,
            learner.device,
        )
        if loaded_version != selection.entry.version:
            raise ValueError("historical opponent version does not match registry")
        if len(cache) >= config.league_history_window:
            cache.pop(next(iter(cache)))
        cache[selection.entry.key] = opponent
    return opponent, selection.entry.key


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


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
