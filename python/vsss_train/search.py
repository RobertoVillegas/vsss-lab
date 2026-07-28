"""Persistent, bounded multi-fidelity search contracts for M14."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import optuna

Fidelity = Literal["smoke", "screen", "confirm"]
FIDELITY_SEEDS: dict[Fidelity, int] = {"smoke": 1, "screen": 3, "confirm": 5}


@dataclass(frozen=True)
class SearchParameters:
    learning_rate: float
    entropy_coefficient: float
    clip_epsilon: float
    goal_coefficient: float
    progress_coefficient: float
    congestion_coefficient: float
    defensive_coefficient: float

    def validate(self) -> None:
        bounds = {
            "learning_rate": (1e-5, 1e-3),
            "entropy_coefficient": (1e-4, 5e-2),
            "clip_epsilon": (0.05, 0.3),
            "goal_coefficient": (5.0, 30.0),
            "progress_coefficient": (0.0, 2.0),
            "congestion_coefficient": (0.0, 0.5),
            "defensive_coefficient": (0.0, 1.0),
        }
        for name, (minimum, maximum) in bounds.items():
            value = float(getattr(self, name))
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} outside bounded search space")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TrialLineage:
    study: str
    trial: int
    fidelity: Fidelity
    seeds: tuple[int, ...]
    code_commit: str
    config_sha256: str
    parent_trial: int | None
    compute_seconds: float
    status: str
    pruning_reason: str | None
    objectives: tuple[float, ...] | None


@dataclass(frozen=True)
class FidelityResult:
    terminal_score: float
    coordination_failure: float
    compute_seconds: float
    physically_valid: bool = True
    non_finite: bool = False


TrialEvaluator = Callable[[SearchParameters, Fidelity, tuple[int, ...]], FidelityResult]


def create_study(
    *,
    name: str,
    storage_path: Path,
) -> optuna.study.Study:
    """Create or resume an NSGA-II study backed by local SQLite."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.NSGAIISampler(seed=14)
    return optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{storage_path.resolve()}",
        directions=("maximize", "minimize", "minimize"),
        sampler=sampler,
        load_if_exists=True,
    )


def suggest_parameters(trial: optuna.trial.Trial) -> SearchParameters:
    parameters = SearchParameters(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        entropy_coefficient=trial.suggest_float("entropy_coefficient", 1e-4, 5e-2, log=True),
        clip_epsilon=trial.suggest_float("clip_epsilon", 0.05, 0.3),
        goal_coefficient=trial.suggest_float("goal_coefficient", 5.0, 30.0),
        progress_coefficient=trial.suggest_float("progress_coefficient", 0.0, 2.0),
        congestion_coefficient=trial.suggest_float("congestion_coefficient", 0.0, 0.5),
        defensive_coefficient=trial.suggest_float("defensive_coefficient", 0.0, 1.0),
    )
    parameters.validate()
    return parameters


def seed_set(*, trial_number: int, fidelity: Fidelity, base_seed: int = 140_000) -> tuple[int, ...]:
    count = FIDELITY_SEEDS[fidelity]
    start = base_seed + trial_number * 100
    return tuple(start + index for index in range(count))


def record_lineage(path: Path, lineage: TrialLineage) -> None:
    """Append one canonical trial record while rejecting conflicting retries."""
    existing: list[dict[str, object]] = []
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = asdict(lineage)
    identity = (lineage.study, lineage.trial, lineage.fidelity)
    for record in existing:
        record_identity = (record["study"], record["trial"], record["fidelity"])
        if record_identity == identity:
            if json.dumps(record, sort_keys=True) != json.dumps(payload, sort_keys=True):
                raise ValueError("conflicting lineage for completed fidelity")
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def current_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_multifidelity_trial(
    study: optuna.study.Study,
    evaluator: TrialEvaluator,
    *,
    lineage_path: Path,
    confirmation_floor: float = 0.45,
) -> optuna.trial.FrozenTrial:
    """Execute smoke→screen→confirm and persist every fidelity decision."""
    trial = study.ask()
    parameters = suggest_parameters(trial)
    commit = current_commit()
    parent = trial.number - 1 if trial.number else None
    final: FidelityResult | None = None
    for fidelity in ("smoke", "screen", "confirm"):
        seeds = seed_set(trial_number=trial.number, fidelity=fidelity)
        result = evaluator(parameters, fidelity, seeds)
        invalid = not result.physically_valid or result.non_finite
        below_floor = fidelity != "smoke" and result.terminal_score < confirmation_floor
        pruning_reason = (
            "physical_or_numeric_invalidity"
            if invalid
            else "terminal_score_below_fidelity_floor"
            if below_floor
            else None
        )
        record_lineage(
            lineage_path,
            TrialLineage(
                study=study.study_name,
                trial=trial.number,
                fidelity=fidelity,
                seeds=seeds,
                code_commit=commit,
                config_sha256=parameters.digest,
                parent_trial=parent,
                compute_seconds=result.compute_seconds,
                status="pruned" if pruning_reason else "complete",
                pruning_reason=pruning_reason,
                objectives=(
                    result.terminal_score,
                    result.coordination_failure,
                    result.compute_seconds,
                ),
            ),
        )
        trial.set_user_attr(f"{fidelity}_seeds", seeds)
        trial.set_user_attr(f"{fidelity}_result", asdict(result))
        if pruning_reason:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            return study.trials[trial.number]
        final = result
    if final is None:
        raise AssertionError("multi-fidelity trial produced no result")
    study.tell(
        trial,
        values=(
            final.terminal_score,
            final.coordination_failure,
            final.compute_seconds,
        ),
    )
    return study.trials[trial.number]
