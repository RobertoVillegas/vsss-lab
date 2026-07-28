"""Paired terminal-outcome evaluation for evidence-gated policy promotion."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal

Outcome = Literal[-1, 0, 1]


@dataclass(frozen=True)
class PairedMatch:
    seed: int
    scenario: str
    candidate_blue: Outcome
    candidate_yellow: Outcome

    @property
    def paired_score(self) -> float:
        """Color-balanced score in [0, 1], with a draw worth one half."""
        return (_unit(self.candidate_blue) + _unit(self.candidate_yellow)) / 2.0


@dataclass(frozen=True)
class PairedEstimate:
    matches: int
    games: int
    mean_score: float
    lower_confidence: float
    upper_confidence: float
    confidence: float
    bootstrap_samples: int
    seeds: tuple[int, ...]
    scenarios: tuple[str, ...]


def paired_estimate(
    matches: tuple[PairedMatch, ...],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
) -> PairedEstimate:
    """Return a deterministic percentile interval over paired color fixtures."""
    if len(matches) < 2:
        raise ValueError("paired evaluation requires at least two independent seeds")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    seeds = tuple(match.seed for match in matches)
    if len(seeds) != len(set(seeds)):
        raise ValueError("paired evaluation seeds must be unique")
    scores = tuple(match.paired_score for match in matches)
    digest = hashlib.sha256(
        json.dumps([asdict(match) for match in matches], sort_keys=True).encode()
    ).digest()
    generator = random.Random(int.from_bytes(digest[:8]))
    estimates = sorted(
        fmean(scores[generator.randrange(len(scores))] for _ in scores)
        for _ in range(bootstrap_samples)
    )
    alpha = (1.0 - confidence) / 2.0
    return PairedEstimate(
        matches=len(matches),
        games=2 * len(matches),
        mean_score=fmean(scores),
        lower_confidence=_percentile(estimates, alpha),
        upper_confidence=_percentile(estimates, 1.0 - alpha),
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seeds=seeds,
        scenarios=tuple(sorted({match.scenario for match in matches})),
    )


def write_evaluation_artifact(
    output: Path,
    *,
    candidate: str,
    baseline: str,
    suite: str,
    matches: tuple[PairedMatch, ...],
    estimate: PairedEstimate,
) -> None:
    """Atomically persist the complete paired evidence behind a decision."""
    payload = {
        "schema_version": 1,
        "candidate": candidate,
        "baseline": baseline,
        "suite": suite,
        "estimate": asdict(estimate),
        "matches": [asdict(match) for match in matches],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def _unit(outcome: Outcome) -> float:
    return (float(outcome) + 1.0) / 2.0


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = quantile * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
