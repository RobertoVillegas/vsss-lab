"""Offline hidden-truth metrics for causal camera estimates and predictions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class ErrorSummary:
    samples: int
    rmse_m: float | None
    p95_m: float | None
    maximum_m: float | None


@dataclass(frozen=True)
class VisionMetrics:
    replay: str
    ticks: int
    estimate_coverage: float
    accepted_measurement_rate: float | None
    estimate_age_p95_s: float | None
    ball_estimation: ErrorSummary
    trajectory_prediction: ErrorSummary
    goalkeeper_interception: ErrorSummary

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_replay(replay_path: Path, analysis_path: Path | None = None) -> VisionMetrics:
    """Measure only after a replay is complete; results are never policy-visible."""
    header: dict[str, Any] | None = None
    ticks: list[dict[str, Any]] = []
    for line in replay_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["type"] == "header":
            header = record
        elif record["type"] == "tick":
            ticks.append(record)
    if header is None:
        raise ValueError("replay header is missing")
    if not ticks:
        raise ValueError("replay has no ticks")

    estimate_errors: list[float] = []
    estimate_ages: list[float] = []
    accepted: list[bool] = []
    interception_errors: list[float] = []
    period = float(header["config"]["control_period"])
    by_index = {int(tick["index"]): tick for tick in ticks}
    for tick in ticks:
        perception = tick.get("perception")
        estimate = perception.get("ball_estimate") if perception else None
        if estimate is not None:
            truth = tick["snapshot"]["ball"]
            estimate_errors.append(
                math.hypot(
                    float(truth["x"]) - float(estimate["state"][0]),
                    float(truth["y"]) - float(estimate["state"][3]),
                )
            )
            estimate_ages.append(float(estimate["update_time"]) - float(estimate["effective_time"]))
            accepted.append(bool(estimate["measurement_accepted"]))
        interception = perception.get("goalkeeper_interception") if perception else None
        if interception is not None:
            target_index = int(tick["index"]) + round(float(interception["elapsed"]) / period)
            target = by_index.get(target_index)
            same_episode = target is not None and int(target.get("episode", 0)) == int(
                tick.get("episode", 0)
            )
            if same_episode:
                assert target is not None
                truth = target["snapshot"]["ball"]
                interception_errors.append(
                    math.hypot(
                        float(truth["x"]) - float(interception["x"]),
                        float(truth["y"]) - float(interception["y"]),
                    )
                )

    prediction_errors: list[float] = []
    resolved_analysis = analysis_path or replay_path.with_suffix(".analysis.jsonl")
    if resolved_analysis.is_file():
        for line in resolved_analysis.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["type"] == "prediction_error":
                prediction_errors.append(float(record["error"]))

    return VisionMetrics(
        replay=str(replay_path.resolve()),
        ticks=len(ticks),
        estimate_coverage=len(estimate_errors) / len(ticks),
        accepted_measurement_rate=fmean(accepted) if accepted else None,
        estimate_age_p95_s=_percentile(estimate_ages, 0.95),
        ball_estimation=_summarize(estimate_errors),
        trajectory_prediction=_summarize(prediction_errors),
        goalkeeper_interception=_summarize(interception_errors),
    )


def _summarize(errors: list[float]) -> ErrorSummary:
    return ErrorSummary(
        samples=len(errors),
        rmse_m=math.sqrt(fmean(error * error for error in errors)) if errors else None,
        p95_m=_percentile(errors, 0.95),
        maximum_m=max(errors) if errors else None,
    )


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]
