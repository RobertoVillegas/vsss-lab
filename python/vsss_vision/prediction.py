"""Causal ball trajectory projection and interception queries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vsss_vision.contracts import BallEstimate, Prediction


@dataclass(frozen=True)
class FieldPredictionModel:
    length: float = 1.5
    width: float = 1.3
    ball_radius: float = 0.0215
    linear_damping: float = 0.15
    restitution: float = 0.3
    model_id: str = "m12-field-v1"


def analytic_ball_prediction(
    estimate: BallEstimate,
    *,
    generated_time: float,
    horizon: float = 1.0,
    interval: float = 0.1,
    maximum_age: float = 0.25,
    damping: float = 0.15,
) -> Prediction:
    stale = generated_time - estimate.effective_time > maximum_age
    x, vx, ax, y, vy, ay = estimate.state
    samples = []
    count = math.floor(horizon / interval)
    for index in range(count + 1):
        elapsed = index * interval
        decay = math.exp(-damping * elapsed)
        projected_x = x + (vx * elapsed + 0.5 * ax * elapsed * elapsed) * decay
        projected_y = y + (vy * elapsed + 0.5 * ay * elapsed * elapsed) * decay
        samples.append((elapsed, projected_x, projected_y))
    return Prediction(
        source_time=estimate.effective_time,
        generated_time=generated_time,
        model_id="m12-analytic-ca-v1",
        samples=tuple(samples),
        stale=stale,
    )


def collision_aware_ball_prediction(
    estimate: BallEstimate,
    *,
    generated_time: float,
    model: FieldPredictionModel = FieldPredictionModel(),
    horizon: float = 1.0,
    interval: float = 0.1,
    integration_step: float = 0.005,
    maximum_age: float = 0.25,
) -> Prediction:
    x, vx, _ax, y, vy, _ay = estimate.state
    limit_x = model.length / 2.0 - model.ball_radius
    limit_y = model.width / 2.0 - model.ball_radius
    elapsed = 0.0
    next_sample = 0.0
    samples: list[tuple[float, float, float]] = []
    while elapsed <= horizon + integration_step / 2.0:
        if elapsed + integration_step / 2.0 >= next_sample:
            samples.append((next_sample, x, y))
            next_sample += interval
        step = min(integration_step, horizon - elapsed)
        if step <= 0.0:
            break
        decay = math.exp(-model.linear_damping * step)
        vx *= decay
        vy *= decay
        x += vx * step
        y += vy * step
        if abs(x) > limit_x:
            x = math.copysign(2.0 * limit_x - abs(x), x)
            vx = -vx * model.restitution
        if abs(y) > limit_y:
            y = math.copysign(2.0 * limit_y - abs(y), y)
            vy = -vy * model.restitution
        elapsed += step
    return Prediction(
        source_time=estimate.effective_time,
        generated_time=generated_time,
        model_id=model.model_id,
        samples=tuple(samples),
        stale=generated_time - estimate.effective_time > maximum_age,
    )


def segment_interception(
    prediction: Prediction,
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float] | None:
    """Return the first sampled path crossing of an oriented line segment."""
    sx, sy = start
    ex, ey = end
    segment_x = ex - sx
    segment_y = ey - sy
    previous = prediction.samples[0] if prediction.samples else None
    for current in prediction.samples[1:]:
        assert previous is not None
        t0, x0, y0 = previous
        t1, x1, y1 = current
        path_x = x1 - x0
        path_y = y1 - y0
        denominator = path_x * segment_y - path_y * segment_x
        if abs(denominator) > 1e-12:
            path_fraction = ((sx - x0) * segment_y - (sy - y0) * segment_x) / denominator
            segment_fraction = ((sx - x0) * path_y - (sy - y0) * path_x) / denominator
            if 0.0 <= path_fraction <= 1.0 and 0.0 <= segment_fraction <= 1.0:
                return (
                    t0 + (t1 - t0) * path_fraction,
                    x0 + path_x * path_fraction,
                    y0 + path_y * path_fraction,
                )
        previous = current
    return None
