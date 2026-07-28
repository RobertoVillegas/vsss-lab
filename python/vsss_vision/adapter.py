"""Versioned optional predictive features for controlled policy ablations."""

from __future__ import annotations

from dataclasses import dataclass

from vsss_vision.contracts import (
    BallEstimate,
    PolicyVisionRecord,
    PredictiveFeatures,
)
from vsss_vision.prediction import (
    DEFAULT_FIELD_PREDICTION_MODEL,
    FieldPredictionModel,
    collision_aware_ball_prediction,
    goalkeeper_interception,
)

PREDICTIVE_FEATURE_SCHEMA = 1


@dataclass(frozen=True)
class PredictiveObservationAdapter:
    adapter_id: str = "m12-predictive-v1"
    horizon: float = 1.0
    offsets: tuple[float, ...] = (0.2, 0.5)
    field: FieldPredictionModel = DEFAULT_FIELD_PREDICTION_MODEL

    @property
    def feature_width(self) -> int:
        return len(self.offsets) * 4 + 4

    def build(
        self,
        estimate: BallEstimate | None,
        *,
        decision_time: float,
    ) -> PolicyVisionRecord:
        if estimate is None:
            features = PredictiveFeatures(
                PREDICTIVE_FEATURE_SCHEMA,
                self.adapter_id,
                False,
                (0.0,) * self.feature_width,
            )
            return PolicyVisionRecord(decision_time, None, None, None, features)
        prediction = collision_aware_ball_prediction(
            estimate,
            generated_time=decision_time,
            model=self.field,
            horizon=self.horizon,
        )
        if prediction.stale:
            features = PredictiveFeatures(
                PREDICTIVE_FEATURE_SCHEMA,
                self.adapter_id,
                False,
                (0.0,) * self.feature_width,
            )
            return PolicyVisionRecord(decision_time, estimate, prediction, None, features)
        values: list[float] = []
        for offset in self.offsets:
            index = min(
                range(len(prediction.samples)),
                key=lambda candidate: abs(prediction.samples[candidate][0] - offset),
            )
            _, x, y = prediction.samples[index]
            _, sigma_x, sigma_y = prediction.uncertainty[index]
            values.extend(
                (
                    x / self.field.length,
                    y / self.field.width,
                    sigma_x / self.field.length,
                    sigma_y / self.field.width,
                )
            )
        interception = goalkeeper_interception(prediction, self.field)
        values.extend(
            (
                interception.elapsed / self.horizon if interception is not None else 0.0,
                interception.x / self.field.length if interception is not None else 0.0,
                interception.y / self.field.width if interception is not None else 0.0,
                1.0 if interception is not None else 0.0,
            )
        )
        features = PredictiveFeatures(
            PREDICTIVE_FEATURE_SCHEMA,
            self.adapter_id,
            True,
            tuple(values),
        )
        return PolicyVisionRecord(
            decision_time,
            estimate,
            prediction,
            interception,
            features,
        )
