from vsss_vision import (
    BallEstimate,
    PredictiveObservationAdapter,
)


def estimate(*, effective_time: float = 1.0) -> BallEstimate:
    return BallEstimate(
        effective_time=effective_time,
        update_time=effective_time + 0.02,
        source_sequence=4,
        state=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        covariance=tuple(
            tuple(0.01 if row == column else 0.0 for column in range(6)) for row in range(6)
        ),
        measurement_accepted=True,
        rejection_reason=None,
    )


def test_predictive_adapter_has_fixed_versioned_width() -> None:
    adapter = PredictiveObservationAdapter()

    record = adapter.build(estimate(), decision_time=1.02)

    assert record.features.schema_version == 1
    assert record.features.adapter_id == "m12-predictive-v1"
    assert record.features.available
    assert len(record.features.values) == adapter.feature_width
    assert record.estimate is not None
    assert record.prediction is not None


def test_predictive_adapter_masks_missing_and_stale_estimates() -> None:
    adapter = PredictiveObservationAdapter()

    missing = adapter.build(None, decision_time=1.0)
    stale = adapter.build(estimate(effective_time=0.0), decision_time=1.0)

    assert not missing.features.available
    assert not stale.features.available
    assert missing.features.values == stale.features.values
    assert all(value == 0.0 for value in missing.features.values)
