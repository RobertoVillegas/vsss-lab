from vsss_vision import (
    BallEstimate,
    FieldPredictionModel,
    analytic_ball_prediction,
    collision_aware_ball_prediction,
    segment_interception,
)


def estimate(*, x: float = 0.0, vx: float = 1.0, y: float = 0.0, vy: float = 0.0) -> BallEstimate:
    return BallEstimate(
        effective_time=1.0,
        update_time=1.02,
        source_sequence=4,
        state=(x, vx, 0.0, y, vy, 0.0),
        covariance=tuple(
            tuple(0.01 if row == column else 0.0 for column in range(6)) for row in range(6)
        ),
        measurement_accepted=True,
        rejection_reason=None,
    )


def test_analytic_prediction_is_causal_and_marks_stale_estimate() -> None:
    current = estimate()
    first = analytic_ball_prediction(current, generated_time=1.1)
    second = analytic_ball_prediction(current, generated_time=1.1)
    stale = analytic_ball_prediction(current, generated_time=1.3)

    assert first == second
    assert first.samples[-1][1] > first.samples[0][1]
    assert first.uncertainty[-1][1] > first.uncertainty[0][1]
    assert not first.stale
    assert stale.stale


def test_collision_prediction_reflects_from_field_wall() -> None:
    prediction = collision_aware_ball_prediction(
        estimate(x=0.70, vx=2.0, y=0.3),
        generated_time=1.0,
        model=FieldPredictionModel(restitution=1.0, linear_damping=0.0001),
        horizon=0.3,
        interval=0.05,
    )

    xs = [sample[1] for sample in prediction.samples]
    assert max(xs) <= 0.75 - 0.0215
    assert xs[-1] < max(xs)


def test_segment_interception_returns_time_and_point() -> None:
    prediction = analytic_ball_prediction(
        estimate(vx=1.0),
        generated_time=1.0,
        damping=0.0001,
        horizon=1.0,
        interval=0.05,
    )

    interception = segment_interception(prediction, (0.5, -0.2), (0.5, 0.2))

    assert interception is not None
    time, x, y = interception
    assert 0.49 < time < 0.51
    assert abs(x - 0.5) < 1e-9
    assert abs(y) < 1e-9


def test_collision_prediction_enters_goal_mouth_and_hits_back_wall() -> None:
    prediction = collision_aware_ball_prediction(
        estimate(x=0.70, vx=2.0),
        generated_time=1.0,
        model=FieldPredictionModel(restitution=1.0, linear_damping=0.0001),
        horizon=0.3,
        interval=0.01,
    )

    xs = [sample[1] for sample in prediction.samples]
    assert max(xs) > 0.75
    assert max(xs) <= 0.75 + 0.1 - 0.0215
    assert xs[-1] < max(xs)


def test_collision_prediction_deflects_from_chamfer() -> None:
    prediction = collision_aware_ball_prediction(
        estimate(x=0.60, vx=1.0, y=0.50, vy=1.0),
        generated_time=1.0,
        model=FieldPredictionModel(restitution=1.0, linear_damping=0.0001),
        horizon=0.4,
        interval=0.01,
    )

    reach = max(x + y for _, x, y in prediction.samples)
    assert reach < 1.31
