# Design

## Kickoff sampling

Replace the rectangular ball sampling in `_seeded_snapshot`
(`python/vsss_train/marl_env.py`) with uniform-in-area sampling inside the center
circle:

```text
radius = full_match_kickoff_radius * sqrt(uniform(0, 1))
angle  = uniform(0, 2 pi)
ball.x = radius * cos(angle)
ball.y = radius * sin(angle)
```

The squared-radius draw gives a uniform spatial density. `radius = 0` is the
exact center and equals the `routine-kickoff-center` skill scenario.

Robot placements stay at the current seeded starts. The attacking-half robots may
enter the circle (legal); the defending-half starts (`x = 0.31..0.52` after the
existing jitter) keep every defender outside the 20 cm circle.

## Configuration

Add a validated `full_match_kickoff_radius: float = 0.20` field to `MarlConfig`
in `python/vsss_train/config.py`, in metres. Validation:

- must be finite;
- must satisfy `0.0 <= full_match_kickoff_radius <= 0.20`.

The upper bound is the VSSS Rule 7 center-circle radius, so the knob can only
shrink the legal distribution, never widen it into illegal territory. The value
is threaded from the learner into the env factories that construct
`TeamBatch` / `Env` and is read by `_seeded_snapshot`.

## Contract choice

The kickoff distribution is a training-side concern, not a canonical simulation
contract: the simulator already restores any validated state, and Rule 7 governs
tournament placement, not the physics contract. The knob therefore lives in the
versioned training config (MARLConfig) rather than in `vsss-spec` `ResetRules`.
Changing `ResetRules` would be a spec contract change requiring golden-fixture
and contract-test churn for a value the simulator does not consume.

## Compatibility

- Existing checkpoints and replays stay readable; a changed kickoff distribution
  does not alter the replay schema.
- Behavior and returns are not comparable with runs trained on the rectangular
  distribution; the next run must start from a fresh run directory.
- `distill_dynamic_teacher`, `evaluate_against_random`, and the paired-evaluation
  harnesses receive the same legal kickoff so their seeds remain comparable.

## Rollback

Rollback is a two-line commit plus a config value: restore the rectangular
sampler and any `full_match_kickoff_radius` value is ignored by the reverted
code. No artifact or checkpoint is deleted.
