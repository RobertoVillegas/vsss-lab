# Evidence

## Reproduction

Replay `vsss-semantic-run-0005`, iteration 175, changes between frames 1679
and 1680 at simulation time 33.60 seconds:

- speed falls discontinuously from 0.1934932 m/s to exactly zero;
- the ball is in open field, about 0.204 m from the nearest robot;
- there is no goal, reset, collision, or score change.

The preceding motion remains below Rapier's default normalized linear sleep
threshold of 0.4 for approximately its two-second sleep timer. This identifies
automatic rigid-body sleeping—not damping or replay rendering—as the cause.

## Reference comparison

- pSim 0.2.4 uses a zero-gravity Box2D world and applies its ball drag force
  with `wake=True` on every step. Its bundled Box2D linear sleep tolerance is
  0.01 m/s, so a ball moving at 0.19 m/s does not enter sleep.
- `simulation_vsss` permits Gazebo auto-disable for a genuinely resting ball,
  but its three-dimensional ODE model does not establish Rapier's generic
  0.4-length-units/s threshold as appropriate for the VSSS ball.

The implementation therefore disables sleeping only on the ball while retaining
the calibrated linear and angular damping. Robots keep their existing sleeping
behavior.

## Verification

- `cargo test -p vsss-physics-rapier --test correctness -- --nocapture`:
  13 passed.
- Low-speed regression: a free ball starting at 0.2 m/s remains continuously
  moving and monotonically decelerating for 3 simulated seconds; final speed is
  within 0.12–0.14 m/s and displacement within 0.45–0.55 m.
- Stationary regression: an active ball remains exactly stationary for
  6 simulated seconds.
- Existing collision regressions still prove robot/robot and robot/ball
  non-overlap.
- `just build`: passed.
- `just lint`: passed.
- `just test`: Rust workspace passed, viewer 7 passed, Python 191 passed.
- CUDA integration smoke: 2 iterations, 64 vector environments, 32,768
  environment steps, 113 matches, and both checkpoints completed.
- `openspec validate fix-ball-low-speed-continuity --strict`: passed.
