# Design

An agent decision is classified as idle spin when all conditions hold:

- normalized differential wheel intensity exceeds `0.13`;
- normalized forward drive intensity is below `0.07`;
- robot translation is below `0.08 m/s`;
- the robot is farther than `0.12 m` from the ball.

The behavior is measured immediately but is penalized only after `0.5 s` of
consecutive decisions. The penalty scales with turn intensity and is averaged
over active teammates. Its coefficient is `0.005`.

Immutable semantic evaluation uses the same definition. A checkpoint with more
than 8% idle-spin decisions is behavior-ineligible regardless of aggregate
skill success. Behavior eligibility precedes semantic score in checkpoint
ranking and resets the consecutive phase-promotion streak.

Warm initialization loads policy and critic weights from the selected healthy
checkpoint while resetting optimizer, policy version, RNG, league, and
curriculum state.
