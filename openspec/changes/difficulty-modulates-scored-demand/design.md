# Design

## The measurement chain

At iteration 1100 of `vsss-m24-3-run-0003`, with the reward decomposition instrumented:

- clearance 0 of 10 at difficulty 0.10, 0.25, 0.40 and 0.65 alike;
- 38 of 40 clearance trials touched the ball, 2.48 touches on average, and ended in
  `timeout`, so the failure was displacement rather than approach;
- forcing the requested intensity to full recovered 7 clearances, so the capability existed
  and the policy was choosing not to use it;
- the reward term that pays for imparting ball speed contributed 4e-6 per decision against
  1.16e-2 for goals, so nothing rewarded using it;
- the ball's starting depth was `-0.62` or `-0.48` by coin flip at every difficulty, so the
  demand never fell.

The last point is the only one that explains failure at the *easiest* band, and it is the
one the curriculum was designed to control.

## Why the depth, and why that axis

`spawn_distance` already governs where the primary robot starts relative to the ball, which
is spawn geometry. The ball's depth is the same kind of quantity and the one the family is
scored on, so it interpolates across the same axis: shallow and close at the easy end, deep
and far at the hard end. The predicate keeps its fixed threshold, so a clearance continues
to mean "the ball left the defensive third".

## Compatibility

Scenario geometry changes, so every parameter and state digest changes, so the generator
revision is bumped from `m17` to `m24.3`. `SkillScenarioParameters` rejects a foreign
revision at construction, which turns a silent mix of holdouts into an error. Prior
evaluations remain valid records of the previous revision and are not comparable with new
ones.

Nothing else changes: no predicate, no phase gate, no coefficient, no checkpoint format.

## Validation

A generator test asserts the depth is monotone in difficulty, that the easy end asks for
materially less displacement, and that it still starts inside the defensive third. Then the
existing iteration-1100 checkpoint is re-evaluated: clearance moves from 0.00 to 0.47, with
10 of 10 at the easiest band and 0 of 10 at the two hardest. The ramp is a gradient rather
than a wall, and it clears the phase gate.
