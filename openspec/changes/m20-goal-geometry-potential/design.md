# Design

## Potential

The controlled team's dynamic attacker defines a state-only potential:

`Φ = .45 alignment + .25 aperture + .15 proximity + .15 field_progress`

All components and the result are bounded in `[0, 1]`. Alignment measures the
robot-to-ball ray against the ball-to-goal-center ray. Aperture projects that
ray onto the opponent goal line and is positive only inside the usable opening.
Proximity represents controllability. Field progress ensures that preserving a
valid line while moving the ball toward goal improves the state.

The shaping reward is:

`F(s, s') = coefficient * (discount * Φ(s') - Φ(s))`

This is potential-based shaping. Repeating an unchanged state yields
`coefficient * (discount - 1) * Φ(s)`, which is non-positive. Approaching a
useful pose pays only for the improvement; continuing the play must improve
aperture, control, or ball progress.

## Safety against reward loopholes

- The M20 profile disables the old per-step ball-direction reward.
- Goal and semantic terminal rewards remain much larger than shaping.
- Corners are not categorically penalized; a recoverable line can improve.
- The shared team reward uses a dynamic role assignment, preserving rotation.

## Corner containment

The 70 mm clipped corner uses a diagonal cuboid with wall thickness rather than
a triangle edge. Continuous collision detection remains enabled. Correctness
is measured against the full support extent of the rotated 75 mm square robot,
not merely its center.
