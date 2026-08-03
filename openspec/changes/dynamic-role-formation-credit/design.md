# Design

## Shared outcome, dynamic shaping

Goals and match outcomes remain team-level. Formation shaping uses the existing hysteretic role
assignment and the existing support and coverage targets. It is evaluated after physics and
before the next observation is emitted, so the reward and actor see the same assignment.

Only active support and coverage robots contribute. The mean keeps the potential in `[0, 1]`
for 2v1 and 3v3 rosters; 1v0 has no formation contribution. Exponential distance gives a smooth
gradient without declaring a field zone intrinsically valuable.

## Scale

The coefficient is `0.20`: its discounted-return bound is two per cent of a goal, large enough
to be visible beside the existing action and congestion terms but far below the carry bound of
`5` and goal value of `10`. On the final run-0013 replay its mean absolute contribution is
`0.00061` per decision, about one third of carry. Selection is based on role behaviour and match
outcomes, never on total return.

## Rollback

Set `role_formation_coefficient = 0.0` or revert the change. No observation or action contract is
changed.
