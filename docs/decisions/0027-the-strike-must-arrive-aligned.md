# ADR 0027: The strike must arrive aligned, not merely near

- Status: proposed
- Date: 2026-08-04

## Context

Run `vsss-m24-5-run-0002` sat in the `defense` phase for one thousand iterations, passing the
full-match gate once in forty evaluations, with every full match a draw. The reward terms and the
role machinery are healthy; the blocker is the one ADR 0021 measured and ADR 0022 left open:
**the action set cannot finish an angled chance**, and matches present that chance three times in
four.

Re-measured on the current code with `tools/probe_finishing_angle.py`, which places the striker
at a chosen angle around the ball instead of trusting the drill (the drill only ever places it on
the shooting line):

| | 0° | 30° | 60° | 90° | 120° | 150° |
| --- | --- | --- | --- | --- | --- | --- |
| strike | 1.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| navigate_goal | 1.00 | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 |
| navigate_ball | 1.00 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 |

Nothing converts from 60 degrees or more. ADR 0022 fixed the chase, not the angle; this table is
unchanged by it.

Instrumenting one 60° attempt (ball at (0.45, 0.10), striker 0.20 m away at 60° off the shooting
line) shows the mechanism, and it is not a tuning problem:

- The strike targets the point 0.10 m behind the ball along the exit direction and drives straight
  at it (`_strike_target`).
- The robot reaches the ball's contact radius (0.082 m) at tick ~50 with its heading still ~70°
  off the exit direction. The drive-through's alignment gate (0.60 rad) correctly refuses to
  engage — but the robot is already touching the ball.
- That first contact pushes the ball along the robot's heading, up the wing and away from goal:
  the ball-to-goal distance *grows* over the attempt (0.316 m → 0.372 m). The strike spends the
  rest of the episode chasing a ball it has already knocked off the line.

The acquisition point is reachable and aligned in principle: a robot standing exactly on it faces
the ball along the exit direction. The straight-line approach never lets it get there aligned,
because from an angled start the ball sits beside the approach path, inside the contact radius.

## Decision

Give the strike a **ball-clearing approach phase**. Until the robot can reach the behind-ball
acquisition point without entering the ball's contact radius, the strike targets a clearing point
offset to the side of the ball that keeps the robot outside the contact radius while it turns onto
the exit direction. Once the robot is within the existing alignment gate (0.11 m, 0.60 rad), the
drive-through engages exactly as today.

The alignment gate, the exit direction, and the drive-through geometry are unchanged. Only the
approach path changes: the robot goes around the ball instead of through it.

## Consequences

- This is a primitive change, not a reward change. The carry gradient (ADR 0021) already pays the
  team to bring the ball to a convertible position; this lets the action set actually convert it.
  The two close a gap neither could close alone.
- The alignment gate is not widened and the exit direction is not softened, so a badly placed
  chance still goes where the geometry says. Only the arrival changes. A conversion that the
  geometry forbids stays forbidden.
- The `shot` drill's `ball_angle` axis, withdrawn in cycle 4 as unreachable, becomes measurable
  again. Re-running `tools/audit_skill_difficulty.py` over `shot` is part of the work, and the
  ladder may gain a real axis.
- Measured by the finishing-angle probe before and after: the on-line conversion must stay at
  1.00, and the angled conversion must rise materially above zero. Dribbling and navigate are
  untouched by construction and re-measured to confirm.
- Rollback is a single branch in `_strike_target`: the previous straight-line approach is the
  `else` case, disabled by configuration.
