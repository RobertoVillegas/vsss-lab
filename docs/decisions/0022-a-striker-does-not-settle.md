# ADR 0022: A striker does not settle

- Status: accepted
- Date: 2026-08-02

## Context

`go_to_target` tapers its forward command inside half a metre — `min(1, 2·distance)` — so a robot
comes to rest on the point it was sent to. Every primitive uses it.

The strike primitive sends the robot to a contact point ten centimetres behind the ball, and
that point moves with the ball. Instrumenting one angled attempt showed what the taper does to a
chase:

| decision | robot | ball | strike target | distance |
| --- | --- | --- | --- | --- |
| 45 | (0.33, 0.07) | (0.39, 0.09) | (0.32, 0.13) | 0.058 |
| 105 | (0.39, 0.09) | (0.60, 0.16) | (0.55, 0.24) | 0.223 |

Six centimetres from the target the taper leaves twelve per cent of the wheel command. The
striker crawls, the ball rolls away, and the gap grows from 0.058 to 0.223 over sixty decisions.
It travelled six centimetres while the ball travelled twenty-one and died ten centimetres short
of the goal.

## Decision

`go_to_target` takes a `settle` flag, defaulting to true so every existing caller is unchanged.
The strike primitive passes false while it is closing on the contact point and true once it is
driving through the ball.

A striker closing on a point that moves with the ball is not trying to stop there.

## Consequences

- The stall is gone. On the same attempt the striker now travels 0.41 m and finishes behind the
  ball at (0.54, 0.17) against the ball at (0.65, 0.17), where before it stopped at (0.39, 0.09)
  and never arrived.
- **It does not improve finishing, and this is not the fix for the angle problem.** Measured on a
  bench that places the striker at a chosen angle around the ball, scoring rate goes from

  | | 0° | 30° | 60° | 90° | 120° | 150° |
  | --- | --- | --- | --- | --- | --- | --- |
  | before | 1.00 | 0.46 | 0.00 | 0.00 | 0.00 | 0.00 |
  | after | 1.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |

  Twelve conversions against eleven out of twenty-four at 30 degrees is noise, and nothing
  converts from 60 degrees or more either way. The finding of ADR 0021's evidence stands: the
  action set cannot finish an angled chance, and that needs a different change.
- Dribbling is untouched, by construction and by measurement: the driving intents score
  identically before and after at every angle.
- The change is kept because the defect it removes is real and directly observed, not because it
  scores more. Those are different claims and only the first one is supported.
