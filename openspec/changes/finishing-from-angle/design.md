# Design

## The defect, measured

`_strike_target` returns the point `ball − 0.10 · exit` and the caller drives straight at it.
From an angled start the straight segment from the robot to that point passes inside the ball's
contact radius (0.082 m). The robot touches the ball while its heading is still far off the exit
direction; the drive-through's alignment gate (`acquisition_error ≤ 0.11` and robot→ball within
0.60 rad of `exit`) correctly refuses, but contact has already happened and the ball deflects
along the robot's heading — away from goal. Measured over one 60° attempt the ball-to-goal
distance grows from 0.316 m to 0.372 m.

## The approach phase

Give the strike a **clearing waypoint** behind the ball, on the exit line and beyond the
acquisition point:

```text
clearing_waypoint = ball − (contact_offset + clearing_distance) · exit
                    ball − (0.10 + 0.16) · exit            # 0.26 m behind the ball
```

Until the robot can reach the acquisition point aligned, `_strike_target` returns the waypoint
with `settle=True` — the waypoint is a standing point, and a full-speed passage through it
drifts into an orbit around the ball while turning (measured). The robot settles there, outside
the contact radius, and the go-to-target arc rotates its heading onto the exit direction. Once
inside the clearing region (`acquisition_error ≤ 0.20`, still aligned), the target switches back
to the acquisition point: the re-aimed turn carries the heading onto the exit direction while
still clear of contact, and only then does the existing alignment gate (0.11 m, 0.60 rad) engage
the drive-through, unchanged.

### Authority, measured

The waypoint approach runs at a reduced **approach authority** `CLEARING_APPROACH_AUTHORITY = 0.35`
applied to **both** wheel requests. Scaling both keeps the commanded path identical but lets the
yaw build against the acceleration limit, and the tracked arc pulls clear of the ball. A
forward-only cut of the arc instead passes 0.061 m from the ball — inside the 0.082 contact
radius — and was rejected by measurement.

The re-aimed turn at the acquisition is **exempt** from both the approach authority and the
acquisition scale (0.72): it is a turn in place (the settle taper already zeroes the forward),
and scaling the yaw turns the release into a crawl. The caller recognizes the two clearing
phases by exact equality of the returned target with the values `_strike_target` computes
(`static_acquisition` and `clearing_waypoint`), so the authority policy lives in one place and
the native port keeps the same identity check.

The condition is geometric and stateless: it depends only on the robot, ball, and exit
direction, so the native port is a straight translation and the Python/native equivalence test
still applies.

## What does not change

- The alignment gate (0.11 m, 0.60 rad) and the drive-through offset (0.28 m) are untouched.
- The exit direction is still the policy's chosen heading; the approach phase does not second-
  guess it.
- `navigate` and `stop` are untouched; the branch lives entirely inside the strike path.
- The reward is untouched. This is a primitive change; the carry gradient (ADR 0021) already
  pays for reaching a convertible position, and this change lets the action set convert it.

## Configuration and rollback

- `strike_clearing_enabled: bool = true` — when false, the straight-line approach runs exactly
  as before. Added to `LEGACY_NEUTRAL_CONFIG` at `false` so older checkpoints load unchanged.
- `strike_clearing_distance: float = 0.16` — extra distance behind the ball the waypoint adds to
  the acquisition offset, validated to exceed the contact radius.
- Rollback is configuration-only: set `strike_clearing_enabled = false`. No checkpoint or
  artifact is deleted.

## Fingerprint

The primitive change alters executed behaviour, so the fresh M24.6 run starts with a new policy
fingerprint, new optimizer, RNG, registry, and curriculum state. It is not return-comparable
with M24.5; comparison is on outcome (goals per minute, draw rate, phase progression) only.

## Measured outcome

At the match scale (`max_wheel_speed = 30.0` from the golden config, which the probe now reads
instead of its legacy 12.0 constant) with a 480-decision budget, `tools/probe_finishing_angle.py`
reports:

| | 0° | 30° | 60° | 90° | 120° | 150° |
| --- | --- | --- | --- | --- | --- | --- |
| strike, straight-line (clearing off) | 1.00 | 0.42 | 0.00 | 0.00 | 0.00 | 0.00 |
| strike, clearing on | 1.00 | 0.46 | 0.67 | 0.04 | 0.00 | 0.00 |

A 60° trace converts in 288 decisions with the run-in never inside the contact radius
(minimum robot–ball separation ~0.113 m); with clearing off the same trace drags the ball off
line. The 12.0 scale (a pre-config legacy stress value) still fails the 60° budget because every
phase scales with wheel speed; the acceptance scale is the match scale the trainer executes.
