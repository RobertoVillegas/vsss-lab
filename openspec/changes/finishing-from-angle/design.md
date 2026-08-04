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

Add a clearing waypoint between the robot and the acquisition point whenever the straight
approach would enter the ball's contact radius. The waypoint sits on the side of the ball the
robot is approaching from, offset perpendicular to the exit direction far enough that the
robot's path stays outside the contact radius while it turns onto the exit direction:

```text
approach_clear = straight_line(robot, acquisition) comes within contact_radius of the ball
if approach_clear:
    target = acquisition point
else:
    side     = sign of the robot's offset from the ball-goal line
    waypoint = ball + side · clearing_distance · perp(exit) − contact_offset · exit
    target   = waypoint
```

`clearing_distance` is larger than the contact radius so the robot swings wide, turns, and only
then is sent to the acquisition point. Once the robot is within the alignment gate the
drive-through engages unchanged, so the final push through the ball is identical to today.

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
- `strike_clearing_distance: float = 0.16` — the perpendicular offset, validated to exceed the
  contact radius.
- Rollback is configuration-only: set `strike_clearing_enabled = false`. No checkpoint or
  artifact is deleted.

## Fingerprint

The primitive change alters executed behaviour, so the fresh M24.6 run starts with a new policy
fingerprint, new optimizer, RNG, registry, and curriculum state. It is not return-comparable
with M24.5; comparison is on outcome (goals per minute, draw rate, phase progression) only.
