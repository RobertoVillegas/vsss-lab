# Finishing from an angle

## Why

Run `vsss-m24-5-run-0002` spent one thousand iterations in the `defense` phase and passed the
full-match gate once in forty evaluations; every full match ended in a draw. The carry gradient
(ADR 0021) and the role machinery (ADR 0024/0026) are healthy — the blocker is the one the
`finishing-and-carry-scale` change left open and ADR 0022 measured again: **the action set cannot
finish an angled chance**, and matches present that chance three times in four.

Re-measured on current code with `tools/probe_finishing_angle.py` (which reads the scale from
the match config, `max_wheel_speed = 30.0`, rather than the legacy 12.0 constant):

| | 0° | 30° | 60° | 90° | 120° | 150° |
| --- | --- | --- | --- | --- | --- | --- |
| strike | 1.00 | 0.42 | 0.00 | 0.00 | 0.00 | 0.00 |

Instrumenting one 60° attempt locates the mechanism. The strike drives straight at the point
0.10 m behind the ball along the exit direction; from an angled start the robot reaches the
ball's contact radius with its heading still ~70° off the exit direction, the alignment gate
correctly refuses to engage, and that first misaligned contact knocks the ball up the wing — the
ball-to-goal distance *grows* over the attempt. The straight-line approach crosses the ball's
contact radius before the robot can align.

## Milestone and non-goals

This is the M24.6 evidence-driven correction, implementing the `parametric-control` requirement
"An angled chance is convertible" from `finishing-and-carry-scale`. Non-goals:

- no change to the alignment gate, the exit direction, or the drive-through geometry;
- no change to the observation, the network, the role machinery, or the carry/goal coefficients;
- no widening of what counts as a convertible chance — a chance the geometry forbids stays
  forbidden;
- dribbling is not nerfed or removed; it remains a measured alternative;
- no port to the native crate until the primitive is settled against the probe.

## What changes

- `_strike_target` gains a ball-clearing approach phase: until the robot can reach the
  behind-ball acquisition point aligned, it targets a clearing waypoint on the exit line
  beyond the acquisition point (ball − 0.26 · exit), settles there clear of the ball's contact
  radius, turns onto the exit direction, re-aims at the acquisition, and only then drives
  through as today. The waypoint run-in runs at reduced authority on both wheels so the tracked
  arc stays outside the contact radius; the re-aimed turn keeps full turning authority.
- The `shot` drill's `ball_angle` axis was re-audited with `tools/audit_skill_difficulty.py`:
  the drill's angle span is restored to 3-63° and the axis is declared again (see the success
  criteria for the measurement).
- The fresh run gets a new policy fingerprint; it is not return-comparable with earlier runs.

## Success criteria

- `tools/probe_finishing_angle.py` at the config scale: on-line (0°) conversion stays at 1.00
  (1.00 → 1.00); angled conversion (≥60°) rises materially above zero (0.00 → 0.67 at 60°,
  0.04 at 90°); navigate and dribble are unchanged;
- the difficulty audit over `shot` shows `ball_angle` as a usable axis where it was withdrawn;
  it is re-declared: the action-set race at the match scale reads 0.70 0.70 0.55 0.50 0.40
  (strike) where cycle 5 measured 0.55 0.05 0.00 0.00 0.00, the drill compiles at every level,
  and the other shot ladders survive the widened placement;
- a fresh run's full-match gate shows goals per minute above the floor and draw rate below the
  ceiling where M24.5 could not;
- lint, build, and the full test suite remain green; rollback is a single branch in
  `_strike_target`, disabled by configuration.
