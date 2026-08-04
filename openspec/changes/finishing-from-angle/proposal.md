# Finishing from an angle

## Why

Run `vsss-m24-5-run-0002` spent one thousand iterations in the `defense` phase and passed the
full-match gate once in forty evaluations; every full match ended in a draw. The carry gradient
(ADR 0021) and the role machinery (ADR 0024/0026) are healthy — the blocker is the one the
`finishing-and-carry-scale` change left open and ADR 0022 measured again: **the action set cannot
finish an angled chance**, and matches present that chance three times in four.

Re-measured on current code with `tools/probe_finishing_angle.py`:

| | 0° | 30° | 60° | 90° | 120° | 150° |
| --- | --- | --- | --- | --- | --- | --- |
| strike | 1.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |

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
  behind-ball acquisition point without entering the ball's contact radius, it targets a clearing
  point offset to the side of the ball so the robot turns onto the exit direction outside the
  contact radius, then drives through as today.
- The `shot` drill's `ball_angle` axis is re-audited with `tools/audit_skill_difficulty.py`,
  since a primitive that finishes from an angle changes what the ladder measures.
- The fresh run gets a new policy fingerprint; it is not return-comparable with earlier runs.

## Success criteria

- `tools/probe_finishing_angle.py`: on-line (0°) conversion stays at 1.00; angled conversion
  (≥60°) rises materially above zero; navigate and dribble are unchanged;
- the difficulty audit over `shot` shows `ball_angle` as a usable axis where it was withdrawn;
- a fresh run's full-match gate shows goals per minute above the floor and draw rate below the
  ceiling where M24.5 could not;
- lint, build, and the full test suite remain green; rollback is a single branch in
  `_strike_target`, disabled by configuration.
