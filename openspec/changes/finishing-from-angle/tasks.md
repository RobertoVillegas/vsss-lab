 # Tasks

 - [x] Re-measure the finishing-angle probe on current code: strike converts 1.00 on the line,
      0.50 at 30°, 0.00 from 60° and beyond; the run-0002 full-match gate stall is recorded.
- [x] Instrument one 60° attempt and locate the mechanism: the straight-line approach to the
      behind-ball acquisition point enters the ball's contact radius while the robot's heading is
      ~70° off the exit direction, and that first misaligned contact knocks the ball up the wing
      (ball-to-goal distance grows over the attempt).
- [x] Accept ADR 0027 (the strike must arrive aligned, not merely near). Status accepted.
- [x] Implement the ball-clearing approach phase in `_strike_target`, behind
      `strike_clearing_enabled` / `strike_clearing_distance`, Python and native.
- [x] Add LEGACY_NEUTRAL_CONFIG entries and executable tests: clearing path avoids the contact
      radius, alignment gate and drive-through unchanged, `navigate`/`stop` untouched, legacy
      checkpoints load with the flag off, non-default flag rejects a legacy fingerprint.
- [x] Re-measure the finishing-angle probe at the config scale: 0° stays 1.00, ≥60° rises
      materially above zero (0.00 → 0.67 at 60°, 0.04 at 90°), dribbling unchanged.
- [x] Re-run `tools/audit_skill_difficulty.py` over `shot` and decide whether `ball_angle`
      becomes a declared axis. Decision: declared. The drill's angle span was restored to the
      cycle-4 range (heading_error 3-63°, heading-only rotation, so the cycle-4 overlap bug
      does not recur) and `FAMILY_AXES["shot"]` regains `ball_angle`. The action-set race
      (`tools/primitive_race.py angle` at the match scale) reads 0.70 0.70 0.55 0.50 0.40 for
      strike where cycle 5 measured 0.55 0.05 0.00 0.00 0.00; the audit reports no invalid or
      inverted axes and the other shot ladders survive (spawn_distance 0.70→0.05, ball_speed
      0.70→0.00 by the same race). The scripted probe cannot shoot, so the policy-side ladder
      is measured after the M24.6 run.
- [x] Give the fresh M24.6 run a new policy fingerprint and configuration; record the
      full-match gate signals against the M24.5 stall. Done in evidence.md: fingerprint
      `ebc5fb8b33c7499b`, run `vsss-m24-6-run-0002` (2,900 it / 47.5M steps, early-stopped
      on 12 consecutive holdout regressions while mid-`integration` phase). Semantic peak
      0.768 vs M24.5's 0.671; phase progression defense → integration (M24.5 stuck in
      defense); full-match draw 0.834 vs gate 0.70 — gate not crossed, trend still
      improving at stop. Rotations attempted (6/3 completed vs M24.5's 0).
- [x] Run lint, build, test, and a from-scratch smoke run before starting the new run.
      Rust 61 passed, pytest 330 passed, ruff and clippy clean; smoke
      `vsss-m24-6-run-0001` passed (25 it, curriculum `allocation_valid`).

## Narrowed during measurement

- The alignment gate and the exit direction are not the defect and are not changed: the gate
  correctly refuses a misaligned drive-through. The defect is the approach path, so only the
  approach path changes.
- The clearing waypoint sits on the exit line behind the ball (ball − 0.26 · exit), not
  perpendicular to it: the robot settles there clear of contact and turns onto the exit
  direction in place. The waypoint approach runs at 0.35 authority on both wheels — a
  forward-only cut of the arc passes 0.061 m from the ball (inside 0.082) and was rejected by
  measurement. The re-aimed turn at the acquisition keeps full turning authority; scaling it
  turns the release into a crawl.
- The probe's scale now reads from the match config (`max_wheel_speed = 30.0`); the legacy 12.0
  constant remains available as an argument and is documented as a pre-config stress value
  outside the acceptance scale.
- The `shot` drill's angle demand is the striker's heading misalignment at placement (the
  fixed 0.18 rad was inert: every primitive converted it at match speed, and the race showed
  no gradient). Re-widening to the cycle-4 span (3-63°) restores a real ladder; the generator
  revision is bumped to `m24.6-clearing-angle` so the old drill distribution cannot mix into
  the fresh curriculum state.
