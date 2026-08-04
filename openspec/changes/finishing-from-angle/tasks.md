 # Tasks

- [x] Re-measure the finishing-angle probe on current code: strike converts 1.00 on the line,
      0.50 at 30°, 0.00 from 60° and beyond; the run-0002 full-match gate stall is recorded.
- [x] Instrument one 60° attempt and locate the mechanism: the straight-line approach to the
      behind-ball acquisition point enters the ball's contact radius while the robot's heading is
      ~70° off the exit direction, and that first misaligned contact knocks the ball up the wing
      (ball-to-goal distance grows over the attempt).
- [ ] Accept ADR 0027 (the strike must arrive aligned, not merely near). The ADR is proposed;
      implementation starts once it is accepted.
- [ ] Implement the ball-clearing approach phase in `_strike_target`, behind
      `strike_clearing_enabled` / `strike_clearing_distance`.
- [ ] Add LEGACY_NEUTRAL_CONFIG entries and executable tests: clearing path avoids the contact
      radius, alignment gate and drive-through unchanged, `navigate`/`stop` untouched, legacy
      checkpoints load with the flag off, non-default flag rejects a legacy fingerprint.
- [ ] Re-measure the finishing-angle probe: 0° stays 1.00, ≥60° rises materially above zero,
      dribbling unchanged.
- [ ] Re-run `tools/audit_skill_difficulty.py` over `shot` and decide whether `ball_angle`
      becomes a declared axis.
- [ ] Give the fresh M24.6 run a new policy fingerprint and configuration; record the
      full-match gate signals against the M24.5 stall.
- [ ] Run lint, build, test, and a from-scratch smoke run before starting the new run.

## Narrowed during measurement

- The alignment gate and the exit direction are not the defect and are not changed: the gate
  correctly refuses a misaligned drive-through. The defect is the approach path, so only the
  approach path changes.
