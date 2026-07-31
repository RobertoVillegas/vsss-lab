# Tasks

- [x] Accept ADR 0017 before implementation begins.
- [x] Sweep one axis at a time in the audit, with the others held easy.
- [x] Classify each cell: ramp, cliff, no-gradient, inverted, invalid, beyond-reference.
- [x] Support a capable probe, since a scripted controller can neither shoot nor pass.
- [x] Repair save_deflection so every angle is goal-bound by construction.
- [x] Repair shot so its easy end is a short range rather than on top of the defenders.
- [x] Aim the pass at its receiver and give it a speed floor that always arrives.
- [x] Make the rotation the support performs an axis instead of a constant.
- [x] Give approach a reach that spans easy to hard.
- [x] Bring interception and save_deflection ball speed into the interceptable range.
- [x] Declare the per-family axis map and advance only declared axes.
- [x] Record both probe matrices as evidence.
- [x] Complete local gates: full suite, `mise run lint`, OpenSpec strict, end-to-end smoke.
- [ ] `pass_receive.ball_angle` is a live cliff and `rotation_recovery.spawn_distance`
      declines noisily. Both are usable and neither is inverted, but they are the next two
      ladders to smooth.
- [ ] Decide whether `target_width` and `opponent_pressure` earn an effect or leave the
      declared space. They are currently declared only where measured.
- [ ] Re-baseline: every prior evaluation belongs to an earlier generator revision.
