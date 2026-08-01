# Tasks

- [x] Accept ADR 0019 before implementation begins.
- [x] Record the profile that motivates the work, including the physics share.
- [x] Add the crate that will hold the ported computation, depending on `vsss-spec` only.
- [x] Slice 1: observations. Native batch call, golden-equivalence test, throughput reported
      at 76x on the ported work.
- [x] Check whether role assignment is genuinely called twice per world per decision before
      porting it; a redundant call is cheaper to delete than to port. It is called twice and the
      duplication is load-bearing: the reward's call must stay stateless for the potential to be
      a function of the state, and the two disagree on 6.8 per cent of decisions.
- [x] Slice 2: role assignment, both the hysteretic and the stateless call, at 45.6x together.
- [x] Slice 3: the goal-geometry potential, asserted term by term against the decomposition.
- [x] Slice 4: idle-spin detection.
- [x] Slice 4b: contact and deadlock. Recorded first as already vectorized, which was wrong —
      the profile showed one call per world per decision. Ported after the correction.
- [x] Slice 4c: the scripted opponent, which the profile made the largest remaining stage at
      36 per cent once the listed slices were done. Not in the original list.
- [x] Slice 4d: the four per-world scalars — ball touch, closest distance, teammate congestion
      and defensive distance — computed in one pass rather than four crossings.
- [x] Slice 5: the free-ball restart. The state is now edited where it lives, with no JSON in
      the loop at all.
- [x] Slice 6, taken early: the action executor, at 107.7x. Reordered ahead of the reward
      because the stage profile put it at 17.8 per cent against the reward geometry's 15.0.
- [ ] Retire each Python reference only after its slice has been green for a full run.
- [ ] Re-measure the floor after every slice; the thirteen-times estimate is an extrapolation
      and should be replaced by measurement as it is approached.
